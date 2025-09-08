import os
import tarfile
import io
from collections import deque
from typing import List, Dict, Set
import cv2
import json


class DatasetSaver:
    """Handles 2-frame delayed saving of frames with active tracks into WebP + tar shards."""

    def __init__(self, output_dir: str = 'dataset', shard_max_images: int = 2000, webp_quality: int = 92, delay_frames: int = 2, tracking_region=None):
        self.output_dir = output_dir
        self.shard_dir = os.path.join(output_dir, 'shards')
        os.makedirs(self.shard_dir, exist_ok=True)
        self.manifest_path = os.path.join(output_dir, 'manifest.csv')
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'w', encoding='utf-8') as f:
                f.write('frame_number,shard,file,tracks,json_meta\n')
        self.buffer = deque()  # stores recent frames for delay
        self.delay_frames = delay_frames
        self.webp_quality = webp_quality
        self.shard_max_images = shard_max_images
        # Determine starting shard index by scanning existing shard files
        existing = [f for f in os.listdir(self.shard_dir) if f.startswith(
            'data_') and f.endswith('.tar')]
        max_index = -1
        for name in existing:
            try:
                idx = int(name.split('_')[1].split('.')[0])
                if idx > max_index:
                    max_index = idx
            except Exception:
                continue
        self.current_shard_index = max_index + 1  # next index (0 if none)
        self.current_shard_count = 0
        self.tar = None
        self.tracking_region = tracking_region  # (x,y,w,h) crop before saving
        self._open_new_shard()
        # Stats
        self.total_images_saved = 0
        self._last_progress_len = 0

    def _print_progress(self):
        """Render an inline shard progress bar."""
        try:
            progress = self.current_shard_count / \
                float(self.shard_max_images) if self.shard_max_images else 0
            bar_len = 30
            filled = int(bar_len * progress)
            bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'
            msg = f"Shard {self.current_shard_index-1:05} {bar} {self.current_shard_count}/{self.shard_max_images} Total:{self.total_images_saved}"
            print('\r' + msg + ' ' * max(0, self._last_progress_len -
                  len(msg)), end='', flush=True)
            self._last_progress_len = len(msg)
        except Exception:
            pass

    def _open_new_shard(self):
        if self.tar is not None:
            self.tar.close()
            # Finish line for previous shard progress
            try:
                print()
            except Exception:
                pass
        shard_name = f"data_{self.current_shard_index:05}.tar"
        shard_path = os.path.join(self.shard_dir, shard_name)
        self.tar = tarfile.open(shard_path, 'w')
        self.current_shard_count = 0
        self.current_shard_name = shard_name
        self.current_shard_index += 1
        # Write checkpoint whenever a new shard is opened
        try:
            self._write_checkpoint()
        except Exception:
            pass

    # Video progress context (set externally)
    current_video_index = 0
    current_video_url = None

    def set_video_context(self, video_index: int, video_url: str):
        self.current_video_index = video_index
        self.current_video_url = video_url
        # Update checkpoint with new video context
        try:
            self._write_checkpoint()
        except Exception:
            pass

    def _checkpoint_path(self):
        return os.path.join(self.output_dir, 'checkpoint.json')

    def _write_checkpoint(self):
        data = {
            'video_index': getattr(self, 'current_video_index', 0),
            'video_url': getattr(self, 'current_video_url', None),
            'next_shard_index': self.current_shard_index,
            'current_shard_name': getattr(self, 'current_shard_name', None)
        }
        import json as _json
        with open(self._checkpoint_path(), 'w', encoding='utf-8') as f:
            _json.dump(data, f, indent=2)

    def _write_to_tar(self, filename: str, data: bytes):
        info = tarfile.TarInfo(name=filename)
        info.size = len(data)
        self.tar.addfile(info, io.BytesIO(data))

    def _encode_webp(self, image) -> bytes:
        ok, enc = cv2.imencode(
            '.webp', image, [cv2.IMWRITE_WEBP_QUALITY, self.webp_quality])
        if not ok:
            return b''
        return enc.tobytes()

    def _maybe_rotate_shard(self):
        if self.current_shard_count >= self.shard_max_images:
            self._open_new_shard()

    def _commit_frame(self, record: Dict):
        img = record['image']
        # Optional crop
        crop_offset = (0, 0)
        if self.tracking_region is not None:
            tx, ty, tw, th = self.tracking_region
            img = img[ty:ty+th, tx:tx+tw]
            crop_offset = (tx, ty)
        image_bytes = self._encode_webp(img)
        if not image_bytes:
            return
        frame_number = record['frame_number']
        webp_name = f"frame_{frame_number:06}.webp"
        # Filter boxes to only effective track ids
        eff_ids = record['effective_tracks']
        eff_boxes = [b for b in record.get(
            'boxes', []) if b['track_id'] in eff_ids]
        # Adjust boxes to be relative to crop if needed
        rel_boxes = []
        ox, oy = crop_offset
        for b in eff_boxes:
            rx = b['x'] - ox
            ry = b['y'] - oy
            # Clamp to image bounds after crop (avoid negatives / overflow)
            h_img, w_img = img.shape[:2]
            rw = b['w']
            rh = b['h']
            if rx < 0:
                rw += rx  # reduce width
                rx = 0
            if ry < 0:
                rh += ry
                ry = 0
            if rx + rw > w_img:
                rw = max(0, w_img - rx)
            if ry + rh > h_img:
                rh = max(0, h_img - ry)
            if rw <= 0 or rh <= 0:
                continue
            rel_boxes.append({
                'track_id': b['track_id'],
                'x': int(rx), 'y': int(ry), 'w': int(rw), 'h': int(rh),
                'player': b.get('player', 'Unknown'),
                'card_type': b.get('card_type', 'Unknown')
            })
        meta = {
            'frame_number': frame_number,
            'game_active': record['game_active'],
            'tracks': rel_boxes if self.tracking_region is not None else [
                {
                    'track_id': b['track_id'],
                    'x': b['x'], 'y': b['y'], 'w': b['w'], 'h': b['h'],
                    'player': b.get('player', 'Unknown'),
                    'card_type': b.get('card_type', 'Unknown')
                } for b in eff_boxes
            ]
        }
        json_name = f"frame_{frame_number:06}.json"
        self._write_to_tar(webp_name, image_bytes)
        self._write_to_tar(json_name, json.dumps(meta).encode('utf-8'))
        with open(self.manifest_path, 'a', encoding='utf-8') as f:
            # manifest lists track ids (post-invalidation)
            track_id_list = '|'.join(str(t['track_id'])
                                     for t in meta['tracks'])
            f.write(
                f"{frame_number},{self.current_shard_name},{webp_name};{json_name},{track_id_list},{json_name}\n")
        self.current_shard_count += 1
        self._maybe_rotate_shard()
        # Update global counts & show progress bar
        self.total_images_saved += 1
        self._print_progress()

    def handle_frame(self, frame_number: int, frame, game_active: bool, active_tracks: List, removed_tracks: List):
        """Add frame to buffer, apply invalidations, and commit frames older than delay."""
        # Snapshot track ids & boxes now
        track_ids = set()
        boxes = []
        for t in active_tracks:
            if not t.positions:
                continue
            last = t.positions[-1]
            track_ids.add(t.track_id)
            boxes.append({'track_id': t.track_id, 'x': last['x'], 'y': last['y'],
                         'w': last['w'], 'h': last['h'], 'card_type': t.card_type, 'player': t.player})
        record = {
            'frame_number': frame_number,
            'image': frame.copy(),  # copy to decouple
            'game_active': game_active,
            'tracks': track_ids,
            'invalidated': set(),
            'removed_tracks_meta': [],
            'boxes': boxes
        }
        self.buffer.append(record)

        # Apply invalidations for removed tracks (each is (track_id, last_seen_frame))
        for track_id, last_seen in removed_tracks:
            for r in self.buffer:
                if r['frame_number'] in (last_seen, last_seen - 1):
                    if track_id in r['tracks']:
                        r['invalidated'].add(track_id)
                        r['removed_tracks_meta'].append(track_id)

        # Pop frames older than delay
        while self.buffer and (frame_number - self.buffer[0]['frame_number']) >= self.delay_frames:
            oldest = self.buffer.popleft()
            effective_tracks = oldest['tracks'] - oldest['invalidated']
            oldest['effective_tracks'] = effective_tracks
            if oldest['game_active'] and len(effective_tracks) > 0:
                self._commit_frame(oldest)

    def close(self):
        if self.tar is not None:
            self.tar.close()
        # Final checkpoint on close
        try:
            self._write_checkpoint()
        except Exception:
            pass
