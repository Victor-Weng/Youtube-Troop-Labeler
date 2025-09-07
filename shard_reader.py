"""
Shard Reader / Interactive Reviewer
===================================

Purpose:
    Inspect dataset shards produced by DatasetSaver (WebP + JSON pairs inside .tar files),
    visualize bounding boxes, and optionally prune bad frames interactively.

Two Modes:
    1. Sequential Display (--show)
             Quickly cycle through up to --max frames (ESC stops early). Optionally export overlays.
    2. Interactive Review (--interactive)
             Step frame-by-frame, choose to keep or delete. Deletions rewrite the shard in-place
             (atomic replace using a temporary tar). Overlay shows progress and removals.

Flags:
    --dataset      Root dataset directory containing 'shards/' and 'manifest.csv'. Default: output_dataset
    --shard        Zero-based shard index (sorted lexicographically). Default: 0
    --max          Maximum frames to process/display (caps iteration in both modes). Default: 10
    --output       Optional directory: save each annotated frame as PNG (kept regardless of interactive deletion)
    --show         Enable non-interactive quick viewing (press ESC to stop early)
    --interactive  Enable interactive review controls (mutually exclusive intent with --show; if both,
                                 interactive logic takes precedence)

Interactive Key Bindings (WASD style):
    d : Next frame (keep current state)
    a : Previous frame (if available)
    w : Mark current frame for deletion AND advance to next (idempotent)
    s : Restore (unmark) current frame if it was previously marked deleted (stays on frame)
    q / ESC : Quit review and apply deletions
    (other keys ignored)

Deletion Mechanics:
    - Mark/unmark via w (mark) and s (restore). Deletions are not applied until exit.
    - Removal set contains base names (e.g., frame_000123). On quit, shard is rewritten excluding marked frames.
    - Manifest is NOT updated automatically (future enhancement possible).

Safeguards / Notes:
    - Only the specified shard file is rewritten; other shards untouched.
    - JSON missing for a frame silently skips that frame.
    - Frames with decode errors are skipped automatically.
    - If --max < total frames, only the first --max frames are offered for review.

Examples (PowerShell):
    python shard_reader.py --dataset output_dataset --shard 0 --show --max 50 --output reviewed_pngs
    python shard_reader.py --dataset output_dataset --shard 0 --interactive 

Planned Extensions (not implemented yet):
    - Optional manifest pruning / regeneration
    - Player / card-type filtering
    - Export approved sequence to video
    - Bulk multi-shard sampling
"""

import os, tarfile, json, io, cv2, shutil, tempfile
import argparse


def list_shards(dataset_dir):
    shard_dir = os.path.join(dataset_dir, 'shards')
    if not os.path.isdir(shard_dir):
        raise SystemExit(f"Shard directory not found: {shard_dir}")
    shards = sorted([f for f in os.listdir(shard_dir) if f.endswith('.tar')])
    return shard_dir, shards

def open_shard(shard_dir, shard_index, shards):
    if shard_index < 0 or shard_index >= len(shards):
        raise SystemExit(f"Shard index {shard_index} out of range (0..{len(shards)-1})")
    path = os.path.join(shard_dir, shards[shard_index])
    return path

def decode_webp(buf):
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

import numpy as np

def draw_frame(image, meta):
    img = image.copy()
    for t in meta.get('tracks', []):
        x, y, w, h = t['x'], t['y'], t['w'], t['h']
        card = t.get('card_type', 'Unknown')
        player = t.get('player', 'Unknown')
        color = (0,255,0) if player.lower()=='ally' else (0,128,255)
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        label = f"T{t['track_id']} {player}:{card}"
        cv2.putText(img, label, (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(img, f"Frame {meta.get('frame_number')} Active {len(meta.get('tracks', []))}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
    return img

def rewrite_tar_without(shard_path, remove_frames):
    """Rewrite the tar excluding given frame base names (e.g. frame_000123)."""
    if not remove_frames:
        return
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.tar', prefix='tmp_shard_')
    os.close(tmp_fd)
    keep = []
    with tarfile.open(shard_path, 'r') as tar:
        for m in tar.getmembers():
            base = os.path.splitext(m.name)[0]
            if base in remove_frames:
                continue
            keep.append(m)
        with tarfile.open(tmp_path, 'w') as out_tar:
            for m in keep:
                f = tar.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                info = tarfile.TarInfo(name=m.name)
                info.size = len(data)
                info.mtime = m.mtime
                out_tar.addfile(info, io.BytesIO(data))
    shutil.move(tmp_path, shard_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='output_dataset', help='Dataset root (contains shards/, manifest.csv)')
    ap.add_argument('--shard', type=int, default=0, help='Shard index to inspect')
    ap.add_argument('--max', type=int, default=10, help='Max frames to visualize from shard')
    ap.add_argument('--output', default=None, help='Optional directory to save annotated PNGs')
    ap.add_argument('--show', action='store_true', help='Display images in a window (sequential non-interactive)')
    ap.add_argument('--interactive', action='store_true', help='Interactive review: SPACE=keep & next, D=delete, A=previous, Q/ESC=quit')
    args = ap.parse_args()

    shard_dir, shards = list_shards(args.dataset)
    if not shards:
        print('No shards found.')
        return
    shard_path = open_shard(shard_dir, args.shard, shards)
    print(f"Reading shard: {os.path.basename(shard_path)} ({args.shard}/{len(shards)-1})")

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Gather frame members first
    with tarfile.open(shard_path, 'r') as tar:
        members = tar.getmembers()
        webps = sorted([m for m in members if m.name.endswith('.webp')], key=lambda m: m.name)
        frame_entries = []  # list of (frame_base, webp_member, json_member)
        for m in webps:
            frame_base = os.path.splitext(m.name)[0]
            json_name = frame_base + '.json'
            try:
                json_member = tar.getmember(json_name)
            except KeyError:
                json_member = None
            frame_entries.append((frame_base, m, json_member))

    # Non-interactive sequential mode
    if args.show and not args.interactive:
        shown = 0
        with tarfile.open(shard_path, 'r') as tar:
            for frame_base, w_m, j_m in frame_entries:
                if shown >= args.max:
                    break
                if not j_m:
                    continue
                webp_f = tar.extractfile(w_m); json_f = tar.extractfile(j_m)
                img_bytes = webp_f.read(); meta = json.loads(json_f.read().decode('utf-8'))
                img = decode_webp(img_bytes)
                if img is None: continue
                annotated = draw_frame(img, meta)
                if args.output:
                    out_path = os.path.join(args.output, f"{frame_base}.png")
                    cv2.imwrite(out_path, annotated)
                cv2.imshow('shard_reader', annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                shown += 1
        cv2.waitKey(0); cv2.destroyAllWindows()
        print(f"Processed {shown} frames from shard {args.shard}")
        return

    if args.interactive:
        idx = 0
        removed = set()
        total = min(len(frame_entries), args.max) if args.max>0 else len(frame_entries)

        def load(i):
            with tarfile.open(shard_path, 'r') as tar:
                frame_base, w_m, j_m = frame_entries[i]
                wf = tar.extractfile(w_m)
                jf = tar.extractfile(j_m) if j_m else None
                img = decode_webp(wf.read()) if wf else None
                meta = json.loads(jf.read().decode('utf-8')) if jf else {'tracks':[], 'frame_number': frame_base}
                return frame_base, img, meta

        while 0 <= idx < len(frame_entries) and idx < total:
            frame_base, img, meta = load(idx)
            if img is None:
                idx += 1
                continue
            ann = draw_frame(img, meta)
            status_marked = frame_base in removed
            instr = f"[{idx+1}/{total}] d=next a=prev w=delete+next s=restore q/ESC=quit  Del:{len(removed)}"
            cv2.putText(ann, instr, (5, ann.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,200,255),1, cv2.LINE_AA)
            if status_marked:
                # Top-right marker
                h, w_img = ann.shape[:2]
                label = "MARKED"
                cv2.putText(ann, label, (w_img-10-80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2, cv2.LINE_AA)
            cv2.imshow('shard_review', ann)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('d'):
                idx += 1
            elif key == ord('a'):
                if idx > 0:
                    idx -= 1
            elif key == ord('w'):
                # mark & advance
                removed.add(frame_base)
                idx += 1
            elif key == ord('s'):
                # restore (unmark)
                removed.discard(frame_base)
            else:
                # ignore other keys
                pass
        cv2.destroyAllWindows()
        if removed:
            print(f"Rewriting shard without {len(removed)} frames ...")
            rewrite_tar_without(shard_path, removed)
            print("Rewrite complete.")
        print(f"Interactive review finished. Removed={len(removed)} kept={total-len(removed)}")
        return

    # Fallback: just list frame count
    print(f"Shard contains {len(frame_entries)} frames. Use --show or --interactive to view.")

if __name__ == '__main__':
    main()
