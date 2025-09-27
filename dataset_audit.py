import os, tarfile, json, math, collections
from io import BytesIO

# Optional perceptual hash (install pillow & imagehash for better accuracy)
try:
    from PIL import Image
    import imagehash
    def phash_bytes(data: bytes):
        return str(imagehash.phash(Image.open(BytesIO(data)).convert("RGB")))
except Exception:
    # Fallback tiny average hash (16x16)
    from PIL import Image
    def phash_bytes(data: bytes):
        img = Image.open(BytesIO(data)).convert("L").resize((16,16))
        pixels = list(img.getdata())
        avg = sum(pixels)/len(pixels)
        bits = ''.join('1' if p>avg else '0' for p in pixels)
        # compress to hex
        return hex(int(bits, 2))[2:]

SHARD_DIR = os.path.join("output_dataset", "shards")
LIMIT_IMAGES = None  # set e.g. 5000 to sample
PRINT_TOP_N = 25

class_counts = collections.Counter()
hash_counts = collections.Counter()
frames_processed = 0
duplicates = 0

def process_tar(path):
    global frames_processed, duplicates
    with tarfile.open(path, 'r') as tf:
        # Collect json + webp members by stem
        members = tf.getmembers()
        json_members = [m for m in members if m.name.endswith(".json")]
        for jm in json_members:
            if LIMIT_IMAGES and frames_processed >= LIMIT_IMAGES:
                return
            f = tf.extractfile(jm)
            if not f:
                continue
            meta = json.load(f)
            # Count classes (card_type per track)
            for t in meta.get("tracks", []):
                ct = t.get("card_type", "Unknown")
                class_counts[ct] += 1
            # Hash corresponding image if present
            stem = jm.name[:-5]  # drop .json
            webp_name = stem + ".webp"
            try:
                im_member = tf.getmember(webp_name)
                im_file = tf.extractfile(im_member)
                if im_file:
                    data = im_file.read()
                    h = phash_bytes(data)
                    if hash_counts[h] > 0:
                        duplicates += 1
                    hash_counts[h] += 1
            except KeyError:
                pass
            frames_processed += 1

def main():
    shards = sorted([os.path.join(SHARD_DIR, f) for f in os.listdir(SHARD_DIR) if f.endswith(".tar")])
    for s in shards:
        process_tar(s)
        if LIMIT_IMAGES and frames_processed >= LIMIT_IMAGES:
            break

    unique_images = len(hash_counts)
    duplicate_rate = (duplicates / frames_processed)*100 if frames_processed else 0.0

    print(f"Frames analyzed: {frames_processed}")
    print(f"Unique perceptual hashes: {unique_images}")
    print(f"Duplicate frames (hash seen before): {duplicates} ({duplicate_rate:.2f}%)")
    print(f"Classes observed: {len(class_counts)}")
    print("Top classes:")
    total_labels = sum(class_counts.values())
    for cls, cnt in class_counts.most_common(PRINT_TOP_N):
        pct = (cnt/total_labels)*100 if total_labels else 0
        print(f"  {cls:20s} {cnt:6d} ({pct:5.2f}%)")
    # Simple long tail hint
    rare = [c for c,n in class_counts.items() if n < 50]
    if rare:
        print(f"Rare classes (<50 samples): {len(rare)} -> {', '.join(list(rare)[:15])}{' ...' if len(rare)>15 else ''}")

if __name__ == "__main__":
    main()