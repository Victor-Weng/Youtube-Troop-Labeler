"""youtube_url_generation.py

Simple, robust script to extract every video URL from a YouTube playlist
into a text file (one URL per line) using yt-dlp.

Features:
 - Uses yt-dlp Python API (no video downloads)
 - Handles large playlists (900+ entries)
 - Skips unavailable/removed/private videos safely
 - Optional cookies.txt support for private/unlisted access
 - De-duplicates while preserving playlist order
 - Retries & ignores transient extraction errors
 - Idempotent: re-running overwrites output file cleanly

Usage:
  python youtube_url_generation.py PLAYLIST_URL
  python youtube_url_generation.py PLAYLIST_URL --cookies cookies.txt
  python youtube_url_generation.py PLAYLIST_URL --output my_urls.txt

You can also set PLAYLIST_URL env var and omit it on the command line.

Requirements:
  pip install yt-dlp
  (cookies optional) Export browser cookies to cookies.txt if needed.
"""

from __future__ import annotations

import argparse
import os
import sys
import signal
from typing import List, Set

try:
    from yt_dlp import YoutubeDL
except ImportError:
    print("yt-dlp not installed. Install with: pip install yt-dlp", file=sys.stderr)
    sys.exit(1)


def build_ydl_opts(cookiefile: str | None) -> dict:
    """Return yt-dlp options tuned for fast playlist metadata extraction only."""
    # Playlist reverse flag is intentionally ignored here; ordering control
    # now happens when consuming youtube_urls.txt (not during extraction).
    opts = {
        # Don't download media
        "skip_download": True,
        # Flatten to avoid per-video secondary extraction passes (faster)
        "extract_flat": True,
        # Be quiet except for warnings/errors we print ourselves
        "quiet": True,
        "no_warnings": True,
        # Continue on errors (missing / private videos)
        "ignoreerrors": True,
        # Retry network hiccups
        "retries": 5,
        "fragment_retries": 5,
        # Keep original playlist order during extraction
        "playlistreverse": False,
    }
    if cookiefile:
        opts["cookiefile"] = cookiefile
    return opts


def extract_playlist_urls(playlist_url: str, cookiefile: str | None) -> List[str]:
    """Extract full video URLs from a YouTube playlist.

    Returns a list of https://www.youtube.com/watch?v=... links in original order.
    """
    ydl_opts = build_ydl_opts(cookiefile)
    urls: List[str] = []
    seen: Set[str] = set()

    def try_extract(url: str) -> List[str]:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return []
            entries = info.get("entries") or []
            out = []
            for entry in entries:
                if not entry:
                    continue
                video_id = entry.get("id")
                if not video_id:
                    raw_url = entry.get("url")
                    if raw_url and raw_url.startswith("http"):
                        full_url = raw_url
                    else:
                        continue
                else:
                    full_url = f"https://www.youtube.com/watch?v={video_id}"
                if full_url not in seen:
                    seen.add(full_url)
                    out.append(full_url)
            return out

    # Try original URL first
    urls = try_extract(playlist_url)
    if not urls:
        # Try canonical playlist URL if original failed
        import re
        match = re.search(r"list=([A-Za-z0-9_-]+)", playlist_url)
        if match:
            canonical_url = f"https://www.youtube.com/playlist?list={match.group(1)}"
            urls = try_extract(canonical_url)
    return urls


def write_urls(urls: List[str], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all video URLs from a YouTube playlist into a text file.")
    parser.add_argument("playlist", nargs="?",
                        help="YouTube playlist URL (or set PLAYLIST_URL env var; or edit PLAYLIST_URL at top of script)")
    parser.add_argument("--cookies", dest="cookies",
                        help="Path to cookies.txt (optional)")
    parser.add_argument("--output", dest="output", default="youtube_urls.txt",
                        help="Output filename (default: youtube_urls.txt)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    playlist_url = args.playlist or os.getenv("PLAYLIST_URL")
    if not playlist_url:
        print("Error: No playlist URL provided. Pass as argument or set PLAYLIST_URL env var.", file=sys.stderr)
        return 2

    if args.cookies and not os.path.isfile(args.cookies):
        print(
            f"Warning: cookies file '{args.cookies}' not found, continuing without it.", file=sys.stderr)
        args.cookies = None

    print(f"Extracting playlist: {playlist_url}")
    if args.cookies:
        print(f"Using cookies: {args.cookies}")
    print("Working... (this may take a while for large playlists)")

    try:
        urls = extract_playlist_urls(playlist_url, args.cookies)
    except KeyboardInterrupt:
        print("\nInterrupted. Partial results will not be saved.")
        return 130
    except Exception as e:
        print(f"Failed to extract playlist: {e}", file=sys.stderr)
        return 1

    if not urls:
        print("No URLs extracted (playlist empty or inaccessible).", file=sys.stderr)
        return 3

    write_urls(urls, args.output)
    print(f"Done. Extracted {len(urls)} unique video URLs -> {args.output}")
    return 0


def _install_sigint_handler():
    def handler(signum, frame):  # noqa: ARG001
        print("\nSIGINT received, aborting...", file=sys.stderr)
        sys.exit(130)
    signal.signal(signal.SIGINT, handler)


if __name__ == "__main__":
    # === SET YOUR PLAYLIST URL HERE ===
    PLAYLIST_URL = "https://www.youtube.com/watch?v=rw6Uz5XKxf8&list=PLEA_uuAasSWZwfaLsvQEnSjb3o8pAND-2&ab_channel=TVroyale"

    if not os.getenv("PLAYLIST_URL"):
        os.environ["PLAYLIST_URL"] = PLAYLIST_URL
    _install_sigint_handler()
    raise SystemExit(main())
