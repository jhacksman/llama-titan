#!/usr/bin/env python3
"""Generate icon.icns for Madrone Way.app — pure stdlib (zlib PNG writer).

Draws a rounded-square sunset-over-the-hill icon: gradient sky, sun disc,
and a dark forested ridge (the McDonald Forest skyline), then packs PNGs
into the .icns container (modern icns types embed PNG directly).
"""

import math
import struct
import sys
import zlib
from pathlib import Path


def write_png(path_or_none, w, h, rgba_rows):
    raw = b"".join(b"\x00" + bytes(row) for row in rgba_rows)
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    if path_or_none:
        Path(path_or_none).write_bytes(png)
    return png


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def render(size):
    s = size
    cx_sun, cy_sun, r_sun = 0.5 * s, 0.46 * s, 0.17 * s
    corner = 0.225 * s  # macOS-style rounded square
    top, mid, bot = (12, 22, 58), (235, 120, 60), (255, 200, 90)

    rows = []
    for y in range(s):
        row = bytearray()
        ty = y / s
        sky = lerp(top, mid, min(1, ty * 1.6)) if ty < 0.62 else lerp(mid, bot, (ty - 0.62) / 0.38)
        for x in range(s):
            # rounded-rect alpha mask
            dx = max(corner - x, x - (s - 1 - corner), 0)
            dy = max(corner - y, y - (s - 1 - corner), 0)
            d = math.hypot(dx, dy)
            if d > corner:
                row += b"\x00\x00\x00\x00"
                continue
            a = 255 if d < corner - 1.5 else int(255 * (corner - d) / 1.5)

            r, g, b = sky
            # sun disc with soft edge
            ds = math.hypot(x - cx_sun, y - cy_sun)
            if ds < r_sun * 1.25:
                t = max(0.0, min(1.0, (r_sun - ds) / (r_sun * 0.18) + 0.5))
                r, g, b = lerp((r, g, b), (255, 244, 200), t)
            # forested ridge silhouette
            ridge = 0.66 + 0.05 * math.sin(x / s * 6.2) + 0.03 * math.sin(x / s * 17 + 1.4)
            if ty > ridge:
                depth = min(1.0, (ty - ridge) * 6)
                r, g, b = lerp((30, 54, 38), (12, 24, 18), depth)
            row += bytes((r, g, b, a))
        rows.append(row)
    return rows


def scale_box(rows, src, dst):
    """Box-sample RGBA rows from src×src down to dst×dst."""
    k = src // dst
    out = []
    for y in range(dst):
        row = bytearray()
        for x in range(dst):
            acc = [0, 0, 0, 0]
            for yy in range(k):
                r = rows[y * k + yy]
                for xx in range(k):
                    o = (x * k + xx) * 4
                    for c in range(4):
                        acc[c] += r[o + c]
            row += bytes(v // (k * k) for v in acc)
        out.append(row)
    return out


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "icon.icns"
    base = 1024
    rows1024 = render(base)
    sizes = {  # icns type -> pixel size
        b"ic10": 1024, b"ic09": 512, b"ic08": 256, b"ic07": 128,
    }
    blobs = []
    for typ, px in sizes.items():
        rows = rows1024 if px == base else scale_box(rows1024, base, px)
        png = write_png(None, px, px, rows)
        blobs.append(typ + struct.pack(">I", len(png) + 8) + png)
    body = b"".join(blobs)
    out.write_bytes(b"icns" + struct.pack(">I", len(body) + 8) + body)
    print(f"wrote {out} ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
