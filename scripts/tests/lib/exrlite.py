"""
Minimal reader for UNCOMPRESSED scanline OpenEXR files (what Blender writes with exr_codec = 'NONE'),
including multilayer files. numpy only, so it runs locally (no OpenEXR module here) and inside Blender.

    ch = read_exr(path)            # {channel name: float32 array (H, W), top row first}
    rgb = stack(ch, 'ViewLayer.Combined')   # (H, W, 3)
"""
import struct
import numpy as np

_PT = {0: np.uint32, 1: np.float16, 2: np.float32}


def _cstr(buf, i):
    j = buf.index(b'\0', i)
    return buf[i:j].decode(), j + 1


def _header(buf, i):
    attrs = {}
    while buf[i] != 0:
        name, i = _cstr(buf, i)
        typ, i = _cstr(buf, i)
        size = struct.unpack_from('<i', buf, i)[0]
        i += 4
        attrs[name] = (typ, buf[i:i + size])
        i += size
    return attrs, i + 1


def _chlist(raw):
    chans, j = [], 0
    while raw[j] != 0:
        nm, j = _cstr(raw, j)
        pt = struct.unpack_from('<i', raw, j)[0]
        j += 16
        chans.append((nm, pt))
    return chans


def read_exr(path):
    """All channels of all parts: {name: float32 (H, W)}. Single- or multi-part, uncompressed scanline."""
    buf = open(path, 'rb').read()
    if buf[:4] != b'\x76\x2f\x31\x01':
        raise ValueError('not an EXR: ' + path)
    flags = struct.unpack_from('<I', buf, 4)[0]
    if flags & 0x200:
        raise ValueError('tiled EXR not supported')
    multi = bool(flags & 0x1000)
    i = 8
    parts = []
    while True:
        attrs, i = _header(buf, i)
        parts.append(attrs)
        if not multi or buf[i] == 0:
            break
    if multi:
        i += 1
    out = {}
    tables = []
    for attrs in parts:
        comp = attrs['compression'][1][0]
        if comp != 0:
            raise ValueError('only uncompressed EXR supported (compression=%d); save with exr_codec NONE' % comp)
        xmin, ymin, xmax, ymax = struct.unpack('<4i', attrs['dataWindow'][1])
        W, H = xmax - xmin + 1, ymax - ymin + 1
        n = struct.unpack('<i', attrs['chunkCount'][1])[0] if 'chunkCount' in attrs else H
        tables.append((attrs, W, H, ymin, np.frombuffer(buf, '<u8', n, i)))
        i += 8 * n
    for attrs, W, H, ymin, offsets in tables:
        chans = _chlist(attrs['channels'][1])
        for nm, _ in chans:
            out[nm] = np.zeros((H, W), np.float32)
        for off in offsets:
            o = int(off) + (4 if multi else 0)
            y = struct.unpack_from('<i', buf, o)[0] - ymin
            o += 8
            for nm, pt in chans:
                dt = np.dtype(_PT[pt]).newbyteorder('<')
                out[nm][y] = np.frombuffer(buf, dt, W, o).astype(np.float32)
                o += W * dt.itemsize
    return out


def stack(ch, prefix, comps='RGB'):
    keys = [k for k in ch if k.startswith(prefix + '.')] if prefix else list(ch)
    if not keys:
        raise KeyError('%s not in %s' % (prefix, sorted(ch)))
    if len(comps) == 1 or (prefix and (prefix + '.' + comps[0]) not in ch and len(keys) == 1):
        return ch[keys[0]]
    return np.stack([ch[prefix + '.' + c] for c in comps], -1)
