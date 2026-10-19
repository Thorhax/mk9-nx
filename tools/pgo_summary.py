#!/usr/bin/env python3
"""Fills in the object summary of PGO .gcda files merged from the console's streams (tools/win/pgo_merge.ps1).

The instrumented NRO streams its counts with __gcov_info_to_gcda (app/src/sa2_pgo.cpp), which writes every object's
OBJECT_SUMMARY as zero (runs = 0, sum_max = 0). libgcov's own exit dump writes runs = 1 and sum_max = the largest arc
count of the whole program, and GCC's -fprofile-use derives its hot / never-executed thresholds from them (hot:
count >= sum_max / hot-bb-count-fraction; never executed: count * unlikely-bb-count-fraction < runs). With zeros every
block would count as hot. This writes runs = <sessions> and sum_max = the largest arc count over every file, as one
libgcov run per session would have (across sessions sum_max is approximated by the merged maximum).

  python tools/pgo_summary.py <build folder> <sessions>

Format (GCC 12+, little-endian target): magic, version, stamp, checksum, then records of tag, length in bytes, data;
OBJECT_SUMMARY = tag 0xa1000000 (runs, sum_max: two u32), arc counters = tag 0x01a10000 (u64 each, low word first).
"""
import os
import struct
import sys

TAG_SUMMARY = 0xA1000000
TAG_ARCS = 0x01A10000
MAGIC = 0x67636461  # 'gcda'


def records(data):
    if len(data) < 16 or struct.unpack_from('<I', data, 0)[0] != MAGIC:
        raise ValueError('not a little-endian gcda file')
    pos = 16
    while pos + 8 <= len(data):
        tag, length = struct.unpack_from('<Ii', data, pos)
        if length < 0:  # counters that are all zero: their count, negated, and no data
            yield tag, pos + 8, 0
            pos += 8
            continue
        yield tag, pos + 8, length
        pos += 8 + length


def main():
    folder, sessions = sys.argv[1], int(sys.argv[2])
    files = [os.path.join(d, f) for d, _, fs in os.walk(folder) for f in fs if f.endswith('.gcda')]
    top = 0
    for path in files:
        data = open(path, 'rb').read()
        for tag, start, length in records(data):
            if tag == TAG_ARCS:
                for i in range(length // 8):
                    top = max(top, struct.unpack_from('<Q', data, start + 8 * i)[0])
    sum_max = min(top, 0xFFFFFFFF)
    patched = 0
    for path in files:
        data = bytearray(open(path, 'rb').read())
        for tag, start, length in records(data):
            if tag == TAG_SUMMARY and length >= 8:
                struct.pack_into('<II', data, start, sessions, sum_max)
                patched += 1
                break
        open(path, 'wb').write(data)
    print(f'{len(files)} .gcda files: summary runs = {sessions}, sum_max = {sum_max} (largest arc count {top}); '
          f'{patched} patched')


if __name__ == '__main__':
    main()
