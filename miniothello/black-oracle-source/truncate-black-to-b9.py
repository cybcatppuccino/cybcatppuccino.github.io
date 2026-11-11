#!/usr/bin/env python3
"""Preserve the first 9 Black decision layers (B9W8, 17-ply) of a full Black LOUDS book.

Run on a *full* B10W9 tree, not on a previously truncated tree. Data bytes in
retained nodes are preserved identically. Requires only Python standard library.
"""
from pathlib import Path
import argparse
import json


def truncate(tree: bytes, data: bytes, layers: int = 9):
    if layers < 1 or len(tree) % 64 or len(data) % 2:
        raise ValueError('Invalid number of layers or input alignment')
    size = len(tree) * 8
    def bit(pos):
        if pos >= size: raise ValueError('LOUDS ended unexpectedly')
        return (tree[pos >> 3] >> (pos & 7)) & 1
    if bit(0) != 1 or bit(1) != 0: raise ValueError('Invalid LOUDS dummy root')
    output = bytearray()
    written = 0
    def emit(value):
        nonlocal written
        if written % 8 == 0: output.append(0)
        if value: output[written >> 3] |= 1 << (written & 7)
        written += 1
    emit(1)
    emit(0)
    pos, count, retained = 2, 1, 0
    counts = []
    for level in range(layers):
        counts.append(count)
        next_count = 0
        for _ in range(count):
            children = 0
            while bit(pos):
                pos += 1
                children += 1
            pos += 1
            if level < layers - 1:
                for _ in range(children): emit(1)
                next_count += children
            emit(0)
        retained += count
        if retained > len(data) // 2: raise ValueError('Node count exceeds data')
        count = next_count
    while len(output) % 64: output.append(0)
    if sum(b.bit_count() for b in output) != retained: raise ValueError('Inconsistent rank count')
    return bytes(output), data[:retained * 2], counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets', type=Path, default=Path(__file__).resolve().parent.parent / 'assets')
    args = ap.parse_args()
    tree_path = args.assets / 'black-opening-tree.dat'
    data_path = args.assets / 'black-opening-data.dat'
    tree, data, counts = truncate(tree_path.read_bytes(), data_path.read_bytes())
    tree_path.write_bytes(tree)
    data_path.write_bytes(data)
    print(json.dumps({'black_layers':9, 'white_layers':8, 'counts':counts, 'nodes':len(data)//2,
      'tree_bytes':len(tree), 'data_bytes':len(data)}, indent=2))

if __name__ == '__main__': main()
