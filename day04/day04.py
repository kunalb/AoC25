import copy
import sys
from collections import defaultdict


def main():
    grid = set()
    lines = sys.stdin.readlines()
    for r, row in enumerate(lines):
        for c, col in enumerate(row.strip()):
            if col == '@':
                grid.add(r + c * 1j)
    cols = c
    rows = r

    box = [
        a + b * 1j
        for a in range(-1, 2)
        for b in range(-1, 2)
        if not (a == 0 and b == 0)
    ]

    removed = 0
    total_removed = 0
    first = True

    while removed or first:
        removed = 0
        counts = defaultdict(lambda: 0)
        next_grid = set()
        for pt in grid:
            count = sum(1 if (pt + b) in grid else 0 for b in box)
            if count < 4:
                removed += 1
            else:
                next_grid.add(pt)

        if first:
            first = False
            print(removed)

        total_removed += removed
        grid = next_grid

    print(total_removed)


if __name__ == "__main__":
    main2()
