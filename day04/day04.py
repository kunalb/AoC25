import copy
import sys
from collections import defaultdict


def main():
    grid = sys.stdin.read().strip()
    grid = list(map(list, grid.split()))

    total_removed = 0
    while (removed := solve(grid)):
        print(removed)
        total_removed += removed

    print(total_removed)


def solve(grid):

    counts = 0
    count_grid = []
    for _ in range(0, len(grid)):
        count_grid.append([0] * len(grid[0]))

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] != '@':
                continue

            for x in range (-1, 2):
                for y in range(-1, 2):
                    p = x + i
                    q = y + j
                    if p < 0 or q < 0:
                        continue

                    if p >= len(grid) or q >= len(grid[i]):
                        continue

                    if x == 0 and y == 0:
                        continue

                    count_grid[p][q] += 1

    total = 0
    for i, row in enumerate(count_grid):
        for j, col in enumerate(row):
            if col < 4 and grid[i][j] == '@':
                grid[i][j] = "."
                total += 1

    return total


if __name__ == "__main__":
    main()
