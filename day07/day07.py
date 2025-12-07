import sys


def main():
    start = True
    l = 0
    splits = 0
    row_paths = []

    for row in sys.stdin:
        if not row.strip():
            continue

        if start:
            beams = {row.index("S")}
            l = len(row)
            row_paths = [0] * l
            row_paths[row.index("S")] = 1
            start = False
            continue

        next_beams = set()
        next_row_paths = [0] * l
        for i in beams:
            if row[i] == '^':
                splits += 1
                if i + 1 < l:
                    next_beams.add(i + 1)
                    next_row_paths[i + 1] += row_paths[i]
                if i - 1 >= 0:
                    next_beams.add(i - 1)
                    next_row_paths[i - 1] += row_paths[i]
            else:
                next_beams.add(i)
                next_row_paths[i] += row_paths[i]

        beams = next_beams
        row_paths = next_row_paths
        # print(",".join(map(str, row_paths)))

    print(splits)
    print(sum(row_paths))


if __name__ == "__main__":
    main()
