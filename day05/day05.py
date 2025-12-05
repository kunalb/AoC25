import sys


def valid_ids(ids, ranges):
    i = 0
    j = 0

    total_ids = 0
    fresh_ids = 0

    while True:
        if j >= len(ids):
            break

        if i >= len(ranges):
            break

        if ids[j] > ranges[i][1]:
            i += 1
        elif ids[j] < ranges[i][0]:
            j += 1
        elif ids[j] >= ranges[i][0]:
            fresh_ids += 1
            j += 1

    return fresh_ids


def dedup(ranges):
    prev = ranges[0]
    fixed = [prev]

    for y in ranges[1:]:
        if y[1] <= prev[1]:
            continue
        if y[0] <= prev[1]:
            y[0] = prev[1] + 1

        fixed.append(y)
        prev = y

    return fixed


def main():
    fresh, ids = sys.stdin.read().split("\n\n")
    ranges = [
        list(map(int, row.split("-")))
        for row in fresh.strip().split()
    ]
    ids = list(map(int, ids.strip().split()))
    ids.sort()
    ranges.sort()
    ranges = dedup(ranges)

    print(valid_ids(ids, ranges))
    print(sum(y[1] - y[0] + 1 for y in ranges))


if __name__ == "__main__":
    main()
