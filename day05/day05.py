import sys


def search_range(ranges, x):
    for range in ranges:
        if x >= range[0] and x <= range[1]:
            return True
    return False


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

    ranges.sort(key=lambda x: x[0])
    result = sum(int(search_range(ranges, x)) for x in ids)
    print(result)

    deduped = dedup(ranges)
    print(sum(y[1] - y[0] + 1 for y in deduped))

main()
