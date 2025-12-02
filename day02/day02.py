import sys
from concurrent.futures import ThreadPoolExecutor

def solve_range(start, stop):
    sum = 0
    sum2 = 0

    for num in range(start, stop + 1):
        strnum = str(num)
        l = len(strnum)

        checked = set()
        for sz in range(l // 2, 0, -1):
            if l % sz != 0:
                continue

            for c in checked:
                if c % sz == 0:
                    continue

            checked.add(sz)

            matched = True
            for i in range(sz, l, sz):
                matched = strnum[i:i+sz] == strnum[0:sz]
                if not matched:
                    break
            if matched:
                sum2 += num
                break

        if l & 1 != 0:
            continue
        if strnum[0:l//2] == strnum[l//2:]:
            sum += num

    return sum, sum2


def main():
    inputs = sys.stdin.read()
    ranges = [
        tuple(map(int, r.split("-")))
        for r in inputs.split(",")
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        sums = list(executor.map(lambda x: solve_range(*x), ranges))

    sum1 = sum(x[0] for x in sums)
    sum2 = sum(x[1] for x in sums)

    print(sum1)
    print(sum2)

if __name__ == "__main__":
    main()
