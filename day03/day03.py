import sys


def solve(nos, sz):
    start = 0
    stop = 1 - sz
    no = 0
    l = len(nos)
    for i in range(sz):
        rnos = nos[start:(l + stop)]
        xi = max(rnos)
        no = no * 10 + xi
        start += rnos.index(xi) + 1
        stop += 1
    return no


def main():
    sum1, sum2 = 0, 0
    for joltage in sys.stdin:
        nos = list(map(int, list(joltage.strip())))
        sum1 += solve(nos, 2)
        sum2 += solve(nos, 12)

    print(sum1)
    print(sum2)

if __name__ == "__main__":
    main()
