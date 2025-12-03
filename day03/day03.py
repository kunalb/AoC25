import sys


def main():
    sum1, sum2 = 0, 0
    for joltage in sys.stdin:
        nos = list(map(int, list(joltage.strip())))

        a = max(nos[:-1])
        ai = nos.index(a)
        b = max(nos[ai + 1:])
        sum1 += 10 * a + b

        start = 0
        stop = -11
        no = 0
        l = len(nos)
        for i in range(12):
            rnos = nos[start:(l + stop)]
            xi = max(rnos)
            no = no * 10 + xi
            start = start + rnos.index(xi) + 1
            stop += 1
        sum2 += no

    print(sum1)
    print(sum2)

if __name__ == "__main__":
    main()
