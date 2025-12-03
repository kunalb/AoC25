import sys

def main2():
    sum = 0
    for joltage in sys.stdin:
        nos = list(map(int, list(joltage.strip())))

        start = 0
        stop = -11
        no = 0
        #print(joltage)
        for i in range(12):
            if stop < 0:
                rnos = nos[start:stop]
            else:
                rnos = nos[start:]
            xi = max(rnos)
            no = no * 10 + xi
            start = start + rnos.index(xi) + 1
            stop += 1
        # print(xi, start, stop, no)
        sum += no

    print(sum)


def main():
    sum = 0
    for joltage in sys.stdin:
        nos = list(map(int, list(joltage.strip())))
        a = max(nos[:-1])
        ai = nos.index(a)
        b = max(nos[ai + 1:])
        sum += 10 * a + b
        #  print(joltage, b, a)
    print(sum)

if __name__ == "__main__":
    # main()
    main2()
