import sys


def main():
    inputs = sys.stdin.read()
    ranges = [
        tuple(map(int, r.split("-")))
        for r in inputs.split(",")
    ]

    sum = 0
    sum2 = 0
    for r in ranges:
        for num in range(r[0], r[1] + 1):
            strnum = str(num)
            l = len(strnum)

            for sz in range(l - 1, 0, -1):
                if l % sz != 0:
                    continue
                pieces = {
                    strnum[i:i+sz]
                    for i in range(0, l, sz)
                }
                if len(pieces) == 1:
                    # print(num)
                    sum2 += num
                    break

            if l & 1 != 0:
                continue
            if strnum[0:l//2] == strnum[l//2:]:
                sum += num

    print(sum)
    print(sum2)

if __name__ == "__main__":
    main()
