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

    print(sum)
    print(sum2)

if __name__ == "__main__":
    main()
