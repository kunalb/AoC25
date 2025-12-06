import sys


def main2():
    inp = sys.stdin.read().strip('\n')
    lines = inp.splitlines()
    cols = max(len(line) for line in lines)

    total = 0

    cur_op = None
    nums = []

    for i in range(cols):
        if i < len(lines[-1]) and (lines[-1][i] == '*' or lines[-1][i] == '+'):
            match cur_op:
                case '+':
                    res = 0
                    for num in nums:
                        res += num
                    total += res
                case '*':
                    res = 1
                    for num in nums:
                        res *= num
                    total += res
            cur_op = lines[-1][i]
            nums = []

        cur = 0
        found = False
        for line in lines[:-1]:
            if i >= len(line):
                continue

            ch = line[i]
            if ch >= '0' and ch <= '9':
                cur = 10 * cur + int(ch)
                found = True

        if not found:
            continue

        nums.append(cur)

    match cur_op:
        case '+':
            res = 0
            for num in nums:
                res += num
            total += res
        case '*':
            res = 1
            for num in nums:
                res *= num
            total += res

    print(total)


def main():
    rows = sys.stdin.read().strip()
    data = [row.split() for row in rows.splitlines()]

    r = len(data)
    c = len(data[0])

    total = 0
    for y in range(c):
        op = data[r - 1][y]
        match op:
            case '+':
                res = sum(int(data[x][y]) for x in range(r -1))
                total += res
            case '*':
                res = 1
                for x in range(r - 1):
                    res *= int(data[x][y])
                total += res
            case _:
                print(op)
                1/0

    print(total)

if __name__ == "__main__":
    main2()
