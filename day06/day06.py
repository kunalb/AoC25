import string
import sys

from operator import mul
from functools import reduce


def resolve(op, nums):
    match op:
        case '+':
            return sum(nums)
        case '*':
            return reduce(mul, nums)
    return 0


def cephalopod_sum(inp):
    lines = inp.splitlines()
    cols = max(len(line) for line in lines)

    total = 0

    cur_op = None
    nums = []

    for i in range(cols):
        if i < len(lines[-1]) and (lines[-1][i] == '*' or lines[-1][i] == '+'):
            total += resolve(cur_op, nums)
            cur_op = lines[-1][i]
            nums = []

        cur = 0
        found = False
        for line in lines[:-1]:
            if i >= len(line):
                continue

            if (ch := line[i]).isdigit():
                cur = 10 * cur + int(ch)
                found = True

        if not found:
            continue

        nums.append(cur)

    total += resolve(cur_op, nums)
    return total


def human_sum(inp):
    data = [row.split() for row in inp.splitlines()]
    r = len(data)
    c = len(data[0])

    total = 0
    for y in range(c):
        op = data[r - 1][y]
        match op:
            case '+':
                total += sum(int(data[x][y]) for x in range(r -1))
            case '*':
                total += reduce(
                    mul,
                    (int(data[x][y]) for x in range(r -1))
                )
            case _:
                1/0

    return total


def main():
    inp = sys.stdin.read().strip('\n')
    print(human_sum(inp))
    print(cephalopod_sum(inp))


if __name__ == "__main__":
    main()
