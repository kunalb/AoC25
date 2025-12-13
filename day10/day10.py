import sys
import math
import itertools
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from fractions import Fraction


###

def pm(mat, title=""):
    if title:
        print(title)
    for row in mat:
        print(" ".join(map(lambda x: f"{x}", row)))
    print()


def reduce_mat(matrix):
    # Only need to handle first n columns
    sq = len(matrix[0]) - 1
    for col in range(sq):
        for row in range(col, len(matrix)):
            if matrix[row][col] != 0:
                mul = Fraction(1, matrix[row][col])
                if row != col:
                    tmp = matrix[col]
                    matrix[col] = matrix[row]
                    matrix[row] = tmp

                    pm(matrix, "reorder")

                for c in range(len(matrix[col])):
                    matrix[col][c] *= mul
                pm(matrix, "mul")

                break
        pm(matrix, f"{col} set1")

        for row in range(col + 1, len(matrix)):
            if matrix[row][col] != 0:
                orig = matrix[row][col] / matrix[col][col]
                for c in range(len(matrix[row])):
                    if c == col:
                        matrix[row][col] = 0
                    else:
                        matrix[row][c] = matrix[row][c] - orig * matrix[col][c]

        pm(matrix, "sub")

    zs = 0
    for i, row in enumerate(matrix):
        nonzero = i < len(row) and row[i] != 0
        if not nonzero:
            continue

        if zs < i:
            tmp = row
            matrix[i] = matrix[zs]
            matrix[zs] = row

        zs += 1

    pm(matrix, "final reorder")


def matrix_solve(machine):
    idx, (joltage, combos) = machine
    print(machine)

    matrix = [[] for _ in joltage]

    for i in range(len(joltage)):
        for j, combo in enumerate(combos):
            matrix[i].append(combo[i] if i < len(combo) else 0)
        matrix[i].append(joltage[i])

    pm(matrix, "start")
    reduce_mat(matrix)

    constraints = sum(any(x != 0 for x in row) for row in matrix)
    variables = len(combos)
    print(f"{variables=} {constraints=}")
    assert variables >= constraints

    if variables == constraints:
        vars = [0] * variables
        for k in range(variables - 1, -1, -1):
            vars[k] = matrix[k][-1]
            for l in range(k + 1, variables):
                vars[k] -= matrix[k][l] * vars[l]

        print("exact")
        print(",".join(map(str, vars)))
        print(sum(vars))

        return sum(vars)
    else:
        dims = (variables - constraints + 1)
        vars = [[0] * dims  for _ in range(variables)]

        param_idx = 0
        row = 0
        for i in range(variables):
            if row < constraints and matrix[row][i] != 0:
                row += 1
            else:
                vars[i][param_idx] = 1
                param_idx += 1
        assert param_idx == dims - 1

        for k in range(constraints - 1, -1, -1):
            col = 0
            while matrix[k][col] == 0:
                col += 1

            vars[col][-1] = matrix[k][-1]
            for l in range(col + 1, variables):
                for m in range(dims):
                     vars[col][m] -= matrix[k][l] * vars[l][m]

        pm(vars, "vars")
        total_val = [0] * dims

        for d in range(dims):
            total_val[d] = sum(v[d] for v in vars)

        print("total", total_val)

        return 0

###

executor2 = ThreadPoolExecutor()


def resolve(idx, combos, joltage, fixed = None, indent=""):
    if not fixed:
        fixed = [None] * len(combos)
    else:
        fixed = list(fixed)

    lowers = list(fixed)
    uppers = list(fixed)

    target_joltage = [
        j - sum(
            f for c, f in zip(combos, fixed)
            if f is not None and i < len(c) and c[i] == 1
        )
        for (i, j) in enumerate(joltage)
    ]
    if any(x < 0 for x in target_joltage):
        return -1

    changed = True
    while changed:
        changed = False
        for i, j in enumerate(target_joltage):
            for k, c in enumerate(combos):
                if i >= len(c) or c[i] != 1:
                    continue

                min_other = sum(
                    l for (m,(l, c)) in enumerate(zip(lowers, combos))
                    if m != k and lowers[m] is not None and
                    i < len(c) and c[i] == 1
                )

                r = joltage[i] - min_other
                if r < 0:
                    return None

                if uppers[k] is None or uppers[k] > r:
                    changed = True
                    uppers[k] = r

        for i, j in enumerate(target_joltage):
            for k, c in enumerate(combos):
                if i >= len(c) or c[i] != 1:
                    continue

                max_other = sum(
                    u for (l,(u, c)) in enumerate(zip(uppers, combos))
                    if l != k and i < len(c) and c[i] == 1
                )
                if (r := joltage[i] - max_other) > 0:
                    if r > uppers[k]:
                        return None

                    if lowers[k] is None or lowers[k] < r:
                        changed = True
                        lowers[k] = r
                elif lowers[k] is None:
                    changed = True
                    lowers[k] = 0

    if False and all(x is None for x in fixed):
        print(",".join(map(
            lambda x: f"{x[0]}-{x[1]}",
            zip(lowers, uppers),
        )))

    next_target = None
    for i, (l, u) in enumerate(zip(lowers, uppers)):
        if l == u:
            fixed[i] = l
        else:
            attractiveness = (len(combos[i]), l - u)
            if next_target is None or (
                attractiveness > next_target[1]
            ):
                next_target = (i, attractiveness)

    if next_target is None:
        for i, j in enumerate(joltage):
            if j != sum(
                f for f, c in zip(fixed, combos)
                if i < len(c) and c[i] == 1
            ):
                return None
        ## print(indent, ">", ",".join(map(str, fixed)))
        return sum(fixed)
    i = next_target[0]

    min_sum = None

    # pt = sum(1 for x in fixed if x is not None)
    # print(f"{idx}/{pt}/{len(fixed)}: {lowers[i]} -> {uppers[i]}")

    for v in range(lowers[i], uppers[i] + 1):
        fixed[i] = v
        # print("Fix", i, v, lowers[i], uppers[i], fixed)
        result = resolve(idx, combos, joltage, fixed, indent + "  ")
        # print(indent, f"resolve({','.join(map(lambda x: '_' if x is None else str(x), fixed))}) = {result}")
        if result is None:
            continue
        if result < 0:
            break
        if min_sum is None or result < min_sum:
            min_sum = result

    return min_sum


def solver(machine):
    idx, (joltage, combos) = machine
    # print(joltage)
    # for combo in combos:
    #     print(combo)

    result = resolve(idx, combos=combos, joltage=joltage)
    print(idx, end=" ", flush=True, file=sys.stderr)
    return result


def solve3(machine):
    import numpy as np
    idx, (joltage, combos) = machine

    coeffs = []
    for i in range(len(joltage)):
        coeffs.append(
            [combo[i] if len(combo) > i else 0
            for combo in combos]
        )

    result = np.linalg.solve(
        np.array(coeffs),
        np.array(joltage)
    )
    return sum(result)


def make_tuple(combo):
    result = [0] * (max(combo) + 1)
    for x in combo:
        result[x] += 1
    return tuple(result)


def dist2(joltage, curr):
    return any(map(lambda x: x[0] < x[1], zip(joltage, curr)))

def dist3(joltage, curr):
    return sum(map(lambda x: x[0] - x[1], zip(joltage, curr)))

def make_number(elems):
    num = 0
    for elem in elems:
        num |= (1 << elem)
    return num


def dist(num, state):
    return 0

    x = num ^ state
    count = 0
    while x != 0:
        x &= (x-1)
        count += 1
    return count


def solve(machine):
    idx, (state, combos) = machine
    steps = 0
    root = 0
    visited = set()
    fringe = {0: 0}
    costs = {}

    while True:
        start_node = min(fringe.items(), key=lambda x: x[1] + dist(state, x[0]))
        fringe.pop(start_node[0])
        if start_node[0] == state:
            # print(idx, state, combos, start_node)
            return start_node[1]

        for combo in combos:
            next_node = start_node[0] ^ combo
            if next_node not in fringe:
                fringe[next_node] = start_node[1] + 1

    1/0


def solve2(machine):
    idx, (joltage, combos) = machine
    # print(joltage, combos)

    steps = 0
    root = 0
    visited = set()
    fringe = {tuple(([0] * len(joltage))): 0}
    costs = {}

    while True:
    # for i in range(10):
        start_node = min(fringe.items(), key=lambda x: x[1] + dist3(joltage, x[0]))
        fringe.pop(start_node[0])
        visited.add(start_node[0])
        if start_node[0] == joltage:
            # print(idx, state, combos, start_node)
            return start_node[1]

        for combo in combos:
            next_node = tuple(map(sum, zip_longest(start_node[0], combo, fillvalue=0)))
            if dist2(joltage, next_node):
                continue

            if next_node not in fringe and next_node not in visited:
                fringe[next_node] = start_node[1] + 1

    1/0

def main():
    machines = []
    machines2 = []

    for row in sys.stdin:
        pieces = list(row.split())
        state = make_number(
            i for (i, c) in enumerate(pieces[0].strip("[]"))
            if c == '#'
        )
        joltage = tuple(map(int, pieces[-1].strip("{}").split(",")))
        combos2 = [
            list(map(int, combo.strip("()").split(",")))
            for combo in pieces[1:-1]
        ]
        combos = [make_number(combo) for combo in combos2]
        combos3 = [make_tuple(combo) for combo in combos2]
        machine = (state, combos)
        machine2 = (joltage, combos3)
        machines.append(machine)
        machines2.append(machine2)

    with ThreadPoolExecutor() as executor:
        # total = sum(executor.map(solve, enumerate(machines)))
        # print(total)

        # total2 = sum(executor.map(solver, enumerate(machines2)))
        total2 = sum(map(matrix_solve, enumerate(machines2)))
        print(total2)


if __name__ == "__main__":
    main()
