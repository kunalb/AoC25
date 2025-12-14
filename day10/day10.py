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


def reduce_matrix(mat):
    log = False

    n = min(len(mat), len(mat[0]))

    cur_row = 0
    for col in range(n):
        # Swap in first non-zero row
        found1 = False
        for r in range(cur_row, len(mat)):
            if mat[r][col] == 0:
                continue

            found1 = True
            if r == cur_row:
                break

            tmp = mat[r]
            mat[r] = mat[cur_row]
            mat[cur_row] = tmp

            log and pm(mat, f"{cur_row=} {col=} swapped")
            break

        if not found1:
            continue

        # Make the value 1
        if mat[cur_row][col] != 1:
            ratio = Fraction(1, mat[cur_row][col])
            for c in range(col, len(mat[cur_row])):
                mat[cur_row][c] *= ratio
            log and pm(mat, f"{cur_row=} {col=} normalized")

        # Reduce all other rows
        zeroed = False
        for row in mat[cur_row +1:]:
            if row[col] == 0:
                continue
            zeroed = True
            for c in range(len(row) - 1, col - 1, -1):
                row[c] -= mat[cur_row][c] * Fraction(row[col], mat[cur_row][col])
        if zeroed:
            log and pm(mat, f"{cur_row=} {col=} zero'd out rest")

        cur_row += 1


def solve_val_constraints(vals, total_val, params=None):
    dims = len(vals[0])
    log = False
    limits = [[0, None] for _ in range(dims - 1)]
    checks = []

    log and print("params", params, all(p is not None for p in params) if params else None)
    if params is None:
        params = [None] * (dims -1)

    if all(p is not None for p in params):
        for val in vals:
            v = val[-1] + sum(a * b for (a, b) in zip(params, val))
            log and print("val check", v, val)
            assert v >= 0, f"val check failed {val}"

        return total_val[-1] + sum(a * b for (a, b) in zip (params, total_val))

    # Each value must be greater than 0
    for val in vals:
        unknowns = sum(1 for i, x in enumerate(val[:-1]) if x != 0 and params[i] is None)
        if unknowns > 1: # 2 unknowns, manually confirm
            checks.append(val)
            continue

        if unknowns == 0:
            continue

        for i, p in enumerate(val[:-1]):
            if p == 0:
                continue

            limit = -Fraction(val[-1] + sum(q * v for (q, v) in zip(params, val) if q is not None), p)
            if p > 0:
                if limits[i][0] < limit:
                    limits[i][0] = limit
            elif p < 0:
                if limits[i][1] is None or limits[i][1] > limit:
                    limits[i][1] = limit

    for limit in limits:
        if limit[1] is not None and limit[0] > limit[1]:
            assert False, "Impossible limits found"

    log and print("limits", limits)
    explore = min(
        filter(lambda x: params[x[0]] is None, enumerate(limits)),
        key=lambda x: (x[1][1] or math.inf) - x[1][0]
    )

    if total_val[explore[0]] >= 0:
        guess = explore[1][0]
        while explore[1][1] is None or guess <= explore[1][1]:
            params[explore[0]] = guess
            try:
                return solve_val_constraints(vals, total_val, params)
            except Exception as e:
                log and print(e)
            guess += 1
    else:
        guess = explore[1][1]
        while guess >= explore[1][0]:
            params[explore[0]] = guess
            try:
                return solve_val_constraints(vals, total_val, params)
            except Exception as e:
                log and print(e)
            guess -= 1

    assert False, "No guess worked"


def val_constraints(vals):
    log = False

    dims = len(vals[0])
    log and pm(vals, "vals")
    total_val = [0] * dims
    for d in range(dims):
        total_val[d] = sum(v[d] for v in vals)
    log and print("total", total_val)

    return solve_val_constraints(vals, total_val)


def matrix_solve(machine):
    idx, (joltage, combos) = machine
    print(machine)

    matrix = [[] for _ in joltage]

    for i in range(len(joltage)):
        for j, combo in enumerate(combos):
            matrix[i].append(combo[i] if i < len(combo) else 0)
        matrix[i].append(joltage[i])

    pm(matrix, "start")
    reduce_matrix(matrix)
    pm(matrix, "reduced")

    constraints = sum(any(x != 0 for x in row) for row in matrix)
    variables = len(combos)
    print(f"{variables=} {constraints=}")
    assert variables >= constraints

    if variables == constraints:
        vals = [0] * variables
        for k in range(variables - 1, -1, -1):
            vals[k] = matrix[k][-1]
            for l in range(k + 1, variables):
                vals[k] -= matrix[k][l] * vals[l]

        print("exact")
        print(",".join(map(str, vals)))
        print(sum(vals))

        return sum(vals)

    dims = (variables - constraints + 1)
    vals = [[0] * dims  for _ in range(variables)]

    param_idx = 0
    row = 0
    for i in range(variables):
        if row < constraints and matrix[row][i] != 0:
            row += 1
        else:
            vals[i][param_idx] = 1
            param_idx += 1
    assert param_idx == dims - 1

    for k in range(constraints - 1, -1, -1):
        col = 0
        while matrix[k][col] == 0:
            col += 1

        vals[col][-1] = matrix[k][-1]
        for l in range(col + 1, variables):
            for m in range(dims):
                 vals[col][m] -= matrix[k][l] * vals[l][m]

    result = val_constraints(vals)
    print("result: ", result)
    print()

    return result

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
