import sys
import time
import math
import numbers
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
    print(flush=True)


def reduce_matrix(mat):
    log = False

    n = min(len(mat), len(mat[0]))

    cur_row = 0
    # for col in range(n):
    col = -1
    while cur_row < len(mat):
        col += 1
        if col >= len(mat[0]):
            break

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


def reduce_limits(vals, params):
    log = False

    limit_updated = True
    limits = [
        [0, math.inf] if p is None else [p, p]
        for p in params
    ]

    while limit_updated:
        limit_updated = False

        for val in vals:
            log and print("val:", val)

            for i, p in enumerate(params):
                log and print(f"param {i}: {p} | {val[i]}")
                if p is not None or val[i] == 0:
                    continue

                gtsum = [-val[-1], -val[-1]]
                for j, q in enumerate(params):
                    if j == i or val[j] == 0:
                        continue

                    if q is not None:
                        gtsum = [g - q * val[j] for g in gtsum]
                    else:
                        pl = sorted([val[j] * limits[j][k] for k in range(2)])
                        gtsum = [g - m for (g, m) in zip(gtsum, pl)]

                gtsum = sorted([g / val[i] for g in gtsum])
                if val[i] > 0:
                    if limits[i][0] < gtsum[0]:
                        limits[i][0] = math.ceil(gtsum[0])
                        limit_updated = True
                else:
                    if limits[i][1] > gtsum[1]:
                        limits[i][1] = math.floor(gtsum[1])
                        limit_updated = True

                log and print("limits", limits)

        all_set = True
        for limit in limits:
            if any(l is math.inf for l in limit):
                all_set = False
                break

    log and print("final limits", limits)
    return limits


def solve_val_constraints(vals, total_val, params=None):
    dims = len(vals[0])
    log = False
    checks = []

    log and print("params", params, all(p is not None for p in params) if params else None)
    if params is None:
        params = [None] * (dims -1)
    else:
        params = list(params)

    if all(p is not None for p in params):
        lcm = 1
        total = 0
        for val in vals:
            v = val[-1] + sum(a * b for (a, b) in zip(params, val))
            log and print("val check", v, val)
            assert v >= 0, f"val check failed {val}"
            assert Fraction(v).denominator == 1, f"All vals must be integers {v}"
            total += v

        return total

    # Repeatedly check limits and update bounds
    # Someting is looping in the limits
    limits = reduce_limits(vals, params)

    for k, limit in enumerate(limits):
        if limit[1] is not None and limit[0] > limit[1]:
            assert False, "Impossible limits found"

    log and print()
    log and print("limits", limits, params)
    explore = min(
        filter(lambda x: params[x[0]] is None, enumerate(limits)),
        key=lambda x: (x[1][1] or math.inf) - x[1][0]
    )
    log and print(f"Walking limit: {explore}")

    increment = Fraction(1, math.lcm(*[
        Fraction(g).denominator for g in explore[1] if g is not math.inf and g is not -math.inf
    ]))
    log and print("increment", increment)

    min_total = None

    if total_val[explore[0]] >= 0:
        guess = explore[1][0]
        unbounded = explore[1][1] is math.inf
        while (min_total is None and unbounded) or (not unbounded and guess <= explore[1][1]):
            params[explore[0]] = guess
            try:
                result = solve_val_constraints(vals, total_val, params)
                if min_total is None or min_total > result:
                    min_total = result
            except Exception as e:
                log and print(e)
            guess += increment
    else:
        assert explore[1][1] is not math.inf, "Starting from an infinite limit"
        unbounded = explore[1][0] is -math.inf
        guess = explore[1][1]
        while (min_total is None and unbounded) or (not unbounded and guess >= explore[1][0]):
            params[explore[0]] = guess
            try:
                result = solve_val_constraints(vals, total_val, params)
                if min_total is None or min_total > result:
                    min_total = result
            except Exception as e:
                log and print(e)
            guess -= increment

    if min_total is not None:
        return min_total
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
    log = False
    start_time_s = time.time()

    idx, (joltage, combos) = machine
    log and print(machine)
    print(f"START {idx}", flush=True)

    matrix = [[] for _ in joltage]

    for i in range(len(joltage)):
        for j, combo in enumerate(combos):
            matrix[i].append(combo[i] if i < len(combo) else 0)
        matrix[i].append(joltage[i])

    log and pm(matrix, "start")
    reduce_matrix(matrix)
    log and pm(matrix, "reduced")

    constraints = sum(any(x != 0 for x in row) for row in matrix)
    variables = len(combos)
    log and print(f"{variables=} {constraints=}", flush=True)
    assert variables >= constraints

    if variables == constraints:
        vals = [0] * variables
        for k in range(variables - 1, -1, -1):
            vals[k] = matrix[k][-1]
            for l in range(k + 1, variables):
                vals[k] -= matrix[k][l] * vals[l]

        log and print("exact")
        log and print(",".join(map(str, vals)))
        log and print(sum(vals))

        dur_s = time.time() - start_time_s
        print(f"DONE {idx} ({dur_s})", flush=True)
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

    # TODO: Introduce constraint that values must be an integer and express that correctly
    # 1) Find LCM of all the constant values and multiply throughout
    # lcm = math.lcm(*[
    #    Fraction(val[-1]).denominator
    #    for val in vals
    # ])
    # for p in range(len(vals[0])):
    #     for val in vals:
    #         val[p] *= lcm

    # 2) Make the parameters really work at search
    # for p in range(len(vals[0]) - 1):
    #     p_lcm = math.lcm(*[
    #         Fraction(val[p]).denominator
    #         for val in vals
    #     ])
    #     for val in vals:
    #         val[p] *= p_lcm
    # log and print("Multiplied vals", vals)

    result = val_constraints(vals)

    log and print("result: ", result)
    log and print()

    dur_s = time.time() - start_time_s
    print(f"DONE {idx} ({dur_s})", flush=True)
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
