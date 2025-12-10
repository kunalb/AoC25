import sys
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest

import numpy as np


def solve3(machine):
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
        total = sum(executor.map(solve, enumerate(machines)))
        print(total)

        total2 = sum(executor.map(solve3, enumerate(machines2)))
        print(total2)


if __name__ == "__main__":
    main()
