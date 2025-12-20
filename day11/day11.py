import sys

from collections import deque, defaultdict

# too low: 565285001952


def inv_paths(invs, to):
    q = list()
    q.append(to)

    paths = defaultdict(lambda: 0)
    paths[to] = 1

    outcount = defaultdict(lambda: 0)
    outcount[to] = 0

    popped = set()

    while q:
        choice = None
        for i, x in enumerate(q):
            if all(
                x not in invs[y]
                for y in q
                if x != y
            ):
                choice = (i, x)
                break

        top = choice[1]
        q.pop(choice[0])

        wt = paths[top]
        popped.add(top)

        if top in invs:
            to_add = []
            for c in invs[top]:
                if c not in paths:
                    q.append(c)
                paths[c] += wt

                assert c not in popped, c

    return paths


def part2(rows):
    invs = defaultdict(set)
    for row in rows:
        begin, ends = row.split(":")
        ends = ends.strip().split()
        for end in ends:
            invs[end].add(begin)

    sinks = {}
    for sink in ["out", "fft", "dac"]:
        sinks[sink] = inv_paths(invs, sink)

    # print(sinks)

    print(
        sinks["dac"]["svr"] *
        sinks["fft"]["dac"] *
        sinks["out"]["fft"]
        +
        sinks["fft"]["svr"] *
        sinks["dac"]["fft"] *
        sinks["out"]["dac"]
    )


def part1(rows):
    conns = {}

    for row in rows:
        begin, ends = row.split(":")
        conns[begin] = ends.strip().split()

    paths = 0
    visited = set()
    q = deque()
    q.append("you")

    while q:
        top = q.popleft()
        if top == "out":
            paths += 1

        visited.add(top)
        if top in conns:
            q.extend(
                conn for conn in conns[top]
                # if conn not in visited
            )

    print(paths)


if __name__ == "__main__":
    rows = list(sys.stdin.readlines())
    part1(rows)
    part2(rows)
