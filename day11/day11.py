import sys

from collections import deque, defaultdict


def toposort(invs):
    visited = set()
    toposorted = []

    for k in invs:
        if k in visited:
            continue

        st = [k]
        visited.add(k)

        while st:
            cur = st[-1]
            unvisited = True

            for a in invs.get(cur, []):
                if a in visited:
                    continue

                unvisited = False
                visited.add(a)
                st.append(a)
                break

            if unvisited:
                toposorted.append(st.pop())

    toposorted.reverse()
    return toposorted


def inv_paths(invs, to):
    q = list()
    q.append(to)

    paths = defaultdict(lambda: 0)
    paths[to] = 1

    popped = set()
    nextq = []

    elems = toposort(invs)
    for top in elems:
        wt = paths[top]
        popped.add(top)

        for c in invs.get(top, []):
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
