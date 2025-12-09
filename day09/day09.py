import itertools
import sys


def inside(pt, ab):
    if ab[0][0] < ab[1][0]:
        xs = (ab[0][0], ab[1][0])
    else:
        xs = (ab[1][0], ab[0][0])

    if ab[0][1] < ab[1][1]:
        ys = (ab[0][1], ab[1][1])
    else:
        ys = (ab[1][1], ab[0][1])

    # print(pt, ab, xs, ys)
    return (
        (xs[0] < pt[0] < xs[1]) and
        (ys[0] < pt[1] < ys[1])
    )


def intersects(h, v):
    if h[0][0] < h[1][0]:
        h0, h1 = h
    else:
        h1, h0 = h
    if v[0][1] < v[1][1]:
        v0, v1 = v
    else:
        v1, v0 = v

    assert v0[0] == v1[0]
    assert h0[1] == h1[1]
    return (
        (h0[0] < v0[0] < h1[0]) and
        (v0[1] < h0[1] < v1[1])
    )

def ray(pt, vert, horz):
    count = 0
    # print(pt)

    for edge in horz:
        if (
            edge[0][1] == pt[1] and
            (edge[0][0] <= pt[0] <= edge[1][0])
        ):
            return True

    for edge in vert:
        # print(edge)
        if edge[0][0] < pt[0]:
            continue

        if edge[0][0] == pt[0] and (edge[0][1] <= pt[1] <= edge[1][1]):
            return True

        if edge[1][1] == pt[1] and edge[0][1] < pt[1]:
            count += 1
            continue

        if edge[0][1] <= pt[1] <= edge[1][1]:
            count += 1

    return count & 1 != 0


def main():
    tiles = [
        tuple(map(int, row.strip().split(",")))
        for row in sys.stdin
    ]
    areas = list(
       (a, b, (1 + abs(a[0] - b[0])) * (1 + abs(a[1] - b[1])))
       for (a,b) in itertools.combinations(tiles, 2)
    )
    areas.sort(key=lambda x:x[2], reverse=True)
    print(areas[0][2])

    horz = []
    vert = []
    for (a, b) in zip(tiles, tiles[1:] + tiles[0:1]):
        if a[0] == b[0]:
            vert.append((a, b) if a[1] <= b[1] else (b, a))
        else:
            assert a[1] == b[1]
            horz.append((a, b) if a[0] <= b[0] else (b, a))

    for i, (a, b, area) in enumerate(areas):
        # print(i, a, b, area)
        failed = False

        for corner in [
            (a[0], b[1]),
            (b[0], a[1]),
        ]:
            vray = ray(corner, vert, horz)
            # print(corner, vray)
            if not vray:
                failed = True
                break

        horiz_sides = [
            [(a[0], a[1]), (b[0], a[1])],
            [(a[0], b[1]), (b[0], b[1])],
        ]
        vert_sides = [
            [(a[0], a[1]), (a[0], b[1])],
            [(b[0], a[1]), (b[0], b[1])],
        ]

        for side in horiz_sides:
            if failed:
                break
            failed = any(intersects(side, v) for v in vert)

        for side in vert_sides:
            if failed:
                break
            failed = any(intersects(h, side) for h in horz)

        for tile in tiles:
            if failed:
                break
            failed = inside(tile, (a, b))

        if not failed:
            print(area)
            return

if __name__ == "__main__":
    main()
