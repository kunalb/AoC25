import math
import sys


def parse_shape(shape):
    mat = []
    for i, row in enumerate(shape):
        mat.append([
            1 if c == '#' else 0
            for c in row
        ])
    return mat


def main():
    shapes = []
    spaces = []

    in_shape = False
    for row in sys.stdin:
        if row == "\n":
            in_shape = False
            continue

        if in_shape:
            shapes[-1].append(row.strip())
            continue

        index, *rest = row.strip().split(":")

        if rest[0]:
            spaces.append(
                (tuple(map(int, index.split("x"))),
                tuple(map(int, rest[0].split())))
            )
        else:
            shapes.append([])
            in_shape = True

    shapes = list(map(parse_shape, shapes))
    # print(shapes, spaces)

    weights = [
        sum(sum(row) for row in shape)
        for shape in shapes
    ]

    solvable = 0
    unsolvable = 0
    for (box, reqs) in spaces:
        boxes = (box[0] // 3) * (box[1] // 3)
        total = sum(reqs)

        space = box[0] * box[1]
        lower_bound = sum(
            a * b
            for (a, b) in
            zip(weights, reqs)
        )

        if lower_bound > space:
            unsolvable += 1

        if boxes >= total:
            solvable += 1

    assert solvable + unsolvable == len(spaces), "Some boxes unknown"
    print(solvable)



if __name__ == "__main__":
    main()
