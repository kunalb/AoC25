import math
import sys
from collections import defaultdict


def main():
    boxes = []
    for row in sys.stdin:
        if not row:
            continue
        boxes.append(list(map(int, row.split(","))))

    count = len(boxes)

    distances = []
    for i in range(0, len(boxes)):
        for j in range(i + 1, len(boxes)):
            distances.append((
                sum((a - b)*(a-b) for (a, b) in zip(boxes[i], boxes[j])),
                i,
                j,
            ))

    distances.sort(key=lambda x: x[0])

    circuit_id = 1
    circuits = [0] * count
    steps = 0
    for pair in distances:
        circuit1 = circuits[pair[1]]
        circuit2 = circuits[pair[2]]
        if circuit1 == circuit2 and circuit1 != 0:
            ...
        elif circuit1 == 0 and circuit2 == 0:
            circuits[pair[1]] = circuit_id
            circuits[pair[2]] = circuit_id
            circuit_id += 1
        elif circuit1 == 0 and circuit2 != 0:
            circuits[pair[1]] = circuit2
        elif circuit1 != 0 and circuit2 == 0:
            circuits[pair[2]] = circuit1
        elif circuit1 != 0 and circuit2 != 0:
            total = 0
            for i in range(count):
                if circuits[i] == circuit1 or circuits[i] == circuit2:
                    circuits[i] = circuit_id
                    total += 1
            if total == count:
                print(boxes[pair[1]][0] * boxes[pair[2]][0])
                return

            circuit_id += 1

        steps += 1
        if steps == 1000:
            sizes = defaultdict(lambda: 0)
            for val in circuits:
                if val != 0:
                    sizes[val] += 1
            print(math.prod(sorted(sizes.values(), reverse=True)[:3]))


if __name__ == "__main__":
    main()
