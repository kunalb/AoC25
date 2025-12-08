import math
import sys
from collections import defaultdict


def main():
    boxes = []
    for row in sys.stdin:
        if not row:
            continue
        boxes.append(list(map(int, row.split(","))))

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
    circuits = [0] * 1000
    steps = 0
    for pair in distances:
        # print(pair, boxes[pair[1]], boxes[pair[2]])
        circuit1 = circuits[pair[1]]
        circuit2 = circuits[pair[2]]
        if circuit1 == circuit2 and circuit1 != 0:
            continue
        if circuit1 == 0 and circuit2 == 0:
            circuits[pair[1]] = circuit_id
            circuits[pair[2]] = circuit_id
            circuit_id += 1
        if circuit1 == 0 and circuit2 != 0:
            circuits[pair[1]] = circuit2
        if circuit1 != 0 and circuit2 == 0:
            circuits[pair[2]] = circuit1
        if circuit1 != 0 and circuit2 != 0:
            total = 0
            for i in range(1000):
                if circuits[i] == circuit1 or circuits[i] == circuit2:
                    circuits[i] = circuit_id
                    total += 1
            if total == 1000:
                print(boxes[pair[1]][0] * boxes[pair[2]][0])
                return

            circuit_id += 1

    # print(circuits[:20])
    sizes = defaultdict(lambda: 0)
    for val in circuits:
        if val != 0:
            sizes[val] += 1
    print(sizes)
    print(math.prod(sorted(sizes.values(), reverse=True)[:3]))


if __name__ == "__main__":
    main()
