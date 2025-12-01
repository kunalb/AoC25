import sys


def main():
    position = 50
    count = 0
    count2 = 0

    for row in sys.stdin:
        prev_position = position
        if row[0] == 'L':
            position -= int(row[1:])
        else:
            position += int(row[1:])

        if position < 0:
            count2 += (-position) // 100
            if prev_position != 0:
                count2 += 1
        elif position == 0:
            count2 += 1
        else:
            count2 += position // 100
        # print(position % 100, position, count2)
        position = position % 100
        if position == 0:
            count+=1
    print(count)
    print(count2)


if __name__ == "__main__":
    main()
