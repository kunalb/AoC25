import sys


def main():
    position = 50
    count = 0
    count2 = 0

    for row in sys.stdin:
        turn, val = row[0], int(row[1:])
        prev_position = position
        match turn:
            case 'L':
                position -= val
            case 'R':
                position += val

        if position < 0:
            count2 += (-position) // 100
            if prev_position != 0:
                count2 += 1
        elif position == 0:
            count2 += 1
        else:
            count2 += position // 100

        position = position % 100
        if position == 0:
            count+=1

    print(count)
    print(count2)


if __name__ == "__main__":
    main()
