import rosnet


def main():
    a = rosnet.rand([2]*8, [2]*4 + [1]*4)
    axes_v = [2, 3, 6]
    u, v = rosnet.schmidt(a, axes_v)


if __name__ == '__main__':
    main()
