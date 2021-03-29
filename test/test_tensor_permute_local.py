from dislib_tensor.tensor import Tensor


def main():
    shape = [2, 2, 2, 2, 2]
    block_rank = 2
    t = Tensor.zeros(shape, block_rank)
    t.permute(0, 1)

    print('Tensor')
    print(str(t.I))
    print(str(t.J))
    print(str(t.volume()))

    print('\nArray')
    print(str(len(t._matrix._blocks)))
    print(str(len(t._matrix._blocks[0])))


if __name__ == '__main__':
    main()
