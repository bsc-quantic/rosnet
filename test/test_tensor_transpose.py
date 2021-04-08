from dislib_tensor import Tensor
import argparse


def main():
    parser = argparse.ArgumentParser(description='Tensor.transpose test')
    parser.add_argument('n', metavar='n', type=int,
                        nargs=1, help='Tensor rank')
    parser.add_argument('m', type=int, nargs=1, help='Block rank')
    args = parser.parse_args()
    n = args.n[0]
    m = args.m[0]

    shape = [2] * n
    block_shape = [2] * (n-m) + [1] * (m)
    t = Tensor.zeros(shape, block_shape)
    print(
        f'Created tensor with rank={t.rank}, shape={t.shape}, block_shape={t.block_shape}, grid={t.grid}')

    perm = list(range(n)[::-1])
    t.transpose(perm)
    print(
        f'Tranposed! Now rank={t.rank}, shape={t.shape}, block_shape={t.block_shape}, grid={t.grid}')

    t.sync()
    print(
        f'Synced! Now rank={t.rank}, shape={t.shape}, block_shape={t.block_shape}, grid={t.grid}')


if __name__ == '__main__':
    main()
