from dislib_tensor import Tensor
import argparse
import numpy as np
import random as rnd


def main():
    parser = argparse.ArgumentParser(description='Tensor.__init__ test')
    parser.add_argument('n', metavar='n', type=int,
                        nargs=1, help='Tensor rank')
    parser.add_argument('m', type=int, nargs=1, help='Block rank')
    parser.add_argument('--samples', dest='num_samples', type=int, nargs=1,
                        default=10, help='#samples')
    args = parser.parse_args()
    n = args.n[0]
    m = args.m[0]
    num_samples = args.num_samples

    shape = [2] * n
    block_shape = [2] * (m) + [1] * (n-m)
    arr = np.random.rand(*shape)
    print("Array generated! Sampling random locations:")
    samples = [tuple(rnd.randint(0, i-1) for i in shape)
               for _ in range(num_samples)]
    for sample in samples:
        print(f"\t{sample} => {arr[sample]}")

    t = Tensor.array(arr, block_shape)
    print(
        f'Created tensor with rank={t.rank}, shape={t.shape}, block_shape={t.block_shape}, grid={t.grid}')
    print("Tensor generated! Sampling random locations:")
    for sample in samples:
        print(f"\t{sample} => {t[sample]}")

    perm = list(range(n)[::-1])
    t.transpose(perm)
    print(
        f'Tranposed! Now rank={t.rank}, shape={t.shape}, block_shape={t.block_shape}, grid={t.grid}')

    samples = [sample[::-1] for sample in samples]
    print("Tensor transposed! Sampling random locations:")
    for sample in samples:
        print(f"\t{sample} => {t[sample]}")

    t.sync()
    print(
        f'Synced! Now rank={t.rank}, shape={t.shape}, block_shape={t.block_shape}, grid={t.grid}')


if __name__ == '__main__':
    main()
