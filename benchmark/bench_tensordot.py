from rosnet import tensordot, ones
import argparse
import random as rnd


def main():
    parser = argparse.ArgumentParser(description='Tensor.tensordot test')
    parser.add_argument('n', type=int, nargs=1, help='Tensor rank')
    parser.add_argument('m', type=int, nargs=1, help='Block rank')
    parser.add_argument('k', type=int, nargs=1, help='#contraction indexes')
    parser.add_argument('--samples', dest='num_samples', type=int, nargs=1,
                        default=10, help='#samples')
    args = parser.parse_args()
    n = args.n[0]
    m = args.m[0]
    k = args.k[0]
    num_samples = args.num_samples

    shape = [2] * n
    block_shape = [2] * (m) + [1] * (n-m)
    samples = [tuple(rnd.randint(0, i-1) for i in shape)
               for _ in range(num_samples)]
    a = ones(shape, block_shape)
    print("Tensor A:")
    for sample in samples:
        print(f"\t{sample} => {a[sample]}")

    print()
    print("Tensor B:")
    b = ones(shape, block_shape)
    for sample in samples:
        print(f"\t{sample} => {b[sample]}")

    axes = [list(range(k)), list(range(k))]
    c = tensordot(a, b, axes)

    c.sync()
    print(
        f'Synced! Now rank={c.rank}, shape={c.shape}, block_shape={c.block_shape}, grid={c.grid}')
    samples = [tuple(rnd.randint(0, i-1) for i in c.shape)
               for _ in range(num_samples)]
    for sample in samples:
        print(f"\t{sample} => {c[sample]}")


if __name__ == '__main__':
    main()
