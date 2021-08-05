import opt_einsum as oe
import rosnet
import cotengra as ctg

n = 10
phys_dim = 3
bond_dim = 10

# start with first site
# O--
# |
# O--
einsum_str = "ab,ac,"

for i in range(1, n - 1):
    # set the upper left/right, middle and lower left/right indices
    # --O--
    #   |
    # --O--
    j = 3 * i
    ul, ur, m, ll, lr = (oe.get_symbol(i) for i in (j - 1, j + 2, j, j - 2, j + 1))
    einsum_str += "{}{}{},{}{}{},".format(m, ul, ur, m, ll, lr)

# finish with last site
# --O
#   |
# --O
i = n - 1
j = 3 * i
(
    ul,
    m,
    ll,
) = (oe.get_symbol(i) for i in (j - 1, j, j - 2))
einsum_str += "{}{},{}{}".format(m, ul, m, ll)


def gen_shapes():
    yield (phys_dim, bond_dim)
    yield (phys_dim, bond_dim)
    for i in range(1, n - 1):
        yield (phys_dim, bond_dim, bond_dim)
        yield (phys_dim, bond_dim, bond_dim)
    yield (phys_dim, bond_dim)
    yield (phys_dim, bond_dim)


shapes = tuple(gen_shapes())
print(shapes)

# arrays = [np.random.randn(*shp) / 4 for shp in shapes]
arrays = [rosnet.rand(shp) for shp in shapes]

opt = ctg.HyperOptimizer(
    methods=["kahypar", "greedy"],
    max_repeats=16,
    parallel=True,
    progbar=True,
)

path, info = oe.contract_path(einsum_str, *arrays, memory_limit=-1)
print(str(info).encode("utf-8"))

path, info = oe.contract_path(einsum_str, *arrays, memory_limit=-1, optimize=opt)
print(str(info).encode("utf-8"))
