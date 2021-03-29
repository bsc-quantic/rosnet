from typing import List


def isunique(l: List) -> bool:
    """ Checks whether all the elements in `l` are unique in `l`.
    """
    return len(set(l)) == len(l)

# def coord(s):
#     c = [s // stride[-1]]
#     mod = s % stride[-1]
#     for stride_j in reversed(stride[:-1]):
#         c.append(mod // stride_j)
#         mod = mod % stride_j
#     c.reverse()
#     return c
