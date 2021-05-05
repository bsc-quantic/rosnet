import logging

try:
    logging.info('"transpose" backend: HPTT')
    from hptt import transpose

except ImportError:
    logging.info('"transpose" backend: numpy')
    import numpy as np

    def transpose(a, axes=None):
        return np.asfortranarray(np.transpose(a, axes))
