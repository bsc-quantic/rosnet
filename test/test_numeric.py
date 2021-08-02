import pytest
import numpy as np
import rosnet.array as rna

def test_qft_zero_2():
    m = 2
    psi0 = np.array([1, 0], dtype=np.cdouble)
    h = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.cdouble)
    r = lambda i: np.array([[1, 0], [0, np.exp(1j * 2 * np.pi * i / np.power(2, m))]], dtype=np.cdouble)
    cr = lambda i: np.block([[np.eye(2, dtype=np.cdouble), np.zeros((2,2), dtype=np.cdouble)],[np.zeros((2,2), dtype=np.cdouble), r(i)]]).reshape([2]*4)

    res = np.einsum('a,b,ac,cbef,fg,e,g->', psi0, psi0, h, cr(1), h, psi0, psi0)
    assert res == pytest.approx(0.5)