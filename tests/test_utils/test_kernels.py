import numpy as np

from src.tedi.utils.kernels import Kernel, Product, Sum


class DummyKernel(Kernel):
    """
    A dummy kernel used for testing composite kernels.
    It just returns the difference r squared.
    """

    def __init__(self, *args: float) -> None:
        super().__init__(*args)
        self.params_number = len(args)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return r**2


def test_kernel_base_class() -> None:
    kernel = Kernel(1.0, 2.0)
    assert np.allclose(
        kernel.pars, np.array([1.0, 2.0])
    ), f"Expected [1.0, 2.0], got {kernel.pars}"

    try:
        kernel(np.array([1.0, 2.0]))
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        pass

    assert (
        repr(kernel) == "Kernel(1.0, 2.0)"
    ), f"Expected 'Kernel(1.0, 2.0)', got {repr(kernel)}"


def test_sum_kernel() -> None:
    k1 = DummyKernel(1.0)
    k2 = DummyKernel(2.0)
    sum_kernel = Sum(k1, k2)

    r = np.array([0.5, 1.0, 1.5])
    result = sum_kernel(r)
    expected = k1(r) + k2(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

    assert (
        repr(sum_kernel) == "DummyKernel(1.0) + DummyKernel(2.0)"
    ), f"Expected 'DummyKernel(1.0) + DummyKernel(2.0)', got {repr(sum_kernel)}"  # NOQA


def test_product_kernel() -> None:
    k1 = DummyKernel(1.0)
    k2 = DummyKernel(2.0)
    prod_kernel = Product(k1, k2)

    r = np.array([0.5, 1.0, 1.5])
    result = prod_kernel(r)
    expected = k1(r) * k2(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

    assert (
        repr(prod_kernel) == "DummyKernel(1.0) * DummyKernel(2.0)"
    ), f"Expected 'DummyKernel(1.0) * DummyKernel(2.0)', got {repr(prod_kernel)}"  # NOQA
