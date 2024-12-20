"""Kernel classes for creating and combining covariance functions."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Kernel(object):
    """
    Base class for all kernel functions.

    Attributes:
        pars (numpy.ndarray): Array containing the kernel's hyperparameters.
    """

    def __init__(self, *args: float) -> None:
        """
        Initialize the kernel with its hyperparameters.

        Args:
            *args (float): Variable number of hyperparameter values.
        """
        self.pars: np.array[float] = np.array(args, dtype=float)  # type: ignore  # NOQA

    def __call__(self, r: np.ndarray) -> np.ndarray:  # type: ignore
        """
        Compute the kernel value between two data points.

        Args:
            r (numpy.ndarray): Difference between two data points.

        Returns:
            numpy.ndarray: Kernel value between the data points.

        Raises:
            NotImplementedError: Base class implementation doesn't define
                                a specific kernel function.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Create string representation of the kernel.

        Returns:
            str: String representation of the kernel type and hyperparameters.
        """
        return "{0}({1})".format(
            self.__class__.__name__, ", ".join(map(str, self.pars))
        )

    def __add__(self, b: "Kernel") -> "Sum":
        """
        Define addition operation between two kernels.

        Args:
            b (Kernel): Another kernel object.

        Returns:
            Sum: Sum kernel object representing the sum of the two kernels.
        """
        return Sum(self, b)

    def __radd__(self, b: "Kernel") -> "Sum":
        """
        Define right addition operation for compatibility.

        Args:
            b (Kernel): Another kernel object.

        Returns:
            Sum: Sum kernel object representing the sum of the two kernels.
        """
        return self.__add__(b)

    def __mul__(self, b: "Kernel") -> "Product":
        """
        Define multiplication operation between two kernels.

        Args:
            b (Kernel): Another kernel object.

        Returns:
            Product: Object representing the product of the two kernels.
        """
        return Product(self, b)

    def __rmul__(self, b: "Kernel") -> "Product":
        """
        Define right multiplication operation for compatibility.

        Args:
            b (Kernel): Another kernel object.

        Returns:
            Product: Object representing the product of the two kernels.
        """
        return self.__mul__(b)


class CompositeKernel(Kernel, ABC):
    """
    Abstract base class for composite kernels.

    This class provides a structure for kernels that combine other kernels.

    Attributes:
        base_kernels (list[Kernel]): List of the base kernels being combined.
    """

    def __init__(self, *kernels: Kernel) -> None:
        """
        Initialize the composite kernel with its base kernels.

        Args:
            *kernels (Kernel): Variable number of base kernel objects.
        """
        super().__init__()
        self.base_kernels: List[Kernel] = list(kernels)
        self.type: str = "composite"

    @abstractmethod
    def _operate(self, r: np.ndarray) -> np.ndarray:
        """
        Abstract method defining the operation of the composite kernel.

        This method should be implemented by subclasses to specify the
        way base kernels are combined (e.g., addition or multiplication).

        Args:
            r (numpy.ndarray): Difference between two data points.

        Returns:
            numpy.ndarray: Kernel value computed by the composite kernel.
        """
        pass

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the kernel value based on the composite operation.

        Calls the abstract `_operate` method to perform the specific
        combination of base kernels.

        Args:
            r (numpy.ndarray): Difference between two data points.

        Returns:
            numpy.ndarray: Kernel value computed by the composite kernel.
        """
        return self._operate(r)

    def __repr__(self) -> str:
        """
        Create string representation of the composite kernel.

        Returns:
            str: String representation of the composite kernel type
                 and base kernels.
        """
        op = "+" if isinstance(self, Sum) else "*"
        return op.join([{str(k)} for k in self.base_kernels])  # type: ignore


class Sum(CompositeKernel):
    """
    Sum kernel representing the sum of two base kernels.

    Attributes:
        base_kernels (list[Kernel]): List of the base kernels being summed.
    """

    def __init__(self, k1: Kernel, k2: Kernel) -> None:
        """
        Initialize the sum kernel with two base kernels.

        Args:
            k1 (Kernel): First base kernel to be summed.
            k2 (Kernel): Second base kernel to be summed.
        """
        super().__init__(k1, k2)
        self.pars = np.concatenate([k1.pars, k2.pars])
        self.params_number = sum(k.params_number for k in self.base_kernels)  # type: ignore  # NOQA

    def _operate(self, r: np.ndarray) -> np.ndarray:
        """
        Sum of the base kernels evaluated at the input.

        Args:
            r (numpy.ndarray): Difference between two data points.

        Returns:
            numpy.ndarray: Sum of the evaluations of the base kernels at r.
        """
        return sum(k(r) for k in self.base_kernels)  # type: ignore

    def __repr__(self) -> str:
        """
        Return string representation of the composite kernel.

        Returns:
            str: String representation of the composite kernel type
                 and base kernels.
        """
        return "{0} + {1}".format(self.base_kernels[0], self.base_kernels[1])

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the kernel value based on the composite operation.

        Args:
            r (numpy.ndarray): Difference between two data points.

        Returns:
            numpy.ndarray: Kernel value computed by the composite kernel.
        """
        return self.base_kernels[0](r) + self.base_kernels[1](r)


class Product(CompositeKernel):
    """
    Product kernel representing the product of two base kernels.

    This kernel computes the product of the evaluations of two base kernels.

    Attributes:
        base_kernels (list[Kernel]): List of the base kernels being multiplied.
    """

    def __init__(self, k1: Kernel, k2: Kernel) -> None:
        """
        Initialize the product kernel with two base kernels.

        Args:
            k1 (Kernel): First base kernel to be multiplied.
            k2 (Kernel): Second base kernel to be multiplied.
        """
        super().__init__(k1, k2)
        self.pars = np.concatenate([k1.pars, k2.pars])
        self.params_number = sum(k.params_number for k in self.base_kernels)  # type: ignore  # NOQA

    def _operate(self, r: np.ndarray) -> np.ndarray:
        """
        Product of the base kernels evaluated at the input.

        Args:
            r (numpy.ndarray): Difference between two data points.

        Returns:
            numpy.ndarray: Product of the evaluations of the base kernels at r.
        """
        return np.prod(k(r) for k in self)  # type: ignore

    def __repr__(self) -> str:
        """
        Create string representation of the composite kernel.

        Returns:
            str: String representation of the composite kernel type
                 and base kernels.
        """
        return "{0} * {1}".format(self.base_kernels[0], self.base_kernels[1])

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the kernel value based on the composite operation.

        Args:
            r (numpy.ndarray): Difference between two data points.

        Returns:
            numpy.ndarray: Kernel value computed by the composite kernel.
        """
        return self.base_kernels[0](r) * self.base_kernels[1](r)
