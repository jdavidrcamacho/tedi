"""Mean classes for creating and combining mean functions."""

from functools import wraps
from typing import List

import numpy as np


def array_input(f):
    """Define decorator to provide the __call__ methods with an array."""

    @wraps(f)
    def wrapped(self, t):
        t = np.atleast_1d(t)
        r = f(self, t)
        return r

    return wrapped


class MeanModel(object):
    """
    Base class for all mean functions.

    Attributes:
        pars (List[float]): List of parameters for the mean model.
    """

    _parsize: int = 0

    def __init__(self, *pars: float) -> None:
        """
        Initialize the mean model with given parameters.

        Args:
            *pars (float): Variable length argument list of parameters.
        """
        self.pars: List[float] = list(pars)

    def __repr__(self) -> str:
        """
        Return a string representation of the instance.

        Returns:
            str: String representation of the instance.
        """
        return "{0}({1})".format(
            self.__class__.__name__, ", ".join(map(str, self.pars))
        )

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the mean function.

        Args:
            t (numpy.ndarray): Time data points.

        Returns:
            numpy.ndarray: Mean value between the data points.


        Raises:
            NotImplementedError: Base class implementation doesn't define
                                a specific mean function.
        """
        raise NotImplementedError

    @classmethod
    def initialize(cls) -> "MeanModel":
        """
        Initialize an instance of the class, setting all parameters to 0.

        Returns:
            MeanModel: An instance of the class with all parameters set to 0.
        """
        return cls(*([0.0] * cls._parsize))

    def __add__(self, other: "MeanModel") -> "Sum":
        """
        Add two mean models together.

        Args:
            other (MeanModel): Another mean model to add.

        Returns:
            Sum: A new instance representing the sum of the two mean models.
        """
        return Sum(self, other)

    def __radd__(self, other: "MeanModel") -> "Sum":
        """
        Define right addition operation for compatibility.

        Args:
            other (MeanModel): Another mean model to add or a float.

        Returns:
            Sum: A new instance representing the sum of the two mean models.
        """
        return self.__add__(other)

    def __mul__(self, other: "MeanModel") -> "Product":
        """
        Define multiplication operation between two means.

        Args:
            other (MeanModel): Another mean object.

        Returns:
            Product: Object representing the product of the two means.
        """
        return Product(self, other)

    def __rmul__(self, other: "MeanModel") -> "Product":
        """
        Define right multiplication operation for compatibility.

        Args:
            other (MeanModel): Another mean object.

        Returns:
            Product: Object representing the product of the two means.
        """
        return self.__mul__(other)


class Sum(MeanModel):
    """
    Represent the sum of two mean functions.

    Attributes:
        m1 (MeanModel): The first mean model.
        m2 (MeanModel): The second mean model.
    """

    def __init__(self, m1: MeanModel, m2: MeanModel) -> None:
        """
        Initialize the Sum instance with two mean models.

        Args:
            m1 (MeanModel): The first mean model.
            m2 (MeanModel): The second mean model.
        """
        self.base_means: List[MeanModel] = [m1, m2]

    @property
    def _parsize(self) -> int:  # type: ignore
        """
        Return the total number of parameters in the summed mean models.

        Returns:
            int: Total number of parameters.
        """
        return self.base_means[0]._parsize + self.base_means[1]._parsize

    @property
    def pars(self) -> List[float]:  # type: ignore
        """
        Return the parameters of the summed mean models.

        Returns:
            List[float]: List of parameters.
        """
        return self.base_means[0].pars + self.base_means[1].pars

    def initialize(self):
        """Initialize the Sum instance."""
        pass

    def __repr__(self) -> str:
        """
        Return a string representation of the Sum instance.

        Returns:
            str: String representation of the Sum instance.
        """
        return "{0} + {1}".format(self.base_means[0], self.base_means[1])

    @array_input
    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the sum of the two mean models at the given input.

        Args:
            t (np.ndarray): Input data.

        Returns:
            np.ndarray: Sum of the two means evaluated at the input data.
        """
        return self.base_means[0](t) + self.base_means[1](t)  # type: ignore


class Product(MeanModel):
    """
    Represent the product of two mean functions.

    Attributes:
        m1 (MeanModel): The first mean model.
        m2 (MeanModel): The second mean model.
    """

    def __init__(self, m1: MeanModel, m2: MeanModel) -> None:
        """
        Initialize the Product instance with two mean models.

        Args:
            m1 (MeanModel): The first mean model.
            m2 (MeanModel): The second mean model.
        """
        self.base_means: List[MeanModel] = [m1, m2]

    @property
    def _parsize(self) -> int:  # type: ignore
        """
        Returns the total number of parameters in the multiplied mean models.

        Returns:
            int: Total number of parameters.
        """
        return self.base_means[0]._parsize + self.base_means[1]._parsize

    @property
    def pars(self) -> List[float]:  # type: ignore
        """
        Returns the parameters of the multiplied mean models.

        Returns:
            List[float]: List of parameters.
        """
        return self.base_means[0].pars + self.base_means[1].pars

    def initialize(self):
        """Initialize the Product instance."""
        pass

    def __repr__(self) -> str:
        """
        Return a string representation of the Product instance.

        Returns:
            str: String representation of the Product instance.
        """
        return "{0} * {1}".format(self.base_means[0], self.base_means[1])

    @array_input
    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the product of the two mean models at the given input.

        Args:
            t (np.ndarray): Input data.

        Returns:
            np.ndarray: Product of the two means evaluated at the input data.
        """
        return self.base_means[0](t) * self.base_means[1](t)  # type: ignore
