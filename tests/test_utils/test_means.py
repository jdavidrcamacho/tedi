import numpy as np

from src.tedi.utils.means import MeanModel, Product, Sum, array_input


def test_mean_model_initialization() -> None:
    model = MeanModel(1.0, 2.0)
    assert model.pars == [1.0, 2.0], "MeanModel initialization failed."
    assert repr(model) == "MeanModel(1.0, 2.0)", "MeanModel repr failed."


def test_mean_model_initialize() -> None:
    MeanModel._parsize = 2
    model = MeanModel.initialize()
    assert model.pars == [0.0, 0.0], "MeanModel initialize failed."


def test_sum_initialization() -> None:
    model1 = MeanModel(1.0)
    model2 = MeanModel(2.0)
    sum_model = model1 + model2
    assert isinstance(sum_model, Sum), "Sum initialization failed."
    assert sum_model.pars == [1.0, 2.0], "Sum pars failed."


def test_sum_repr() -> None:
    model1 = MeanModel(1.0)
    model2 = MeanModel(2.0)
    sum_model = model1 + model2
    assert (
        repr(sum_model) == "MeanModel(1.0) + MeanModel(2.0)"
    ), "Sum repr failed."  # NOQA


def test_sum_call() -> None:
    class MockMean(MeanModel):
        @array_input
        def __call__(self, t):
            return t + 1

    model1 = MockMean()
    model2 = MockMean()
    sum_model = model1 + model2

    result = sum_model(np.array([1.0, 2.0]))
    expected_result = np.array([4.0, 6.0])

    assert np.allclose(result, expected_result), "Sum call failed."


def test_product_initialization() -> None:
    model1 = MeanModel(1.0)
    model2 = MeanModel(2.0)
    product_model = model1 * model2
    assert isinstance(product_model, Product), "Product initialization failed."
    assert product_model.pars == [1.0, 2.0], "Product pars failed."


def test_product_repr() -> None:
    model1 = MeanModel(1.0)
    model2 = MeanModel(2.0)
    product_model = model1 * model2
    assert (
        repr(product_model) == "MeanModel(1.0) * MeanModel(2.0)"
    ), "Product repr failed."


def test_product_call() -> None:
    class MockMean(MeanModel):
        @array_input
        def __call__(self, t):
            return t + 1

    model1 = MockMean()
    model2 = MockMean()
    product_model = model1 * model2
    result = product_model(np.array([1.0, 2.0]))
    expected_result = np.array([4.0, 9.0])
    assert np.allclose(result, expected_result), "Product call failed."
