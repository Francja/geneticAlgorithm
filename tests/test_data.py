import numpy as np
import pytest

from dnnga import TrainingData


def _ok(**kw):
    base = dict(
        X_train=np.zeros((10, 3)),
        y_train=np.arange(10) % 2,
        X_val=np.zeros((4, 3)),
        y_val=np.arange(4) % 2,
    )
    base.update(kw)
    return TrainingData(**base)


class TestTrainingData:
    def test_valid_and_properties(self):
        d = _ok()
        assert d.n_features == 3 and d.n_outputs == 2 and not d.has_test

    def test_lists_are_converted_to_arrays(self):
        d = TrainingData([[1, 2], [3, 4]], [0, 1], [[1, 2]], [0])
        assert isinstance(d.X_train, np.ndarray)

    def test_regression_outputs(self):
        assert (
            _ok(y_train=np.arange(10.0), y_val=np.arange(4.0), task="regression").n_outputs
            == 1
        )
        d = _ok(
            y_train=np.zeros((10, 3)), y_val=np.zeros((4, 3)), task="regression"
        )
        assert d.n_outputs == 3

    @pytest.mark.parametrize(
        "kw",
        [
            dict(task="clustering"),
            dict(X_train=np.zeros(10)),
            dict(X_val=np.zeros((4, 2))),
            dict(y_train=np.zeros(9)),
            dict(y_val=np.zeros(3)),
            dict(X_train=np.zeros((0, 3)), y_train=np.zeros(0)),
            dict(X_val=np.zeros((0, 3)), y_val=np.zeros(0)),
            dict(X_test=np.zeros((2, 3))),
            dict(y_test=np.zeros(2)),
            dict(X_test=np.zeros((2, 5)), y_test=np.zeros(2)),
            dict(X_test=np.zeros((2, 3)), y_test=np.zeros(3)),
            dict(X_test=np.zeros((0, 3)), y_test=np.zeros(0)),
        ],
    )
    def test_invalid_inputs_raise_value_error(self, kw):
        with pytest.raises(ValueError):
            _ok(**kw)

    def test_test_view_swaps_test_into_the_validation_slot(self):
        d = _ok(X_test=np.ones((5, 3)), y_test=np.ones(5))
        v = d.test_view()
        assert len(v.X_val) == 5 and np.all(v.X_val == 1) and not v.has_test
        assert v.X_train is d.X_train

    def test_test_view_without_test_split_raises(self):
        with pytest.raises(ValueError):
            _ok().test_view()
