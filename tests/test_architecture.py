import random

import numpy as np
import pytest

from dnnga.architecture import (
    Architecture,
    architecture_from_any,
    check_within_bounds,
    random_architecture,
    validate_bounds,
)
from dnnga.exceptions import ConfigError, InvalidArchitectureError


class TestValidation:
    def test_valid(self):
        a = Architecture((4, 8, 2))
        assert a.layer_sizes == (4, 8, 2) and a.n_layers == 3

    def test_list_normalised_to_tuple(self):
        assert isinstance(Architecture([4, 8]).layer_sizes, tuple)

    @pytest.mark.parametrize(
        "bad", [(), (4, 0, 2), (4, -1), (4, 2.5), (4, 4.0), (4, True), ("a",), (None,)]
    )
    def test_rejects_bad_widths(self, bad):
        with pytest.raises(InvalidArchitectureError):
            Architecture(bad)

    def test_rejects_non_iterable_and_string(self):
        with pytest.raises(InvalidArchitectureError):
            Architecture(5)  # type: ignore[arg-type]
        with pytest.raises(InvalidArchitectureError):
            Architecture("123")  # type: ignore[arg-type]

    def test_accepts_numpy_integers_and_normalises_to_int(self):
        a = Architecture((np.int64(4), np.int32(8)))
        assert a.layer_sizes == (4, 8)
        assert all(type(w) is int for w in a.layer_sizes)


class TestImmutability:
    def test_frozen(self):
        a = Architecture((4, 8))
        with pytest.raises(Exception):
            a.layer_sizes = (1,)  # type: ignore[misc]

    def test_hashable_and_equal_by_value(self):
        assert Architecture((1, 2)) == Architecture([1, 2])
        assert len({Architecture((1, 2)), Architecture((1, 2)), Architecture((2, 1))}) == 2

    def test_replace_insert_remove_return_new_objects(self):
        a = Architecture((4, 8, 2))
        assert a.replace_layer(1, 99).layer_sizes == (4, 99, 2)
        assert a.insert_layer(1, 16).layer_sizes == (4, 16, 8, 2)
        assert a.insert_layer(0, 1).layer_sizes == (1, 4, 8, 2)
        assert a.insert_layer(3, 1).layer_sizes == (4, 8, 2, 1)
        assert a.remove_layer(1).layer_sizes == (4, 2)
        assert a.layer_sizes == (4, 8, 2), "original must be untouched"

    def test_removing_the_only_layer_is_invalid(self):
        with pytest.raises(InvalidArchitectureError):
            Architecture((4,)).remove_layer(0)


class TestBounds:
    def test_valid(self):
        validate_bounds(min_layers=1, max_layers=3, min_neurons=1, max_neurons=10)

    @pytest.mark.parametrize(
        "kw",
        [
            dict(min_layers=0, max_layers=3, min_neurons=1, max_neurons=10),
            dict(min_layers=5, max_layers=3, min_neurons=1, max_neurons=10),
            dict(min_layers=1, max_layers=3, min_neurons=0, max_neurons=10),
            dict(min_layers=1, max_layers=3, min_neurons=20, max_neurons=10),
        ],
    )
    def test_invalid_raise(self, kw):
        with pytest.raises(ConfigError):
            validate_bounds(**kw)

    def test_check_within_bounds(self):
        ok = dict(min_layers=1, max_layers=3, min_neurons=1, max_neurons=10)
        check_within_bounds(Architecture((4, 8)), **ok)
        for bad in (Architecture((1, 1, 1, 1)), Architecture((99,))):
            with pytest.raises(InvalidArchitectureError):
                check_within_bounds(bad, **ok)
        with pytest.raises(InvalidArchitectureError):
            check_within_bounds(
                Architecture((4,)), min_layers=2, max_layers=3, min_neurons=1, max_neurons=10
            )


class TestRandom:
    def test_respects_bounds(self):
        rng = random.Random(0)
        for _ in range(500):
            a = random_architecture(
                min_layers=2, max_layers=5, min_neurons=3, max_neurons=9, rng=rng
            )
            assert 2 <= a.n_layers <= 5 and all(3 <= w <= 9 for w in a.layer_sizes)

    def test_degenerate_bounds(self):
        a = random_architecture(
            min_layers=3, max_layers=3, min_neurons=5, max_neurons=5, rng=random.Random(0)
        )
        assert a.layer_sizes == (5, 5, 5)

    def test_reproducible(self):
        kw = dict(min_layers=1, max_layers=6, min_neurons=1, max_neurons=64)
        assert random_architecture(rng=random.Random(7), **kw) == random_architecture(
            rng=random.Random(7), **kw
        )

    def test_bad_bounds_raise(self):
        with pytest.raises(ConfigError):
            random_architecture(
                min_layers=5, max_layers=2, min_neurons=1, max_neurons=10, rng=random.Random(0)
            )


class TestFromAny:
    def test_passthrough_list_tuple_array(self):
        a = Architecture((4, 8))
        assert architecture_from_any(a) is a
        assert architecture_from_any([4, 8]).layer_sizes == (4, 8)
        assert architecture_from_any((4, 8)).layer_sizes == (4, 8)
        assert architecture_from_any(np.array([4, 8])).layer_sizes == (4, 8)

    @pytest.mark.parametrize("bad", [42, "48", b"48", None, object()])
    def test_rejects(self, bad):
        with pytest.raises(InvalidArchitectureError):
            architecture_from_any(bad)
