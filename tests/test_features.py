"""Regression tests for the classical DIP feature contract."""
import numpy as np

from ml.features.extractor import FEATURE_NAMES, extract_features


def test_feature_vector_is_finite_and_named() -> None:
    """Every supported face crop produces one finite value per feature name."""
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    vector, names = extract_features(image)
    assert len(vector) == len(names) == len(FEATURE_NAMES)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_feature_vector_is_deterministic() -> None:
    """Repeated extraction does not depend on process state or random values."""
    image = np.random.default_rng(42).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    first, first_names = extract_features(image)
    second, second_names = extract_features(image)
    np.testing.assert_array_equal(first, second)
    assert first_names == second_names