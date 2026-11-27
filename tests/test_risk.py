
import pytest

from src.prediction.risk_utils import risk_label


@pytest.mark.parametrize(
    "probability, expected",
    [
        (0, "LOW"),
        (20, "LOW"),
        (40, "LOW"),
        (41, "MODERATE"),
        (65, "MODERATE"),
        (66, "HIGH"),
        (85, "HIGH"),
        (86, "SEVERE"),
        (100, "SEVERE"),
    ],
)
def test_risk_label(probability, expected):
    assert risk_label(probability) == expected
