"""Guard the serving contract: valid probabilities and the missingness effect end-to-end."""
import pytest

from nfl_draft.config import MODELS_DIR
from nfl_draft.models.predict import predict_one

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "preprocess.joblib").exists(),
    reason="model artifacts missing — run `python -m nfl_draft.models.train` first",
)

_BASE = {"School": "Alabama", "Height": 74, "Weight": 210,
         "Player_Type": "offense", "Position_Type": "backs", "Position": "RB"}
_FULL = {**_BASE, "Age": 22, "Sprint_40yd": 4.5, "Vertical_Jump": 35,
         "Bench_Press_Reps": 20, "Broad_Jump": 120, "Agility_3cone": 7.0, "Shuttle": 4.2}


def test_probability_in_range():
    """The blended output must be a valid probability."""
    assert 0.0 <= predict_one(_FULL)["probability"] <= 1.0


def test_missingness_lowers_probability():
    """A blank-combine player should score lower than the same player with a full combine."""
    assert predict_one(_BASE)["probability"] < predict_one(_FULL)["probability"]
