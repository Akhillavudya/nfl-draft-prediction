"""Guard the feature invariants the model edge depends on: flags fire, blanks stay NaN."""
import numpy as np
import pandas as pd

from nfl_draft.config import DRILL_COLS
from nfl_draft.features.build import add_missingness, add_composites


def test_missingness_flags_fire():
    """Row 0 has every drill, row 1 has none → flags and counts must reflect that."""
    df = pd.DataFrame({c: [1.0, np.nan] for c in DRILL_COLS})
    out = add_missingness(df)
    assert out["Sprint_40yd_missing"].tolist() == [0, 1]
    assert out["num_drills_missing"].tolist() == [0, len(DRILL_COLS)]
    assert out["full_combine"].tolist() == [1, 0]


def test_blank_drill_stays_nan():
    """A missing measurement must remain NaN, not be coerced to 0."""
    df = pd.DataFrame({c: [1.0, np.nan] for c in DRILL_COLS})
    out = add_missingness(df)
    assert out["Sprint_40yd"].isna().tolist() == [False, True]


def test_composites_do_not_zero_fill():
    """A composite built from a missing drill must stay NaN, preserving the signal."""
    row = {c: [np.nan] for c in DRILL_COLS}
    row.update({"Weight": [100.0], "Height": [2.0]})
    out = add_composites(pd.DataFrame(row))
    assert np.isnan(out["power"].iloc[0])
