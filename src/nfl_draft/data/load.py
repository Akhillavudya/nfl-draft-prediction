"""Load the raw competition CSVs from the paths defined in config."""
import pandas as pd

from nfl_draft.config import TRAIN_CSV, TEST_CSV, SAMPLE_SUB


def load_train():
    """Read the labelled training set (has the Drafted target column)."""
    return pd.read_csv(TRAIN_CSV)


def load_test():
    """Read the unlabelled test set (no Drafted column)."""
    return pd.read_csv(TEST_CSV)


def load_sample_submission():
    """Read the submission template (Id + a placeholder Drafted column)."""
    return pd.read_csv(SAMPLE_SUB)
