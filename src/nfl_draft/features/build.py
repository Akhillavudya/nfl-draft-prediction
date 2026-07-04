"""The feature pipeline: one definition of every transform, shared by train and serve."""
import numpy as np

from nfl_draft.config import DRILL_COLS, DRILL_VALS


def add_missingness(df, drill_cols=DRILL_COLS):
    """Turn each blank combine field into an explicit signal (the model's core edge)."""
    df = df.copy()
    for c in drill_cols:
        df[f"{c}_missing"] = df[c].isnull().astype(int)
    df["num_drills_missing"] = df[drill_cols].isnull().sum(axis=1)
    df["full_combine"] = (df["num_drills_missing"] == 0).astype(int)
    return df


def add_composites(df):
    """Build physics-flavoured ratios (BMI, power, speed score, etc.) from raw measurements."""
    df = df.copy()
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["power"] = df["Vertical_Jump"] * df["Weight"]
    df["speed_score"] = df["Weight"] / (df["Sprint_40yd"] ** 4 + 1e-9)
    df["agility_diff"] = df["Agility_3cone"] - df["Shuttle"]
    df["jump_ratio"] = df["Broad_Jump"] / (df["Vertical_Jump"] + 1e-9)
    df["bench_per_kg"] = df["Bench_Press_Reps"] / (df["Weight"] + 1e-9)
    return df


def add_position_z(df, pos_stats, drill_vals=DRILL_VALS):
    """Express each drill as a z-score relative to the player's position group."""
    df = df.copy()
    for d in drill_vals:
        mu = df["Position"].map(pos_stats[(d, "mean")])
        std = df["Position"].map(pos_stats[(d, "std")]).fillna(1).replace(0, 1)
        df[f"{d}_z_pos"] = (df[d] - mu) / std
    return df


def add_frequency(df, school_freq, pos_freq):
    """Attach how common each School / Position is (unseen categories map to 0)."""
    df = df.copy()
    df["School_freq"] = df["School"].map(school_freq).fillna(0)
    df["Position_freq"] = df["Position"].map(pos_freq).fillna(0)
    return df


def engineer_base(df, pos_stats, school_freq, pos_freq):
    """Run the four shared transforms in order; the common base for both models."""
    df = add_missingness(df)
    df = add_composites(df)
    df = add_position_z(df, pos_stats)
    df = add_frequency(df, school_freq, pos_freq)
    return df


def smooth_encode(schools, school_map, global_mean, count_map, smooth=10):
    """Smoothed target-encode School, pulling rare schools toward the global draft rate."""
    enc = []
    for s in schools:
        if s in school_map.index:
            n = count_map.get(s, 0)
            enc.append((n * school_map[s] + smooth * global_mean) / (n + smooth))
        else:
            enc.append(global_mean)
    return np.array(enc)
