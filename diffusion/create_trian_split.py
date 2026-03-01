import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------
# CONFIG
# ----------------------------
INPUT_CSV = "universal_df.csv"
TRAIN_OUT = "train_split.csv"
TEST_OUT = "test_split.csv"
VAL_OUT = "val_split.csv"  # optional
RANDOM_STATE = 42

# ----------------------------
# Load
# ----------------------------
df = pd.read_csv(INPUT_CSV)

# Ensure label formatting
df["class"] = df["class"].astype(str).str.strip().str.lower()

# ----------------------------
# First split: train vs temp (80 / 20)
# ----------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["class"],
    random_state=RANDOM_STATE
)

# ----------------------------
# Second split: val / test (10 / 10)
# ----------------------------
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["class"],
    random_state=RANDOM_STATE
)

# ----------------------------
# Save
# ----------------------------
train_df.to_csv(TRAIN_OUT, index=False)
val_df.to_csv(VAL_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

print("Saved:")
print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")