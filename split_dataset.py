import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("Data/L1 Data set.csv")

print("Dataset Shape:", df.shape)
print("\nClass Distribution (before split):")
print(df["Label"].value_counts())
print(df["Label"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

# ─────────────────────────────────────────────
# 2. BASIC CHECK
# ─────────────────────────────────────────────
print("\nMissing Values:", df.isnull().sum().sum())

# ─────────────────────────────────────────────
# 3. SPLIT  →  70% Train | 15% Validation | 15% Test
#    stratify=Label ensures DoH/NonDoH ratio is
#    preserved in every split
# ─────────────────────────────────────────────

# Step 1: split off 70% train, 30% temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["Label"]
)

# Step 2: split the 30% temp into 15% val and 15% test (50/50 of temp)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["Label"]
)

# ─────────────────────────────────────────────
# 4. VERIFY SIZES AND CLASS BALANCE
# ─────────────────────────────────────────────
total = len(df)

print("\n─── Split Summary ───────────────────────────")
for name, split in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    pct = round(len(split) / total * 100, 1)
    print(f"\n{name} Set: {len(split)} rows ({pct}%)")
    print(split["Label"].value_counts().to_string())

# ─────────────────────────────────────────────
# 5. SAVE TO CSV
# ─────────────────────────────────────────────
train_df.to_csv("Data/train.csv", index=False)
val_df.to_csv("Data/validation.csv", index=False)
test_df.to_csv("Data/test.csv", index=False)

print("\n─────────────────────────────────────────────")
print("✓ train.csv      saved")
print("✓ validation.csv saved")
print("✓ test.csv       saved")
