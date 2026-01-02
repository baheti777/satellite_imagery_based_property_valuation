# ============================
# preprocessing.ipynb
# ============================

import os
import numpy as np
import pandas as pd

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = r"C:\Users\bhuvi\OneDrive\cdc_project"

TRAIN_CSV = os.path.join(BASE_DIR, "train_with_images.csv")
TEST_CSV  = os.path.join(BASE_DIR, "test_with_images.csv")

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
TEST_IMG_DIR  = os.path.join(BASE_DIR, "test")

# ----------------------------
# LOAD DATA
# ----------------------------
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# ----------------------------
# DROP UNWANTED COLUMNS
# ----------------------------
DROP_COLS = [
    'date', 'yr_renovated', 'zipcode',
    'waterfront', 'sqft_lot', 'sqft_lot15'
]

train_df.drop(columns=DROP_COLS, inplace=True)
test_df.drop(columns=DROP_COLS, inplace=True)

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
train_df['has_basement'] = (train_df['sqft_basement'] != 0).astype(int)
test_df['has_basement']  = (test_df['sqft_basement'] != 0).astype(int)

train_df['relative_size'] = train_df['sqft_living'] / train_df['sqft_living15']
test_df['relative_size']  = test_df['sqft_living'] / test_df['sqft_living15']

train_df['house_age'] = 2025 - train_df['yr_built']
test_df['house_age']  = 2025 - test_df['yr_built']

train_df.drop(columns=['sqft_basement', 'sqft_above', 'grade','yr_built'], inplace=True)
test_df.drop(columns=['sqft_basement', 'sqft_above', 'grade','yr_built'], inplace=True)

# ----------------------------
# HANDLE MISSING
# ----------------------------
train_df.dropna(subset=['price', 'image_path'], inplace=True)
test_df.dropna(subset=['image_path'], inplace=True)

# ----------------------------
# FIX IMAGE PATHS
# ----------------------------
train_df['image_path'] = train_df['image_path'].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, os.path.basename(x))
)

test_df['image_path'] = test_df['image_path'].apply(
    lambda x: os.path.join(TEST_IMG_DIR, os.path.basename(x))
)

# ----------------------------
# LOG TARGET
# ----------------------------
train_df['price_log'] = np.log1p(train_df['price'])

# ----------------------------
# SAVE CLEAN DATA
# ----------------------------
train_df.to_csv(os.path.join(BASE_DIR, "train_processed.csv"), index=False)
test_df.to_csv(os.path.join(BASE_DIR, "test_processed.csv"), index=False)

print("✅ preprocessing complete")

############################################################################################
