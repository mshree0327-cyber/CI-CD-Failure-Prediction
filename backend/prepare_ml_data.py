import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "train_temporal.csv"
TEST_FILE = DATA_DIR / "test_temporal.csv"

TRAIN_OUTPUT = DATA_DIR / "train_ml.csv"
TEST_OUTPUT = DATA_DIR / "test_ml.csv"

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("========== ML DATA PREPARATION ==========")

# Columns that contain no information in the dataset
columns_to_remove = [
    "gh_description_complexity",
    "gh_num_commits_in_push",
    "commit_frequency",
    "gh_num_pr_comments",
    "gh_num_issue_comments"
]

train_df = train_df.drop(columns=columns_to_remove)
test_df = test_df.drop(columns=columns_to_remove)

# Encode categorical feature using training data only
categories = train_df["git_prev_commit_resolution_status"].dropna().unique()

category_mapping = {
    category: index
    for index, category in enumerate(categories)
}

train_df["git_prev_commit_resolution_status"] = (
    train_df["git_prev_commit_resolution_status"]
    .map(category_mapping)
)

test_df["git_prev_commit_resolution_status"] = (
    test_df["git_prev_commit_resolution_status"]
    .map(category_mapping)
)

# Handle any unseen category in future/test data
train_df["git_prev_commit_resolution_status"] = (
    train_df["git_prev_commit_resolution_status"].fillna(-1)
)

test_df["git_prev_commit_resolution_status"] = (
    test_df["git_prev_commit_resolution_status"].fillna(-1)
)

train_df.to_csv(TRAIN_OUTPUT, index=False)
test_df.to_csv(TEST_OUTPUT, index=False)

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))

print("\nTraining columns:", len(train_df.columns))
print("Testing columns:", len(test_df.columns))

print("\nCategorical mapping:")
print(category_mapping)

print("\nTraining missing values:")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0])

print("\nTesting missing values:")
print(test_df.isnull().sum()[test_df.isnull().sum() > 0])

print("\nSaved:")
print(TRAIN_OUTPUT)
print(TEST_OUTPUT)