import pandas as pd

# Quick Submission Format Checker

# Load the saved submission
test_df = pd.read_csv('test.csv')
sub = pd.read_csv('submission.csv')

# Show a preview
print(sub.head())
print(sub.columns)

# Check column names
assert list(sub.columns) == ['id', 'Score'], "❌ Column names must be exactly ['id', 'Score']!"

# Check for duplicates
assert sub['id'].is_unique, "❌ Duplicate IDs found in submission!"

# Check for missing values
assert not sub.isnull().any().any(), "❌ There are missing values in submission!"

# Check row count matches test set
assert len(sub) == len(test_df), "❌ Submission row count does not match test set!"

print("✅ Submission file format is correct and ready for Kaggle upload!")
