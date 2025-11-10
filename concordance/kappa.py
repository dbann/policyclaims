import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# --- User settings ---
# Option 1: If file is in current directory
excel_path = "concordance/EC_all_abstracts_policy_claims_terms_LLM_trial db_ec2_0812.xlsx"

# Option 2: If file is in a subdirectory (uncomment and modify as needed)
# excel_path = "concordance tests/EC_all_abstracts_policy_claims_terms_LLM_trial db_ec2_0812.xlsx"

# Option 3: Full path (uncomment and modify as needed)
# excel_path = "/full/path/to/your/EC_all_abstracts_policy_claims_terms_LLM_trial db_ec2_0812.xlsx"

reviewer1_col = "MW review"
reviewer2_col = "llm_policy_claim"

# Check if file exists
import os
if not os.path.exists(excel_path):
    print(f"ERROR: File not found: {excel_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # Try to find the file in common locations
    possible_paths = [
        "EC_all_abstracts_policy_claims_terms_LLM_trial db_ec2_0812.xlsx",
        "concordance tests/EC_all_abstracts_policy_claims_terms_LLM_trial db_ec2_0812.xlsx",
        "./EC_all_abstracts_policy_claims_terms_LLM_trial db_ec2_0812.xlsx"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found file at: {path}")
            excel_path = path
            break
    else:
        print("Could not locate the Excel file. Please check the path.")
        exit()

# --- Load data ---
# Specify the sheet name 'in' since the Excel file has multiple sheets
df = pd.read_excel(excel_path, sheet_name='in')

print(f"Total rows loaded: {len(df)}")
print(f"Columns in dataset: {list(df.columns)}")

# Check if the columns exist
if reviewer1_col not in df.columns:
    print(f"ERROR: Column '{reviewer1_col}' not found in dataset")
    print(f"Available columns: {list(df.columns)}")
    exit()

if reviewer2_col not in df.columns:
    print(f"ERROR: Column '{reviewer2_col}' not found in dataset") 
    print(f"Available columns: {list(df.columns)}")
    exit()

# --- Keep only rows where both reviewers have a value ---
print(f"\nRows with non-null {reviewer1_col}: {df[reviewer1_col].notna().sum()}")
print(f"Rows with non-null {reviewer2_col}: {df[reviewer2_col].notna().sum()}")

df_valid = df.dropna(subset=[reviewer1_col, reviewer2_col]).copy()
print(f"Rows with both reviewers having values: {len(df_valid)}")

if len(df_valid) == 0:
    print("ERROR: No rows found where both reviewers have values!")
    print(f"Sample values from {reviewer1_col}: {df[reviewer1_col].dropna().head()}")
    print(f"Sample values from {reviewer2_col}: {df[reviewer2_col].dropna().head()}")
    exit()

# --- Check the data types and values ---
print(f"\nUnique values in {reviewer1_col}: {df_valid[reviewer1_col].unique()}")
print(f"Data type of {reviewer1_col}: {df_valid[reviewer1_col].dtype}")
print(f"\nUnique values in {reviewer2_col}: {df_valid[reviewer2_col].unique()}")
print(f"Data type of {reviewer2_col}: {df_valid[reviewer2_col].dtype}")

# --- Robust boolean conversion ---
def to_bool(val):
    """Convert various boolean representations to True/False"""
    if pd.isna(val):
        return pd.NA
    
    # Handle numeric values (float/int)
    if isinstance(val, (int, float)):
        if val == 1.0 or val == 1:
            return True
        elif val == 0.0 or val == 0:
            return False
        else:
            print(f"Warning: Unrecognized numeric boolean value: {val}")
            return pd.NA
    
    # Handle boolean values that are already boolean type
    if isinstance(val, bool):
        return val
    
    # Convert to string and clean for string representations
    val_str = str(val).strip().lower()
    
    # Handle string representations
    if val_str in ["true", "1", "1.0", "yes", "y"]:
        return True
    elif val_str in ["false", "0", "0.0", "no", "n"]:
        return False
    else:
        print(f"Warning: Unrecognized boolean value: '{val}' (type: {type(val)})")
        return pd.NA

# Convert both columns to boolean
print("\nConverting to boolean values...")
df_valid[reviewer1_col + '_bool'] = df_valid[reviewer1_col].apply(to_bool)
df_valid[reviewer2_col + '_bool'] = df_valid[reviewer2_col].apply(to_bool)

# --- Remove any rows where conversion failed ---
initial_count = len(df_valid)
df_valid = df_valid.dropna(subset=[reviewer1_col + '_bool', reviewer2_col + '_bool'])
final_count = len(df_valid)

print(f"Rows after boolean conversion: {final_count}")
if initial_count != final_count:
    print(f"Removed {initial_count - final_count} rows due to failed boolean conversion")

if len(df_valid) == 0:
    print("ERROR: No valid rows remaining after boolean conversion!")
    exit()

# Use the boolean columns for analysis
rev1_bool = df_valid[reviewer1_col + '_bool']
rev2_bool = df_valid[reviewer2_col + '_bool']

# --- Calculate percent agreement ---
agreement = (rev1_bool == rev2_bool).mean()
print(f"\nPercent agreement: {agreement * 100:.1f}%")

# --- Calculate Cohen's kappa ---
kappa = cohen_kappa_score(rev1_bool, rev2_bool)
print(f"Cohen's kappa: {kappa:.3f}")

# Interpret kappa
if kappa < 0:
    interpretation = "Poor (worse than random)"
elif kappa < 0.20:
    interpretation = "Slight"
elif kappa < 0.40:
    interpretation = "Fair"
elif kappa < 0.60:
    interpretation = "Moderate"
elif kappa < 0.80:
    interpretation = "Substantial"
else:
    interpretation = "Almost perfect"

print(f"Kappa interpretation: {interpretation}")

# --- Show confusion matrix ---
cm = confusion_matrix(rev1_bool, rev2_bool, labels=[True, False])
print("\nConfusion matrix (rows: MW review, columns: llm_policy_claim):")
cm_df = pd.DataFrame(cm, index=["True", "False"], columns=["True", "False"])
print(cm_df)

# --- Calculate additional metrics ---
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\nAdditional metrics:")
print(f"Sensitivity (True Positive Rate): {sensitivity:.3f}")
print(f"Specificity (True Negative Rate): {specificity:.3f}")
print(f"Positive Predictive Value: {ppv:.3f}")
print(f"Negative Predictive Value: {npv:.3f}")

# --- Show mismatches for debugging ---
mismatches = df_valid[rev1_bool != rev2_bool]
print(f"\nNumber of mismatches: {len(mismatches)}")

if not mismatches.empty:
    print("\nMismatches (showing original values):")
    mismatch_display = mismatches[['scopus_id', reviewer1_col, reviewer2_col, 
                                   reviewer1_col + '_bool', reviewer2_col + '_bool']].copy()
    print(mismatch_display.to_string(index=False))
    
    # Count types of mismatches
    mw_true_llm_false = len(mismatches[mismatches[reviewer1_col + '_bool'] & ~mismatches[reviewer2_col + '_bool']])
    mw_false_llm_true = len(mismatches[~mismatches[reviewer1_col + '_bool'] & mismatches[reviewer2_col + '_bool']])
    
    print(f"\nMismatch breakdown:")
    print(f"MW=True, LLM=False: {mw_true_llm_false}")
    print(f"MW=False, LLM=True: {mw_false_llm_true}")

print(f"\nAnalysis complete. Total valid comparisons: {len(df_valid)}")