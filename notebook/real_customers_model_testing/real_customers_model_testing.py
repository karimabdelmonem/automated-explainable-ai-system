## Step 1: Import Required Libraries


import os
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

## Step 2: Load Tabular Deep Learning Model

model_url = "https://raw.githubusercontent.com/karimabdelmonem/automated-explainable-ai-system/main/Models/tabular_dl_model.keras"

if not os.path.exists("tabular_dl_model.keras"):
    print("[*] Downloading model...")
    os.system(f"wget '{model_url}' -O tabular_dl_model.keras")

if os.path.exists("tabular_dl_model.keras"):
    print("[✓] Model ready!")
else:
    raise FileNotFoundError("[-] Model download failed.")

## Step 3: Load Reference Data & Fit Feature Scaler

BASE_URL = "https://raw.githubusercontent.com/karimabdelmonem/automated-explainable-ai-system/main/data/processed/"

if not os.path.exists('data_processed.parquet'):
    print("[*] Downloading data_processed.parquet...")
    r = requests.get(BASE_URL + "data_processed.parquet")
    with open('data_processed.parquet', 'wb') as f:
        f.write(r.content)
    print("[✓] Downloaded!")
else:
    print("[✓] Already exists.")

df_processed = pd.read_parquet('data_processed.parquet')
X_ref        = df_processed.drop(columns=['default'])
train_cols   = X_ref.columns.tolist()

scaler = StandardScaler()
scaler.fit(X_ref)
print("[✓] Scaler fitted!")
print(f"    age={scaler.mean_[0]:.2f} | income={scaler.mean_[1]:.2f} | creditscore={scaler.mean_[3]:.2f}")

## Step 4: Preprocess & Encode New Customer Features

def preprocess_final(raw_df):
    df = raw_df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    cols_to_drop = ['loanid', 'default', 'model_prediction', 'risk_score']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    for col in ['hasmortgage', 'hasdependents', 'hascosigner']:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0)

    edu_order = ['High School', "Bachelor's", "Master's", 'PhD']
    if 'education' in df.columns:
        df['education'] = df['education'].map(
            {v: i for i, v in enumerate(edu_order)}
        ).fillna(0).astype(int)

    df = pd.get_dummies(df, columns=['employmenttype', 'maritalstatus', 'loanpurpose'], drop_first=True)

    df["loan_to_income"]   = df["loanamount"] / (df["income"] + 1)
    df["monthly_income"]   = df["income"] / 12
    df["employment_ratio"] = df["monthsemployed"] / (df["age"] * 12 + 1)
    df["creditscore_band"] = pd.cut(
        df["creditscore"], bins=[0, 580, 670, 740, 850], labels=[0, 1, 2, 3]
    ).astype(float).fillna(0).astype(int)
    df["high_risk_flag"] = ((df["dtiratio"] > 0.45) & (df["creditscore"] < 600)).astype(int)

    for col in train_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[train_cols]

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return scaler.transform(df)

print("[✓] preprocess_final defined.")

## Step 5: Load New Customers Data from Google Drive

from google.colab import files

uploaded = files.upload()
file_path = list(uploaded.keys())[0]

my_real_data = pd.read_csv(file_path)
print(f"[✓] Data loaded: {my_real_data.shape}")

## Step 6: Predict Loan Default Risk & Generate Risk Scores

from tensorflow.keras.models import load_model

deep_model = load_model('tabular_dl_model.keras')
print("[✓] Model loaded!")

X_input   = preprocess_final(my_real_data)
deep_prob = deep_model.predict(X_input).ravel()
deep_pred = (deep_prob > 0.5).astype(int)

cols_to_clear = ['Risk_Score', 'Model_Prediction']
my_real_data  = my_real_data.drop(columns=[c for c in cols_to_clear if c in my_real_data.columns])
my_real_data['Model_Prediction'] = deep_pred
my_real_data['Risk_Score']       = np.round(deep_prob, 4)

if 'Default' in my_real_data.columns:
    acc = accuracy_score(my_real_data['Default'], deep_pred)
    print(f"\n[✓] Model Accuracy: {acc:.2%}")

print("\n--- Prediction Results ---")
print(f"Total    : {len(my_real_data)}")
print(f"Default 0: {(deep_pred == 0).sum()}")
print(f"Default 1: {(deep_pred == 1).sum()}")
print(f"\nRisk Score → min: {deep_prob.min():.4f} | max: {deep_prob.max():.4f} | mean: {deep_prob.mean():.4f}")

display(my_real_data[['LoanID','CreditScore','DTIRatio','Income','EmploymentType','Model_Prediction','Risk_Score']].head(10))

## Step 7: Explain Predictions Using Rule-Based Decision Factors


def explain_prediction(row):
    reasons = []

   
    if row['CreditScore'] < 580:
        reasons.append(f"CreditScore ضعيف جداً ({row['CreditScore']})")
    elif row['CreditScore'] < 670:
        reasons.append(f"CreditScore متوسط ({row['CreditScore']})")
    elif row['CreditScore'] >= 740:
        reasons.append(f"CreditScore ممتاز ({row['CreditScore']})")

    if row['DTIRatio'] > 0.6:
        reasons.append(f"نسبة الديون للدخل عالية جداً ({row['DTIRatio']})")
    elif row['DTIRatio'] > 0.45:
        reasons.append(f"نسبة الديون للدخل مرتفعة ({row['DTIRatio']})")
    elif row['DTIRatio'] < 0.3:
        reasons.append(f"نسبة الديون للدخل منخفضة ✓ ({row['DTIRatio']})")


    if row['Income'] < 30000:
        reasons.append(f"دخل منخفض ({row['Income']:,})")
    elif row['Income'] > 100000:
        reasons.append(f"دخل مرتفع ✓ ({row['Income']:,})")


    if row['EmploymentType'] == 'Unemployed':
        reasons.append("عاطل عن العمل")
    elif row['EmploymentType'] == 'Full-time':
        reasons.append("موظف full-time ✓")


    loan_to_income = row['LoanAmount'] / (row['Income'] + 1)
    if loan_to_income > 3:
        reasons.append(f"القرض كبير جداً مقارنة بالدخل ({loan_to_income:.1f}x)")
    elif loan_to_income < 1:
        reasons.append(f"القرض مناسب للدخل ✓ ({loan_to_income:.1f}x)")

    if row['MonthsEmployed'] < 12:
        reasons.append(f"خبرة عمل قليلة ({row['MonthsEmployed']} شهر)")
    elif row['MonthsEmployed'] > 60:
        reasons.append(f"خبرة عمل طويلة ✓ ({row['MonthsEmployed']} شهر)")


    if row['Model_Prediction'] == 1:
        verdict = "⚠️ خطر default — بسبب: " + " | ".join(reasons)
    else:
        verdict = "✅ لا خطر — بسبب: " + " | ".join(reasons)

    return verdict


my_real_data['Explanation'] = my_real_data.apply(explain_prediction, axis=1)


display(my_real_data[['Model_Prediction','Risk_Score','Explanation']].head(10))
