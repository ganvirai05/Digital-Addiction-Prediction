print("Training pipeline started...")

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# ==============================
# STEP 1: LOAD DATA
# ==============================
print("\nLoading dataset...")
df = pd.read_csv("../data/final.csv")
print("Dataset loaded successfully")

# ==============================
# STEP 2: CLEAN DATA
# ==============================
print("\nCleaning data...")

df = df.drop(columns=["Timestamp", "Name"])

df.columns = [
    "age_group", "screen_time", "social_media", "gaming",
    "streaming", "phone_checks", "sleep_duration",
    "sleep_delay", "work_hours", "irritation",
    "mood_swings", "social_time", "addiction_risk"
]

df = df.dropna()

# Convert all text to lowercase
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.lower().str.strip()

print("Data cleaning completed")
# ==============================
# SMART ADDICTION
# ==============================
print("\nCreating smart addiction risk...")

def calculate_risk(row):
    score = 0
    
    # Screen time
    if row["screen_time"] > 7:
        score += 2
    elif row["screen_time"] > 4:
        score += 1

    # Phone checks
    if row["phone_checks"] > 15:
        score += 2
    elif row["phone_checks"] > 8:
        score += 1

    # Sleep
    if row["sleep_duration"] < 5:
        score += 2
    elif row["sleep_duration"] < 7:
        score += 1

    # Mood swings
    if row["mood_swings"] == "often":
        score += 2
    elif row["mood_swings"] == "sometimes":
        score += 1

    # Social interaction
    if row["social_time"] < 30:
        score += 2
    elif row["social_time"] < 60:
        score += 1

    # Final classification
    if score >= 7:
        return "high risk"
    elif score >= 4:
        return "moderate risk"
    else:
        return "low risk"


df["addiction_risk"] = df.apply(calculate_risk, axis=1)

print("New target created successfully")
print(df["addiction_risk"].value_counts())
# ==============================
# STEP 3: ENCODING
# ==============================
print("\nEncoding categorical variables...")

le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

print("Categorical columns:", list(categorical_cols))

for col in categorical_cols:
    print(f"Encoding {col}")
    df[col] = le.fit_transform(df[col])

print("Encoding completed")

# ==============================
# STEP 4: CHECK DATA TYPES
# ==============================
print("\nData types after encoding:\n")
print(df.dtypes)

# ==============================
# STEP 5: CLASS DISTRIBUTION
# ==============================
print("\nOriginal Class Distribution:")
print(df["addiction_risk"].value_counts())

# ==============================
# STEP 6: SPLIT FEATURES & TARGET
# ==============================
X = df.drop("addiction_risk", axis=1)
y = df["addiction_risk"]

# ==============================
# STEP 7: SCALING
# ==============================
print("\nApplying StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# STEP 8: APPLY SMOTE
# ==============================
print("\nApplying SMOTE to balance dataset...")

smote = SMOTE(k_neighbors=2,random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("\nAfter SMOTE Class Distribution:")
print(pd.Series(y_resampled).value_counts())

# ==============================
# STEP 9: TRAIN-TEST SPLIT
# ==============================
print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ==============================
# STEP 10: TRAIN MODEL
# ==============================
print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully")

# ==============================
# STEP 11: EVALUATION
# ==============================
print("\nEvaluating model...")

y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# ==============================
# STEP 12: SAVE MODEL
# ==============================
print("\nSaving model and scaler...")

joblib.dump(model, "../model/model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")

print("Model and scaler saved successfully")
