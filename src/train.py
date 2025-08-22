import argparse, json, os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

FEATURES = [
    "Gender","Married","Dependents","Education","Self_Employed",
    "ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term",
    "Credit_History","Property_Area"
]

def load_data(path: str, target: str):
    df = pd.read_csv(path)
    y = df[target].str.upper().map({'Y':1,'N':0})
    X = df[FEATURES]
    return X, y

def make_pipeline(X: pd.DataFrame):
    numeric_features = [c for c in X.columns if X[c].dtype != 'O']
    categorical_features = [c for c in X.columns if X[c].dtype == 'O']

    numeric_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", numeric_tf, numeric_features),
        ("cat", categorical_tf, categorical_features)
    ])

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    return Pipeline([("preprocess", pre), ("model", clf)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default="Loan_Status")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    X, y = load_data(args.csv, args.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = make_pipeline(X)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:,1]
    preds = (proba >= 0.5).astype(int)
    print("AUC:", roc_auc_score(y_test, proba))
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.outdir, "model.pkl"))
    with open(os.path.join(args.outdir, "columns.json"), "w") as f:
        json.dump({"columns": list(X.columns)}, f)
    print("✅ Model saved!")

if __name__ == "__main__":
    main()
