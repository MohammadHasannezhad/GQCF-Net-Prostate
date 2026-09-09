import numpy as np
import pandas as pd
import openml

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, matthews_corrcoef
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------
def load_prostate_data():
    print("در حال دریافت دیتاست Prostate...")
    dataset = openml.datasets.get_dataset(1101)
    X_df, y_df, _, _ = dataset.get_data(dataset_format="dataframe")
    target_col = dataset.default_target_attribute
    
    if target_col in X_df.columns:
        y_raw = X_df[target_col]
        X_df = X_df.drop(columns=[target_col])
    else:
        y_raw = y_df
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    
    X_df_encoded = pd.get_dummies(X_df, drop_first=True)
    X_df_encoded = X_df_encoded.fillna(X_df_encoded.median())
    
    return X_df_encoded.to_numpy(dtype=np.float32), y

# ---------------------------------------------------------
models = {
    "SVM (Linear)": SVC(kernel='linear', probability=True, random_state=42),
    "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP (Classic Deep)": MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=500, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes (Gaussian)": GaussianNB()
}

# ---------------------------------------------------------
if __name__ == "__main__":
    X, y = load_prostate_data()
    INPUT_DIM = 256  
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "="*80)
    print("--- BENCHMARKING CLASSICAL ALGORITHMS ON PROSTATE DATASET (256 FEATURES) ---")
    print("="*80)
    
    summary_results = []

    for name, model in models.items():
        accs, precs, recs, f1s, mccs, aucs = [], [], [], [], [], []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            selector = SelectKBest(score_func=mutual_info_classif, k=INPUT_DIM)
            X_tr = selector.fit_transform(X_tr, y_tr)
            X_te = selector.transform(X_te)
            
            model.fit(X_tr, y_tr)
            
            preds = model.predict(X_te)
            probs = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else preds
            
            accs.append(accuracy_score(y_te, preds))
            precs.append(precision_score(y_te, preds, zero_division=0))
            recs.append(recall_score(y_te, preds, zero_division=0))
            f1s.append(f1_score(y_te, preds, average='macro'))
            mccs.append(matthews_corrcoef(y_te, preds))
            aucs.append(roc_auc_score(y_te, probs))
            
        summary_results.append({
            "Model": name,
            "Accuracy": f"{np.mean(accs):.4f}",
            "F1-Score": f"{np.mean(f1s):.4f}",
            "MCC": f"{np.mean(mccs):.4f}",
            "AUC": f"{np.mean(aucs):.4f}"
        })

    # ---------------------------------------------------------
    # ۴. نمایش جدول مقایسه‌ای نهایی
    # ---------------------------------------------------------
    df_results = pd.DataFrame(summary_results)
    print("\n" + df_results.to_string(index=False))
    print("\n" + "="*80)
