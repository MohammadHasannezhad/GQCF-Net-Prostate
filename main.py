import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import openml
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, matthews_corrcoef, roc_curve, auc, confusion_matrix
)
import pennylane as qml

# ---------------------------------------------------------
NUM_QUBITS = 8
INPUT_DIM = 256        # 2^8 = 256 features mapped via Amplitude Encoding
NUM_LAYERS = 3         # StronglyEntanglingLayers depth
LATENT_DIM = 16
EPOCHS = 60
OPTIMAL_THRESHOLD = 0.48

def load_prostate_data():
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
    
    X = X_df_encoded.to_numpy(dtype=np.float32)
    return X, y

# ---------------------------------------------------------
dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface="torch")
def gqcf_quantum_circuit(inputs, weights):
    # Amplitude Encoding for 256 input features across 8 qubits
    qml.AmplitudeEmbedding(features=inputs, wires=range(NUM_QUBITS), pad_with=0.0, normalize=True)
    
    # Strongly Entangling Variational Layers
    qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
    
    # Pauli-Z Observables Measurement
    return [qml.expval(qml.PauliZ(w)) for w in range(NUM_QUBITS)]

# ---------------------------------------------------------
class GQCFNet(nn.Module):
    def __init__(self, num_qubits=NUM_QUBITS, num_layers=NUM_LAYERS, latent_dim=LATENT_DIM):
        super(GQCFNet, self).__init__()
        
        self.q_weights = nn.Parameter(torch.randn(num_layers, num_qubits, 3) * 0.1)
        
        self.classical_backbone = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        self.quantum_projection = nn.Sequential(
            nn.Linear(num_qubits, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU()
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(latent_dim, 2)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        h_classical = self.classical_backbone(x)
        
        q_features = []
        for i in range(batch_size):
            out = gqcf_quantum_circuit(x[i], self.q_weights)
            q_features.append(torch.stack(out))
            
        q_features = torch.stack(q_features).float()
        h_quantum = self.quantum_projection(q_features)
        
        alpha = self.gate_net(x)  # Shape: [batch_size, 1]
        
        h_fused = alpha * h_quantum + (1.0 - alpha) * h_classical
        
        logits = self.classifier(h_fused)
        return logits, alpha

# ---------------------------------------------------------
# ---------------------------------------------------------
if __name__ == "__main__":
    X, y = load_prostate_data()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, precs, recs, f1s, mccs, aucs = [], [], [], [], [], []
    mean_alphas = []
    
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    total_cm = np.zeros((2, 2), dtype=int)
    
    print("\n================ Running GQCF-Net on Prostate Dataset (5-Fold CV) ================")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        selector = SelectKBest(score_func=mutual_info_classif, k=INPUT_DIM)
        X_tr = selector.fit_transform(X_tr, y_tr)
        X_te = selector.transform(X_te)
        
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_te_t = torch.tensor(X_te, dtype=torch.float32)
        y_te_t = torch.tensor(y_te, dtype=torch.long)
        
        n_samples = len(y_tr)
        class_counts = np.bincount(y_tr)
        weights = n_samples / (2.0 * class_counts)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        
        model = GQCFNet(num_qubits=NUM_QUBITS, num_layers=NUM_LAYERS, latent_dim=LATENT_DIM)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            logits, _ = model(X_tr_t)
            loss = criterion(logits, y_tr_t)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        model.eval()
        with torch.no_grad():
            test_logits, test_alphas = model(X_te_t)
            test_probs = torch.softmax(test_logits, dim=1)[:, 1].numpy()
            avg_alpha = test_alphas.mean().item()
            
        preds = (test_probs >= OPTIMAL_THRESHOLD).astype(int)
        
        acc = accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)
        f1 = f1_score(y_te, preds, average='macro')
        mcc = matthews_corrcoef(y_te, preds)
        auc_val = roc_auc_score(y_te, test_probs)
        
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        mccs.append(mcc)
        aucs.append(auc_val)
        mean_alphas.append(avg_alpha)
        
        total_cm += confusion_matrix(y_te, preds)
        
        fpr, tpr, _ = roc_curve(y_te, test_probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC Fold {fold} (AUC = {auc_val:.2f})')
        
        print(f"Fold {fold} | Acc: {acc*100:.2f}% | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f} | AUC: {auc_val:.4f} | Gate α: {avg_alpha:.4f}")

    # ---------------------------------------------------------
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="ROC Curve - GQCF-Net on Prostate Dataset",
           xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right")
    plt.savefig("GQCF_Prostate_ROC_Curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control', 'Prostate Cancer'], 
                yticklabels=['Control', 'Prostate Cancer'])
    plt.title("Aggregated Confusion Matrix - GQCF-Net (Prostate)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("GQCF_Prostate_Confusion_Matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    print("\n================ FINAL EVALUATION SUMMARY (GQCF-Net Prostate) ================")
    print(f"Mean Accuracy  : {np.mean(accs)*100:.2f}% (+/- {np.std(accs)*100:.2f}%)")
    print(f"Mean Precision : {np.mean(precs):.4f}")
    print(f"Mean Recall    : {np.mean(recs):.4f}")
    print(f"Mean F1-Score  : {np.mean(f1s):.4f}")
    print(f"Mean MCC       : {np.mean(mccs):.4f}")
    print(f"Mean ROC-AUC   : {np.mean(aucs):.4f}")
    print(f"Mean Gate Alpha: {np.mean(mean_alphas):.4f}")
    print("=============================================================================")
    print("[INFO] 'GQCF_Prostate_ROC_Curve.png' and 'GQCF_Prostate_Confusion_Matrix.png' saved successfully.")
