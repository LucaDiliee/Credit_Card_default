# Script_Random_Forest_NO_LASSO

import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, cohen_kappa_score, classification_report, confusion_matrix,
    roc_curve
)
from sklearn.preprocessing import StandardScaler

# ===== CONFIGURATION =====
DATASET_PATH = "Dataset3.csv"
TARGET_COLUMN = "default.payment.next.month"
LOG_DIR = "logs"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAVE_LOG = True
THRESHOLD = 0.4

# ===== CREATE LOG FOLDER=====
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(LOG_DIR, f"log_rf_full_features_{timestamp}.txt")

def log(msg):
    print(msg)
    if SAVE_LOG:
        with open(log_file, "a") as f:
            f.write(msg + "\n")

# ===== PREPROCESSING =====
df = pd.read_csv(DATASET_PATH, sep=';')
df = df.drop(columns=['ID'], errors='ignore')
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop(columns=TARGET_COLUMN)
y = df[TARGET_COLUMN]
feature_names = X.columns  # all available features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

log(f"Numero totale di feature usate: {len(feature_names)}")

# ===== RANDOM FOREST =====
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

y_prob = rf.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

# ===== METRICS =====
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Log Loss": log_loss(y_test, y_prob),
    "Cohen Kappa": cohen_kappa_score(y_test, y_pred)
}
log("\n=== RANDOM FOREST SU TUTTE LE FEATURE ===")
for k, v in metrics.items():
    log(f"{k}: {v:.4f}")

# ===== REPORT =====
report = classification_report(y_test, y_pred)
log("\nClassification Report:\n" + report)
with open(os.path.join(LOG_DIR, "classification_report_rf_full.txt"), "w") as f:
    f.write(report)

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
log(f"\nConfusion Matrix:\nTN={tn} FP={fp} FN={fn} TP={tp}")

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix (RF full feature set)")
plt.colorbar()
plt.xticks([0, 1], ['No Default', 'Default'])
plt.yticks([0, 1], ['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('True')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(LOG_DIR, "confusion_matrix_rf_full.png"))
plt.close()

# ===== ROC CURVE =====
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC (RF full feature set)')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(LOG_DIR, "roc_curve_rf_full.png"))
plt.close()

# ===== FEATURE IMPORTANCE RF =====
importances = rf.feature_importances_
feat_imp_rf = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

feat_imp_rf.to_csv(os.path.join(LOG_DIR, "rf_feature_importance_full.csv"), index=False, sep=';')

plt.figure(figsize=(10, 6))
top_feats = feat_imp_rf.head(25)
plt.barh(top_feats["Feature"][::-1], top_feats["Importance"][::-1])
plt.xlabel("Importanza")
plt.title("Top 25 Feature Importances (RF full feature set)")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(LOG_DIR, "rf_feature_importance_full.png"))
plt.close()

# Script_Random_Forest_with_LASSO_

import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, cohen_kappa_score, classification_report, confusion_matrix,
    roc_curve
)
from sklearn.preprocessing import StandardScaler

# ===== CONFIGURATION =====
DATASET_PATH = "Dataset3.csv"
TARGET_COLUMN = "default.payment.next.month"
LOG_DIR = "logs"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAVE_LOG = True
THRESHOLD = 0.4
C = 0.01

# ===== CREATE LOG FOLDER =====
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(LOG_DIR, f"log_lasso_rf_{timestamp}.txt")

def log(msg):
    print(msg)
    if SAVE_LOG:
        with open(log_file, "a") as f:
            f.write(msg + "\n")

# ===== PREPROCESSING =====
df = pd.read_csv(DATASET_PATH, sep=';')
df = df.drop(columns=['ID'], errors='ignore')
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop(columns=TARGET_COLUMN)
y = df[TARGET_COLUMN]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ===== MODEL LASSO =====
lasso = LogisticRegression(
    penalty='l1', solver='saga', C=C,
    random_state=RANDOM_STATE, max_iter=4000
)
lasso.fit(X_train, y_train)

coefs = lasso.coef_[0]
feature_names = X.columns
selected_features = feature_names[coefs != 0]
log(f"Feature selezionate da Lasso: {len(selected_features)} / {len(feature_names)}")

# CoefficientS Saved
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefs
}).sort_values(by="Coefficient", key=lambda x: np.abs(x), ascending=False)
coef_df.to_csv(os.path.join(LOG_DIR, "lasso_coefficients.csv"), index=False, sep=';')

# =====  FEATURE FILTERS =====
X_train_sel = pd.DataFrame(X_train, columns=feature_names)[selected_features]
X_test_sel = pd.DataFrame(X_test, columns=feature_names)[selected_features]

# ===== 4. RANDOM FOREST =====
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight='balanced'
)
rf.fit(X_train_sel, y_train)

y_prob = rf.predict_proba(X_test_sel)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

# ===== METRIX =====
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Log Loss": log_loss(y_test, y_prob),
    "Cohen Kappa": cohen_kappa_score(y_test, y_pred)
}
log("\n=== RANDOM FOREST ON SELECTED FEATURE BY LASSO ===")
for k, v in metrics.items():
    log(f"{k}: {v:.4f}")

# ===== REPORT =====
report = classification_report(y_test, y_pred)
log("\nClassification Report:\n" + report)
with open(os.path.join(LOG_DIR, "classification_report_lasso_rf.txt"), "w") as f:
    f.write(report)

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
log(f"\nConfusion Matrix:\nTN={tn} FP={fp} FN={fn} TP={tp}")

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix (RF su feature Lasso)")
plt.colorbar()
plt.xticks([0, 1], ['No Default', 'Default'])
plt.yticks([0, 1], ['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('True')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(LOG_DIR, "confusion_matrix_rf_lasso.png"))
plt.close()

# ===== ROC CURVE =====
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC (RF su feature Lasso)')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(LOG_DIR, "roc_curve_rf_lasso.png"))
plt.close()

# ===== FEATURE IMPORTANCE RF =====
importances = rf.feature_importances_
feat_imp_rf = pd.DataFrame({
    "Feature": selected_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

feat_imp_rf.to_csv(os.path.join(LOG_DIR, "rf_feature_importance_lasso.csv"), index=False, sep=';')

plt.figure(figsize=(10, 6))
top_feats = feat_imp_rf.head(25)
plt.barh(top_feats["Feature"][::-1], top_feats["Importance"][::-1])
plt.xlabel("Importanza")
plt.title("Top 25 Feature Importances (RF su feature Lasso)")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(LOG_DIR, "rf_feature_importance_lasso.png"))
plt.close()

# Script_LASSO_multirun

import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, cohen_kappa_score, classification_report, confusion_matrix,
    roc_curve
)
from sklearn.preprocessing import StandardScaler

# ===== CONFIGURATIONS =====
DATASET_PATH = "Dataset3.csv"
TARGET_COLUMN = "default.payment.next.month"
LOG_DIR = "logs"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAVE_LOG = True
THRESHOLD = 0.4
C_VALUES = np.logspace(-2, 1, 10)  # 10 valori da 0.01 a 10

# ===== CREATE LOG FOLDER =====
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(LOG_DIR, f"log_lasso_rf_grid_{timestamp}.txt")


def log(msg):
    print(msg)
    if SAVE_LOG:
        with open(log_file, "a") as f:
            f.write(msg + "\n")


# ===== PREPROCESSING =====
df = pd.read_csv(DATASET_PATH, sep=';')
df = df.drop(columns=['ID'], errors='ignore')
df = pd.get_dummies(df, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True)

X = df.drop(columns=TARGET_COLUMN)
y = df[TARGET_COLUMN]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ===== RISULTS FOR EVERY C =====
results = []

for C in C_VALUES:
    log(f"\n=== C = {C:.4f} ===")

    lasso = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=5000, random_state=RANDOM_STATE)
    lasso.fit(X_train, y_train)

    coefs = lasso.coef_[0]
    feature_names = X.columns
    selected_features = feature_names[coefs != 0]

    log(f"Feature selezionate: {len(selected_features)} / {len(feature_names)}")

    if len(selected_features) == 0:
        log("Nessuna feature selezionata. Salto iterazione.")
        continue

    X_train_sel = pd.DataFrame(X_train, columns=feature_names)[selected_features]
    X_test_sel = pd.DataFrame(X_test, columns=feature_names)[selected_features]

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced')
    rf.fit(X_train_sel, y_train)

    y_prob = rf.predict_proba(X_test_sel)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    metrics = {
        "C": C,
        "n_features": len(selected_features),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "LogLoss": log_loss(y_test, y_prob),
        "Kappa": cohen_kappa_score(y_test, y_pred)
    }

    for k, v in metrics.items():
        if k != "C":
            log(f"{k}: {v:.4f}")
    results.append(metrics)

# ===== SAVE AND PLOTTING =====
results_df = pd.DataFrame(results)
print("\nResults Lasso+RF grid search:\n")
print(results_df)

# METRICS PLOT
metrics_to_plot = ["Accuracy", "Recall", "Precision", "F1", "AUC", "LogLoss", "Kappa"]
plt.figure(figsize=(12, 8))
for metric in metrics_to_plot:
    plt.plot(results_df["C"], results_df[metric], marker='o', label=metric)

plt.xscale("log")
plt.xlabel("Valore di C (log scale)")
plt.ylabel("Valore della metrica")
plt.title("Metriche per Random Forest su feature selezionate da Lasso")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(LOG_DIR, "lasso_rf_metrics_vs_C.png"))
ptplt.close()
