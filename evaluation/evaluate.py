import sys
sys.stdout.reconfigure(line_buffering=True)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import numpy as np
import json
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)

print("Loading model and data...")
model       = tf.keras.models.load_model("saved_models/fl_model.h5")
X_test      = np.load("saved_models/X_test.npy")
y_test      = np.load("saved_models/y_test.npy")
class_names = np.load("saved_models/class_names.npy", allow_pickle=True)
class_names = [str(c) for c in class_names]
num_classes = len(class_names)

print("Making predictions...")
y_pred_proba = model.predict(X_test, batch_size=256, verbose=1)
y_pred       = np.argmax(y_pred_proba, axis=1)

# ── Metrics ───────────────────────────────────────────────────────────
acc       = accuracy_score(y_test, y_pred)
macro_f1  = f1_score(y_test, y_pred, average="macro")
macro_p   = precision_score(y_test, y_pred, average="macro")
macro_r   = recall_score(y_test, y_pred, average="macro")

per_f1  = f1_score(y_test, y_pred, average=None)
per_p   = precision_score(y_test, y_pred, average=None)
per_r   = recall_score(y_test, y_pred, average=None)
cm      = confusion_matrix(y_test, y_pred)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Accuracy  : {acc:.4f}")
print(f"Macro F1  : {macro_f1:.4f}")
print(f"Precision : {macro_p:.4f}")
print(f"Recall    : {macro_r:.4f}")
print("\nPer-class:")
for i, cls in enumerate(class_names):
    print(f"  {cls:20s} P:{per_p[i]:.3f} R:{per_r[i]:.3f} F1:{per_f1[i]:.3f}")

# ── Save confusion matrix image ───────────────────────────────────────
os.makedirs("evaluation", exist_ok=True)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("evaluation/confusion_matrix.png", dpi=150)
plt.close()
print("\n✅ Saved evaluation/confusion_matrix.png")

# ── Save JSON for dashboard ───────────────────────────────────────────
results = {
    "accuracy":  round(acc, 4),
    "macro_f1":  round(macro_f1, 4),
    "precision": round(macro_p, 4),
    "recall":    round(macro_r, 4),
    "classes":   class_names,
    "per_f1":    [round(v, 4) for v in per_f1],
    "per_p":     [round(v, 4) for v in per_p],
    "per_r":     [round(v, 4) for v in per_r],
    "cm":        cm.tolist(),
    "total_samples": len(y_test),
}
with open("evaluation/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved evaluation/results.json")
print("\n🎉 EVALUATION COMPLETE!")