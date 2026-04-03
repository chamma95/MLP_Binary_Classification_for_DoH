import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD THE 3 CSVs
# ─────────────────────────────────────────────
train_df = pd.read_csv("Data/train.csv")
val_df   = pd.read_csv("Data/validation.csv")
test_df  = pd.read_csv("Data/test.csv")

print(f"Train:      {train_df.shape}")
print(f"Validation: {val_df.shape}")
print(f"Test:       {test_df.shape}")

# ─────────────────────────────────────────────
# 2. SEPARATE FEATURES AND LABEL
# ─────────────────────────────────────────────
LABEL_COL = "Label"

X_train = train_df.drop(columns=[LABEL_COL])
y_train = train_df[LABEL_COL]

X_val   = val_df.drop(columns=[LABEL_COL])
y_val   = val_df[LABEL_COL]

X_test  = test_df.drop(columns=[LABEL_COL])
y_test  = test_df[LABEL_COL]

# ─────────────────────────────────────────────
# 3. ENCODE LABELS  →  DoH = 1,  NonDoH = 0
# ─────────────────────────────────────────────
le = LabelEncoder()
le.fit(y_train)

y_train = le.transform(y_train)
y_val   = le.transform(y_val)
y_test  = le.transform(y_test)

print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ─────────────────────────────────────────────
# 4. FEATURE SCALING
#    Fit ONLY on train → apply to val and test
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

n_features = X_train.shape[1]
print(f"\nNumber of features: {n_features}")

# ─────────────────────────────────────────────
# 5. BUILD THE MLP MODEL
#    Input(29) → Dense(128) → Dense(64) → Dense(32) → Output(1)
# ─────────────────────────────────────────────
model = Sequential([
    Dense(128, activation="relu", input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

model.summary()

# ─────────────────────────────────────────────
# 6. CALLBACKS
# ─────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,           # stop if val_loss doesn't improve for 15 epochs
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "Results/best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# ─────────────────────────────────────────────
# 7. TRAIN
# ─────────────────────────────────────────────
print("\n─── Training Started ────────────────────────")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("─── Training Complete ───────────────────────\n")

# ─────────────────────────────────────────────
# 8. VISUALIZE TRAINING PROGRESS
# ─────────────────────────────────────────────
hist = history.history
epochs_ran = range(1, len(hist["loss"]) + 1)

fig = plt.figure(figsize=(16, 10))
fig.suptitle("MLP Training Progress — DoH Binary Classification", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Plot 1: Loss ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs_ran, hist["loss"],     label="Train Loss",      color="#2196F3", linewidth=2)
ax1.plot(epochs_ran, hist["val_loss"], label="Validation Loss", color="#F44336", linewidth=2, linestyle="--")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Binary Crossentropy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── Plot 2: Accuracy ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs_ran, hist["accuracy"],     label="Train Accuracy",      color="#2196F3", linewidth=2)
ax2.plot(epochs_ran, hist["val_accuracy"], label="Validation Accuracy", color="#F44336", linewidth=2, linestyle="--")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── Plot 3: AUC ──
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(epochs_ran, hist["auc"],     label="Train AUC",      color="#4CAF50", linewidth=2)
ax3.plot(epochs_ran, hist["val_auc"], label="Validation AUC", color="#FF9800", linewidth=2, linestyle="--")
ax3.set_title("AUC-ROC")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("AUC")
ax3.legend()
ax3.grid(True, alpha=0.3)

# ── Plot 4: Precision & Recall ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(epochs_ran, hist["precision"],     label="Train Precision",      color="#9C27B0", linewidth=2)
ax4.plot(epochs_ran, hist["val_precision"], label="Validation Precision", color="#9C27B0", linewidth=2, linestyle="--")
ax4.plot(epochs_ran, hist["recall"],        label="Train Recall",         color="#009688", linewidth=2)
ax4.plot(epochs_ran, hist["val_recall"],    label="Validation Recall",    color="#009688", linewidth=2, linestyle="--")
ax4.set_title("Precision & Recall")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Score")
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

plt.savefig("Results/training_progress.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ training_progress.png saved")

# ─────────────────────────────────────────────
# 9. FINAL EVALUATION ON TEST SET
# ─────────────────────────────────────────────
print("\n─── Test Set Evaluation ─────────────────────")

y_pred_prob = model.predict(X_test).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC Score: {auc:.4f}")

# ── Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Confusion Matrix — Test Set")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("Results/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ confusion_matrix.png saved")
