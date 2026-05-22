 
#  Parkinson's Disease Detection — Full ML Pipeline
#  Dataset: UCI Parkinson's Voice Dataset (195 samples)
 
 

#  0. IMPORTS  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc,
                             accuracy_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import warnings, os
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"park": "#E63946", "healthy": "#457B9D"}

#  . LOAD DATA  
DATA_PATH = "parkinsons.data"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "parkinsons.data not found. Place it in the same folder as this script.")

df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

#   2. BASIC INFO  
print("\n── Dataset Overview ──")
print(f"Total samples  : {len(df)}")
print(f"Parkinson's (1): {(df['status']==1).sum()}")
print(f"Healthy (0)    : {(df['status']==0).sum()}")
print(f"Missing values : {df.isnull().sum().sum()}")
print(f"Features       : {df.shape[1] - 2}  (excluding name & status)")

#  3. EDA  
os.makedirs("output_charts", exist_ok=True)

# 3a. Class balance + correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

counts = df["status"].value_counts()
axes[0].bar(["Parkinson's (1)", "Healthy (0)"], counts.values,
            color=[COLORS["park"], COLORS["healthy"]], edgecolor="white", width=0.5)
axes[0].set_title("Class Distribution", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Number of Samples")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 1.5, str(v), ha="center", fontsize=12, fontweight="bold")

features = df.drop(columns=["name", "status"])
top_corr = features.corrwith(df["status"]).abs().sort_values(ascending=False).head(10)
corr_matrix = df[top_corr.index.tolist() + ["status"]].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=axes[1], linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[1].set_title("Top 10 Feature Correlations with Status", fontsize=13, fontweight="bold")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("output_charts/eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Saved: output_charts/eda_overview.png")

# 3b. Feature distributions — Jitter vs Shimmer by class
jitter_cols  = [c for c in df.columns if "Jitter"   in c][:3]
shimmer_cols = [c for c in df.columns if "Shimmer"  in c][:3]
plot_cols = jitter_cols + shimmer_cols

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.flatten(), plot_cols):
    for status, color, label in [(1, COLORS["park"], "Parkinson's"),
                                  (0, COLORS["healthy"], "Healthy")]:
        ax.hist(df[df["status"] == status][col], bins=20, alpha=0.65,
                color=color, label=label, edgecolor="white")
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
plt.suptitle("Jitter & Shimmer Distributions by Class",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("output_charts/feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Saved: output_charts/feature_distributions.png")

# 3c. Boxplots — top 6 correlated features
top6 = top_corr.index[:6].tolist()
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.flatten(), top6):
    df.boxplot(column=col, by="status", ax=ax,
               boxprops=dict(color="#264653"),
               medianprops=dict(color=COLORS["park"], linewidth=2))
    ax.set_title(col, fontsize=9, fontweight="bold")
    ax.set_xlabel("Status  (0=Healthy, 1=Parkinson's)")
    ax.set_xticklabels(["Healthy", "Parkinson's"])
plt.suptitle("Top 6 Features — Boxplots by Class",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("output_charts/boxplots.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Saved: output_charts/boxplots.png")

#  4. PREPARE DATA  
X = df.drop(columns=["name", "status"])
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n── Train/Test Split ──")
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

#   5. MODELS  
models = {
    "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, random_state=42),
    "SVM (RBF)":            SVC(kernel="rbf", probability=True, random_state=42),
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7),
}
NEEDS_SCALING = {"SVM (RBF)", "Logistic Regression", "K-Nearest Neighbours"}

# 6. TRAIN + CROSS-VALIDATE  
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n── Cross-Validation Results (5-Fold) ──")
print(f"{'Model':<25} {'CV Accuracy':>12}  {'±':>6}  {'Test Acc':>10}")
print("─" * 58)

for name, model in models.items():
    Xtr = X_train_sc if name in NEEDS_SCALING else X_train.values
    Xte = X_test_sc  if name in NEEDS_SCALING else X_test.values

    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy")
    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)

    results[name] = {
        "model": model, "y_pred": y_pred, "y_proba": y_proba,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        "test_acc": test_acc, "X_test": Xte
    }
    print(f"{name:<25} {cv_scores.mean():.4f}        {cv_scores.std():.4f}   {test_acc:.4f}")

#  7. CONFUSION MATRICES  
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Parkinson's"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=9, fontweight="bold")
plt.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("output_charts/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: output_charts/confusion_matrices.png")

#   8. ROC CURVES  
fig, ax = plt.subplots(figsize=(8, 6))
palette = ["#E63946", "#2A9D8F", "#E9C46A", "#264653", "#F4A261"]

for (name, res), color in zip(results.items(), palette):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig("output_charts/roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Saved: output_charts/roc_curves.png")

#  9. MODEL COMPARISON BAR CHART  
model_names = list(results.keys())
cv_means    = [results[m]["cv_mean"]  for m in model_names]
test_accs   = [results[m]["test_acc"] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width/2, cv_means,  width, label="CV Accuracy (5-fold)",
               color="#457B9D", edgecolor="white")
bars2 = ax.bar(x + width/2, test_accs, width, label="Test Accuracy",
               color="#E63946", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha="right")
ax.set_ylim(0.7, 1.05)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Model Comparison — CV vs Test Accuracy", fontsize=14, fontweight="bold")
ax.legend()

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("output_charts/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Saved: output_charts/model_comparison.png")

#   10. BEST MODEL — DETAILED REPORT 
best_name = max(results, key=lambda m: results[m]["cv_mean"])
best      = results[best_name]

print(f"\n🏆 Best Model: {best_name}")
print(f"   CV Accuracy  : {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
print(f"   Test Accuracy: {best['test_acc']:.4f}")
print("\n── Classification Report ──")
print(classification_report(y_test, best["y_pred"],
                             target_names=["Healthy", "Parkinson's"]))

# 11. FEATURE IMPORTANCE 
if hasattr(best["model"], "feature_importances_"):
    importances = pd.Series(best["model"].feature_importances_, index=X.columns)
    top15 = importances.sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ["#E63946"] * 5 + ["#457B9D"] * 10
    top15.sort_values().plot(kind="barh", ax=ax,
                             color=bar_colors[::-1], edgecolor="white")
    ax.set_title(f"Top 15 Feature Importances — {best_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("output_charts/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Saved: output_charts/feature_importance.png")

print("\n✅ Pipeline complete! All charts saved in output_charts/")


 
#  PART 2 — TELEMONITORING: Predicting Disease Severity
#  Dataset: parkinsons_updrs.data (5,875 voice recordings)
#  Task: Regression — predict motor_UPDRS & total_UPDRS
 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("\n" + "="*60)
print("  PART 2: TELEMONITORING — SEVERITY PREDICTION")
print("="*60)

#  T1. LOAD TELEMONITORING DATA  
TELE_PATH = "telemonitoring/parkinsons_updrs.data"

if not os.path.exists(TELE_PATH):
    print("⚠️  Telemonitoring data not found. Skipping Part 2.")
else:
    tdf = pd.read_csv(TELE_PATH)
    print(f"\n✅ Telemonitoring data loaded: {tdf.shape[0]} recordings, {tdf.shape[1]} columns")
    print(f"   Subjects : {tdf['subject#'].nunique()} patients")
    print(f"   Age range: {tdf['age'].min()}–{tdf['age'].max()} years")
    print(f"   Sex       : {(tdf['sex']==0).sum()} male, {(tdf['sex']==1).sum()} female")
    print(f"\n   motor_UPDRS — min: {tdf['motor_UPDRS'].min():.1f}, "
          f"max: {tdf['motor_UPDRS'].max():.1f}, "
          f"mean: {tdf['motor_UPDRS'].mean():.1f}")
    print(f"   total_UPDRS — min: {tdf['total_UPDRS'].min():.1f}, "
          f"max: {tdf['total_UPDRS'].max():.1f}, "
          f"mean: {tdf['total_UPDRS'].mean():.1f}")

    #  T2. EDA: UPDRS OVER TIME  
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of severity scores
    axes[0].hist(tdf["motor_UPDRS"], bins=30, alpha=0.7,
                 color="#E63946", edgecolor="white", label="Motor UPDRS")
    axes[0].hist(tdf["total_UPDRS"], bins=30, alpha=0.6,
                 color="#457B9D", edgecolor="white", label="Total UPDRS")
    axes[0].set_title("Distribution of UPDRS Severity Scores", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("UPDRS Score")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Severity progression over time (mean per test_time day)
    time_grp = tdf.groupby(tdf["test_time"].astype(int))[["motor_UPDRS", "total_UPDRS"]].mean()
    axes[1].plot(time_grp.index, time_grp["motor_UPDRS"],
                 color="#E63946", lw=2, label="Motor UPDRS")
    axes[1].plot(time_grp.index, time_grp["total_UPDRS"],
                 color="#457B9D", lw=2, label="Total UPDRS")
    axes[1].set_title("Average UPDRS Scores Over Trial Time", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Days Since Recruitment")
    axes[1].set_ylabel("Mean UPDRS Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("output_charts/tele_eda.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n Saved: output_charts/tele_eda.png")

    #   T3. PREPARE DATA 
    voice_features = ["Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
                      "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
                      "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"]

    X_t = tdf[voice_features]
    y_motor = tdf["motor_UPDRS"]
    y_total = tdf["total_UPDRS"]

    scaler_t = StandardScaler()
    X_t_sc = scaler_t.fit_transform(X_t)

    Xtr_t, Xte_t, ym_tr, ym_te, yt_tr, yt_te = train_test_split(
        X_t_sc, y_motor, y_total, test_size=0.2, random_state=42)

    print(f"\n── Regression Train/Test Split ──")
    print(f"Train: {Xtr_t.shape[0]} | Test: {Xte_t.shape[0]}")

    #  T4. REGRESSION MODELS 
    reg_models = {
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42),
        "Ridge Regression":  Ridge(alpha=1.0),
    }

    print("\n── Regression Results ──")
    print(f"{'Model':<22} {'Target':<14} {'R²':>6}  {'MAE':>7}  {'RMSE':>7}")
    print("─" * 60)

    reg_results = {}
    for name, model in reg_models.items():
        model.fit(Xtr_t, ym_tr)
        ym_pred = model.predict(Xte_t)
        r2_m   = r2_score(ym_te, ym_pred)
        mae_m  = mean_absolute_error(ym_te, ym_pred)
        rmse_m = np.sqrt(mean_squared_error(ym_te, ym_pred))

        model2 = type(model)(**model.get_params())
        model2.fit(Xtr_t, yt_tr)
        yt_pred = model2.predict(Xte_t)
        r2_t   = r2_score(yt_te, yt_pred)
        mae_t  = mean_absolute_error(yt_te, yt_pred)
        rmse_t = np.sqrt(mean_squared_error(yt_te, yt_pred))

        reg_results[name] = {
            "motor": {"model": model, "pred": ym_pred, "r2": r2_m, "mae": mae_m, "rmse": rmse_m},
            "total": {"model": model2, "pred": yt_pred, "r2": r2_t, "mae": mae_t, "rmse": rmse_t},
        }
        print(f"{name:<22} {'motor_UPDRS':<14} {r2_m:>6.3f}  {mae_m:>7.2f}  {rmse_m:>7.2f}")
        print(f"{'':<22} {'total_UPDRS':<14} {r2_t:>6.3f}  {mae_t:>7.2f}  {rmse_t:>7.2f}")
        print("─" * 60)

    #   T5. PREDICTED vs ACTUAL PLOTS  
    best_reg = max(reg_results, key=lambda m: reg_results[m]["motor"]["r2"])
    best_r   = reg_results[best_reg]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, label, color in [
        (axes[0], "motor", "Motor UPDRS", "#E63946"),
        (axes[1], "total", "Total UPDRS", "#457B9D"),
    ]:
        actual = ym_te if key == "motor" else yt_te
        pred   = best_r[key]["pred"]
        r2     = best_r[key]["r2"]
        ax.scatter(actual, pred, alpha=0.4, color=color, edgecolors="white", s=30)
        lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        ax.plot(lims, lims, "k--", lw=1.5, label="Perfect prediction")
        ax.set_xlabel(f"Actual {label}", fontsize=11)
        ax.set_ylabel(f"Predicted {label}", fontsize=11)
        ax.set_title(f"{best_reg} — {label}\nR² = {r2:.3f}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    plt.suptitle("Predicted vs Actual UPDRS Severity Scores",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("output_charts/tele_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n📊 Saved: output_charts/tele_predictions.png")

    #   T6. FEATURE IMPORTANCE FOR SEVERITY 
    best_motor_model = best_r["motor"]["model"]
    if hasattr(best_motor_model, "feature_importances_"):
        imp = pd.Series(best_motor_model.feature_importances_, index=voice_features)
        top_imp = imp.sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(9, 5))
        top_imp.sort_values().plot(kind="barh", ax=ax,
                                   color="#E63946", edgecolor="white")
        ax.set_title(f"Top 10 Voice Features for Motor UPDRS Prediction\n({best_reg})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig("output_charts/tele_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: output_charts/tele_feature_importance.png")

    print("\n Telemonitoring analysis complete!")
    print("   Charts saved in output_charts/")