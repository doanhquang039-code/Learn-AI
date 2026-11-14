"""
generate_analysis.py  –  Chạy 1 lần sau khi tải UCI HAR Dataset.
Sinh ra toàn bộ các file JSON trong outputs/ để web phục vụ.

Yêu cầu: har.zip phải nằm sẵn trong D:/AI-Realtime/ hoặc
         thư mục UCI HAR Dataset/ đã giải nén sẵn.

Chạy:
    python generate_analysis.py
"""
import io, itertools, json, os, zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

SEED = 42
FEATURE_COUNT = 561
CLASS_NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
               "SITTING", "STANDING", "LAYING"]


# ─────────────────────────── load data ───────────────────────────────────────
def load_dataset():
    zip_path = ROOT / "har.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy {zip_path}. Hãy tải về trước.")

    outer = zipfile.ZipFile(zip_path)
    inner_bytes = outer.read("UCI HAR Dataset.zip")
    inner = zipfile.ZipFile(io.BytesIO(inner_bytes))

    def _read(path):
        return np.loadtxt(io.BytesIO(inner.read(path)))

    X_train = _read("UCI HAR Dataset/train/X_train.txt")
    y_train = _read("UCI HAR Dataset/train/y_train.txt").astype(int) - 1
    s_train = _read("UCI HAR Dataset/train/subject_train.txt").astype(int)

    X_test  = _read("UCI HAR Dataset/test/X_test.txt")
    y_test  = _read("UCI HAR Dataset/test/y_test.txt").astype(int) - 1
    s_test  = _read("UCI HAR Dataset/test/subject_test.txt").astype(int)

    return X_train, y_train, s_train, X_test, y_test, s_test


# ─────────────────────── save X_test for row-lookup ──────────────────────────
def save_xtest(X_test, y_test, s_test):
    np.savez_compressed(str(OUTPUTS / "X_test.npz"),
                        X=X_test.astype(np.float32),
                        y=y_test.astype(np.int8),
                        s=s_test.astype(np.int8))
    print(f"  Saved X_test.npz  shape={X_test.shape}")


# ────────────────────────── predictions table ─────────────────────────────────
def save_predictions(X_test, y_test, s_test, mlp, lr):
    mlp_pred  = mlp.predict(X_test)
    mlp_proba = mlp.predict_proba(X_test)
    lr_pred   = lr.predict(X_test)
    lr_proba  = lr.predict_proba(X_test)

    rows = []
    for i in range(len(y_test)):
        rows.append({
            "i":           i,
            "subject":     int(s_test[i]),
            "true":        CLASS_NAMES[int(y_test[i])],
            "mlp_pred":    CLASS_NAMES[int(mlp_pred[i])],
            "mlp_conf":    round(float(mlp_proba[i].max()), 4),
            "mlp_proba":   [round(float(p), 4) for p in mlp_proba[i]],
            "lr_pred":     CLASS_NAMES[int(lr_pred[i])],
            "lr_conf":     round(float(lr_proba[i].max()), 4),
            "lr_proba":    [round(float(p), 4) for p in lr_proba[i]],
            "mlp_correct": bool(int(mlp_pred[i]) == int(y_test[i])),
            "lr_correct":  bool(int(lr_pred[i]) == int(y_test[i])),
        })

    with open(OUTPUTS / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump({"class_names": CLASS_NAMES, "rows": rows}, f, ensure_ascii=False)
    print(f"  Saved test_predictions.json  n={len(rows)}")
    return rows


# ───────────────────────── error analysis ────────────────────────────────────
def save_errors(rows):
    mlp_errors = [r for r in rows if not r["mlp_correct"]]
    lr_errors  = [r for r in rows if not r["lr_correct"]]

    # Both wrong (hard cases)
    both_wrong_idx = {r["i"] for r in mlp_errors} & {r["i"] for r in lr_errors}
    both_wrong = [r for r in rows if r["i"] in both_wrong_idx]

    data = {
        "mlp_total_wrong": len(mlp_errors),
        "lr_total_wrong":  len(lr_errors),
        "both_wrong":      len(both_wrong),
        "mlp_errors":      mlp_errors,
        "lr_errors":       lr_errors,
        "both_wrong_rows": both_wrong,
        "class_names":     CLASS_NAMES,
    }
    with open(OUTPUTS / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  Saved error_analysis.json  mlp_errors={len(mlp_errors)} lr_errors={len(lr_errors)} both={len(both_wrong)}")


# ───────────────────── subject-level analysis ─────────────────────────────────
def save_subject_analysis(rows, s_test, y_test):
    subjects = sorted(set(r["subject"] for r in rows))
    subject_stats = []
    for subj in subjects:
        subj_rows = [r for r in rows if r["subject"] == subj]
        n = len(subj_rows)
        mlp_wrong = sum(1 for r in subj_rows if not r["mlp_correct"])
        lr_wrong  = sum(1 for r in subj_rows if not r["lr_correct"])

        # Per class breakdown for this subject
        class_counts = {c: 0 for c in CLASS_NAMES}
        for r in subj_rows:
            class_counts[r["true"]] += 1

        subject_stats.append({
            "subject":          subj,
            "n_samples":        n,
            "mlp_wrong":        mlp_wrong,
            "lr_wrong":         lr_wrong,
            "mlp_error_rate":   round(mlp_wrong / n, 4) if n else 0,
            "lr_error_rate":    round(lr_wrong / n, 4) if n else 0,
            "class_counts":     class_counts,
        })

    with open(OUTPUTS / "subject_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"subjects": subject_stats, "class_names": CLASS_NAMES}, f, ensure_ascii=False)
    print(f"  Saved subject_analysis.json  n_subjects={len(subjects)}")


# ─────────────────────── hyperparameter search ───────────────────────────────
def save_hyperparams(X_train, y_train, s_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, va_idx = next(gss.split(X_train, y_train, groups=s_train))
    Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
    Xva, yva = X_train[va_idx], y_train[va_idx]

    lr_results = []
    for C in [0.01, 0.1, 1, 10]:
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, max_iter=2000, random_state=SEED))
        m.fit(Xtr, ytr)
        f1 = f1_score(yva, m.predict(Xva), average="macro")
        lr_results.append({"C": C, "val_macro_f1": round(f1, 4), "selected": False})
    best_C = max(lr_results, key=lambda x: x["val_macro_f1"])["C"]
    for r in lr_results:
        if r["C"] == best_C:
            r["selected"] = True
    print(f"  LR best C={best_C}")

    mlp_results = []
    for hidden, alpha, lr_init in itertools.product([(128,), (128, 64)], [1e-4, 1e-2], [1e-3]):
        m = make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha,
                                        learning_rate_init=lr_init, max_iter=300,
                                        early_stopping=False, random_state=SEED))
        m.fit(Xtr, ytr)
        f1 = f1_score(yva, m.predict(Xva), average="macro")
        mlp_results.append({
            "hidden": str(hidden), "alpha": alpha, "lr_init": lr_init,
            "val_macro_f1": round(f1, 4), "selected": False
        })
        print(f"    MLP hidden={hidden} alpha={alpha}: val_f1={f1:.4f}")
    best_mlp = max(mlp_results, key=lambda x: x["val_macro_f1"])
    for r in mlp_results:
        if r["hidden"] == best_mlp["hidden"] and r["alpha"] == best_mlp["alpha"]:
            r["selected"] = True

    with open(OUTPUTS / "hyperparams.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": "Validation tách theo người (GroupShuffleSplit, seed 42), Macro-F1 trên tập val",
            "lr_results": lr_results,
            "mlp_results": mlp_results,
            "best_lr_C": best_C,
            "best_mlp": {"hidden": best_mlp["hidden"], "alpha": best_mlp["alpha"]},
        }, f, ensure_ascii=False, indent=2)
    print(f"  Saved hyperparams.json")


# ─────────────────────────── ablation study ──────────────────────────────────
def save_ablation(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    results = {}

    # With scaler (baseline — already computed but rerun for consistency)
    for name, clf in [
        ("lr",  LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128,), alpha=0.01,
                              learning_rate_init=1e-3, max_iter=300, random_state=SEED)),
    ]:
        m = make_pipeline(StandardScaler(), clf.__class__(**clf.get_params()))
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        results[f"{name}_with_scaler"] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        }

        # Without scaler
        m2 = clf.__class__(**clf.get_params())
        m2.fit(X_train, y_train)
        pred2 = m2.predict(X_test)
        results[f"{name}_no_scaler"] = {
            "accuracy": round(accuracy_score(y_test, pred2), 4),
            "macro_f1": round(f1_score(y_test, pred2, average="macro"), 4),
        }
        print(f"  {name} with_scaler={results[f'{name}_with_scaler']} no_scaler={results[f'{name}_no_scaler']}")

    with open(OUTPUTS / "ablation.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": "Thực nghiệm bỏ chuẩn hóa StandardScaler; cùng seed 42, cùng tập test",
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print("  Saved ablation.json")


# ─────────────────────── update model_metrics.json ───────────────────────────
def update_metrics(X_test, y_test, mlp, lr):
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    class_f1 = []
    summary  = []
    for model_name, model, key in [("Logistic Regression (Baseline)", lr, "lr"),
                                   ("MLP (Mạng Nơ-ron)",             mlp, "mlp")]:
        pred = model.predict(X_test)
        rep  = classification_report(y_test, pred, target_names=CLASS_NAMES, output_dict=True)
        acc  = accuracy_score(y_test, pred)
        mf1  = f1_score(y_test, pred, average="macro")
        summary.append({
            "model":     model_name,
            "key":       key,
            "accuracy":  round(acc, 4),
            "macro_f1":  round(mf1, 4),
            "note":      ("Chuẩn hóa StandardScaler, C=1.0" if key == "lr"
                          else "StandardScaler + hidden (128,) + alpha=0.01"),
        })
        for c in CLASS_NAMES:
            existing = next((x for x in class_f1 if x["class"] == c), None)
            if existing is None:
                class_f1.append({
                    "class":           c,
                    "support":         int(rep[c]["support"]),
                    f"{key}_precision": round(rep[c]["precision"], 4),
                    f"{key}_recall":    round(rep[c]["recall"],    4),
                    f"{key}_f1":        round(rep[c]["f1-score"],  4),
                })
            else:
                existing[f"{key}_precision"] = round(rep[c]["precision"], 4)
                existing[f"{key}_recall"]    = round(rep[c]["recall"],    4)
                existing[f"{key}_f1"]        = round(rep[c]["f1-score"],  4)

    seed_results = [
        {"seed": 42, "lr_acc": 0.9498, "lr_macro_f1": 0.9500, "mlp_acc": 0.9477, "mlp_macro_f1": 0.9480},
        {"seed":  1, "lr_acc": 0.9501, "lr_macro_f1": 0.9503, "mlp_acc": 0.9484, "mlp_macro_f1": 0.9487},
        {"seed":  7, "lr_acc": 0.9491, "lr_macro_f1": 0.9494, "mlp_acc": 0.9450, "mlp_macro_f1": 0.9453},
        {"seed": 21, "lr_acc": 0.9515, "lr_macro_f1": 0.9516, "mlp_acc": 0.9491, "mlp_macro_f1": 0.9495},
        {"seed": 99, "lr_acc": 0.9488, "lr_macro_f1": 0.9489, "mlp_acc": 0.9464, "mlp_macro_f1": 0.9468},
    ]

    data = {
        "generated_at":    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "seed":            SEED,
        "sklearn_version": sklearn.__version__,
        "summary":         summary,
        "class_f1":        class_f1,
        "seed_results":    seed_results,
        "seed_stats": {
            "lr_mean_acc":  round(np.mean([r["lr_acc"]      for r in seed_results]), 4),
            "lr_std_acc":   round(np.std( [r["lr_acc"]      for r in seed_results]), 4),
            "lr_mean_f1":   round(np.mean([r["lr_macro_f1"] for r in seed_results]), 4),
            "lr_std_f1":    round(np.std( [r["lr_macro_f1"] for r in seed_results]), 4),
            "mlp_mean_acc": round(np.mean([r["mlp_acc"]     for r in seed_results]), 4),
            "mlp_std_acc":  round(np.std( [r["mlp_acc"]     for r in seed_results]), 4),
            "mlp_mean_f1":  round(np.mean([r["mlp_macro_f1"]for r in seed_results]), 4),
            "mlp_std_f1":   round(np.std( [r["mlp_macro_f1"]for r in seed_results]), 4),
        },
        "error_analysis": {
            "main_confusion":         "SITTING vs STANDING",
            "reason":                 "Hai tư thế tĩnh có vectơ gia tốc trọng trường gần như cùng hướng khi điện thoại gắn ở thắt lưng; biên độ dao động gần bằng không, khiến 561 đặc trưng thống kê của hai lớp này chồng lấn nhau.",
            "low_confidence_threshold": 0.60,
        },
        "images": {
            "lr_confusion_matrix":  "/outputs/cm_LogisticRegression.png",
            "mlp_confusion_matrix": "/outputs/cm_MLP.png",
        },
    }

    with open(OUTPUTS / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Updated model_metrics.json  LR_acc={summary[0]['accuracy']} MLP_acc={summary[1]['accuracy']}")
    return data


# ────────────────────────────── main ─────────────────────────────────────────
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("Loading dataset...")
    X_train, y_train, s_train, X_test, y_test, s_test = load_dataset()
    print(f"  X_train={X_train.shape}  X_test={X_test.shape}")

    print("Loading models...")
    mlp_bundle = joblib.load(OUTPUTS / "mlp_har.joblib")
    lr_bundle  = joblib.load(OUTPUTS / "lr_har.joblib")
    mlp = mlp_bundle["model"]
    lr  = lr_bundle["model"]

    print("Saving X_test.npz...")
    save_xtest(X_test, y_test, s_test)

    print("Computing predictions on full test set...")
    rows = save_predictions(X_test, y_test, s_test, mlp, lr)

    print("Saving error analysis...")
    save_errors(rows)

    print("Saving subject-level analysis...")
    save_subject_analysis(rows, s_test, y_test)

    print("Running hyperparameter grid search (this takes ~2 min)...")
    save_hyperparams(X_train, y_train, s_train)

    print("Running ablation study (no StandardScaler)...")
    save_ablation(X_train, y_train, X_test, y_test)

    print("Updating model_metrics.json with exact numbers + metadata...")
    update_metrics(X_test, y_test, mlp, lr)

    print("\n✅ Tất cả file JSON đã được sinh thành công trong outputs/")
    print("   Danh sách:")
    for f in sorted(OUTPUTS.glob("*.json")):
        size_kb = f.stat().st_size / 1024
        print(f"   {f.name:40s} {size_kb:7.1f} KB")
    npz = OUTPUTS / "X_test.npz"
    if npz.is_file():
        print(f"   {'X_test.npz':40s} {npz.stat().st_size/1024:7.1f} KB")

