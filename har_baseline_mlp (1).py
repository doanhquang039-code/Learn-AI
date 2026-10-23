"""
Chu de 04 - Nhan dang hoat dong qua cam bien smartphone (UCI HAR)
Baseline: Logistic Regression | Cai tien: MLP nho
Chi so: Accuracy, Macro-F1, F1 tung lop, confusion matrix

Cach dung:
  1. Tai & giai nen dataset tu
     https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
     de co thu muc "UCI HAR Dataset" (chua train/, test/, activity_labels.txt, features.txt)
  2. pip install numpy pandas scikit-learn matplotlib joblib
  3. python har_baseline_mlp.py --data "UCI HAR Dataset"
"""
import argparse
import itertools
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------- 1. Doc du lieu
def load_split(root, split):
    X = np.loadtxt(os.path.join(root, split, f"X_{split}.txt"))
    y = np.loadtxt(os.path.join(root, split, f"y_{split}.txt")).astype(int) - 1  # 0..5
    s = np.loadtxt(os.path.join(root, split, f"subject_{split}.txt")).astype(int)
    return X, y, s


def main(root, out="outputs"):
    os.makedirs(out, exist_ok=True)
    labels = pd.read_csv(os.path.join(root, "activity_labels.txt"),
                         sep=r"\s+", header=None, names=["id", "name"])
    class_names = labels["name"].tolist()

    X_train, y_train, s_train = load_split(root, "train")
    X_test, y_test, s_test = load_split(root, "test")

    print("X_train:", X_train.shape, "| X_test:", X_test.shape)
    print("So nguoi train/test:", len(set(s_train)), "/", len(set(s_test)))
    print("Nguoi trung giua train va test:", set(s_train) & set(s_test) or "khong")
    print("\nPhan bo lop (train):")
    print(pd.Series(y_train).map(dict(enumerate(class_names))).value_counts())
    print("\nPhan bo lop (test):")
    print(pd.Series(y_test).map(dict(enumerate(class_names))).value_counts())

    # ------------------------------------------- 2. Validation tach theo nguoi (tu train)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, va_idx = next(gss.split(X_train, y_train, groups=s_train))
    assert not set(s_train[tr_idx]) & set(s_train[va_idx]), "Ro ri nguoi giua train/val!"
    Xtr, ytr, Xva, yva = X_train[tr_idx], y_train[tr_idx], X_train[va_idx], y_train[va_idx]

    # --------------------------------------------------------- 3. Baseline: Logistic Reg
    best_lr, best_lr_f1 = None, -1
    for C in [0.01, 0.1, 1, 10]:
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, max_iter=2000, random_state=SEED))
        m.fit(Xtr, ytr)
        f1 = f1_score(yva, m.predict(Xva), average="macro")
        print(f"[LR] C={C}: val macro-F1 = {f1:.4f}")
        if f1 > best_lr_f1:
            best_lr, best_lr_f1 = C, f1
    baseline = make_pipeline(StandardScaler(),
                             LogisticRegression(C=best_lr, max_iter=2000, random_state=SEED))
    baseline.fit(X_train, y_train)  # fit lai tren toan bo train

    # ------------------------------------------------------------- 4. MLP cai tien
    grid = list(itertools.product([(128,), (128, 64)], [1e-4, 1e-2], [1e-3]))
    best_cfg, best_mlp_f1 = None, -1
    for hidden, alpha, lr in grid:
        m = make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha,
                                        learning_rate_init=lr, max_iter=300,
                                        early_stopping=False, random_state=SEED))
        m.fit(Xtr, ytr)
        f1 = f1_score(yva, m.predict(Xva), average="macro")
        print(f"[MLP] hidden={hidden}, alpha={alpha}: val macro-F1 = {f1:.4f}")
        if f1 > best_mlp_f1:
            best_cfg, best_mlp_f1 = (hidden, alpha, lr), f1
    hidden, alpha, lr = best_cfg
    mlp = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha,
                                      learning_rate_init=lr, max_iter=300,
                                      random_state=SEED))
    mlp.fit(X_train, y_train)

    # ------------------------------------------------------ 5. Danh gia tren test
    results = {}
    for name, model in [("LogisticRegression (baseline)", baseline), ("MLP", mlp)]:
        pred = model.predict(X_test)
        results[name] = pred
        print("\n" + "=" * 60 + f"\n{name}")
        print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
        print(f"Macro-F1 : {f1_score(y_test, pred, average='macro'):.4f}")
        print(classification_report(y_test, pred, target_names=class_names, digits=4))
        cm = confusion_matrix(y_test, pred)
        print("Confusion matrix:\n", cm)
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"cm_{name.split()[0]}.png"), dpi=150)
        plt.close()

    # -------------------------------------- 6. Phan tich >= 3 du doan nham (MLP)
    proba = mlp.predict_proba(X_test)
    pred = results["MLP"]
    wrong = np.where(pred != y_test)[0]
    print(f"\nMLP du doan sai {len(wrong)}/{len(y_test)} mau. 10 mau sai dau tien:")
    rows = [{"idx": i, "subject": s_test[i], "that": class_names[y_test[i]],
             "du_doan": class_names[pred[i]], "do_tin_cay": round(proba[i].max(), 3)}
            for i in wrong[:10]]
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------------------------------------------------------- 7. Luu mo hinh
    joblib.dump({"model": mlp, "class_names": class_names}, os.path.join(out, "mlp_har.joblib"))
    joblib.dump({"model": baseline, "class_names": class_names}, os.path.join(out, "lr_har.joblib"))


# ------------------------------------------------------------------------- 8. Demo
def predict_one(feature_vector, model_path="outputs/mlp_har.joblib"):
    """Nhan vector 561 dac trung -> (nhan, xac suat)."""
    bundle = joblib.load(model_path)
    x = np.asarray(feature_vector, dtype=float).reshape(1, -1)
    if x.shape[1] != 561:
        raise ValueError(f"Can 561 dac trung, nhan duoc {x.shape[1]}")
    p = bundle["model"].predict_proba(x)[0]
    k = int(p.argmax())
    return bundle["class_names"][k], float(p[k])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="UCI HAR Dataset")
    ap.add_argument("--demo", action="store_true", help="demo 1 dong test sau khi train")
    args = ap.parse_args()
    main(args.data)
    if args.demo:
        X_test, y_test, _ = load_split(args.data, "test")
        print("Demo dong test #0:", predict_one(X_test[0]))
