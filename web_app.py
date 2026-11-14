import csv
import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

APP_DIR    = Path(__file__).resolve().parent
OUTPUTS    = APP_DIR / "outputs"
SAMPLE_DIR = APP_DIR / "sample_data"
INDEX_PATH = APP_DIR / "index.html"

MLP_PATH     = OUTPUTS / "mlp_har.joblib"
LR_PATH      = OUTPUTS / "lr_har.joblib"
METRICS_PATH = OUTPUTS / "model_metrics.json"
XTEST_PATH   = OUTPUTS / "X_test.npz"

FEATURE_COUNT           = 561
MAX_UPLOAD_BYTES        = 10 * 1024 * 1024
MAX_ROWS                = 10_000
LOW_CONF_THRESHOLD      = 0.60
CLASS_NAMES             = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
                            "SITTING", "STANDING", "LAYING"]

app = FastAPI(title="UCI HAR – AI Realtime Demo")


# ═══════════════════════════ helpers ════════════════════════════════════════

def _is_number(value: str) -> bool:
    try:
        float(value.strip())
        return True
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=1)
def get_model_bundles() -> Dict[str, Any]:
    bundles: Dict[str, Any] = {}
    if MLP_PATH.is_file():
        bundles["mlp"] = joblib.load(MLP_PATH)
    if LR_PATH.is_file():
        bundles["lr"] = joblib.load(LR_PATH)
    if not bundles:
        raise HTTPException(503, detail="Không tìm thấy mô hình trong outputs/")
    return bundles


def get_model_bundle() -> Dict[str, Any]:
    """Compat helper – trả về MLP bundle."""
    b = get_model_bundles()
    return b.get("mlp") or next(iter(b.values()))


@lru_cache(maxsize=1)
def load_xtest() -> Optional[Dict[str, Any]]:
    if not XTEST_PATH.is_file():
        return None
    data = np.load(str(XTEST_PATH))
    return {"X": data["X"], "y": data["y"], "s": data["s"]}


@lru_cache(maxsize=1)
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predict_bundle(bundle: Dict[str, Any], features: np.ndarray) -> List[Dict[str, Any]]:
    model       = bundle["model"]
    class_names = bundle["class_names"]
    preds       = model.predict(features)
    probas      = model.predict_proba(features)
    classes     = model.classes_

    results = []
    for i, pred in enumerate(preds):
        prob_dict = {
            class_names[int(c)]: round(float(probas[i, j]), 6)
            for j, c in enumerate(classes)
        }
        conf = round(float(probas[i].max()), 6)
        results.append({
            "row":            i + 1,
            "prediction":     class_names[int(pred)],
            "confidence":     conf,
            "low_confidence": conf < LOW_CONF_THRESHOLD,
            "probabilities":  prob_dict,
        })
    return results


def calc_metrics(true_labels: List[str], pred_labels: List[str],
                 class_names: List[str]) -> Dict[str, Any]:
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    total   = len(true_labels)
    accuracy = round(correct / total, 4) if total else 0.0

    f1s = []
    for c in class_names:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == c and p == c)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != c and p == c)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    macro_f1 = round(float(np.mean(f1s)), 4)

    cm = []
    for tc in class_names:
        cm.append([sum(1 for t, p in zip(true_labels, pred_labels) if t == tc and p == pc)
                   for pc in class_names])

    return {"accuracy": accuracy, "macro_f1": macro_f1,
            "correct": correct, "total": total,
            "class_names": class_names, "confusion_matrix": cm}


def _normalize_label(raw: str, class_names: List[str]) -> str:
    s = raw.strip()
    if s.isdigit():
        idx = int(s)
        return class_names[idx - 1] if 1 <= idx <= len(class_names) \
               else (class_names[idx] if idx < len(class_names) else s)
    return next((c for c in class_names if c.lower() == s.lower()), s)


# ═══════════════════════════ parse CSV / text ═════════════════════════════════

def _parse_row(row: List[str], row_number: int) -> List[float]:
    if len(row) != FEATURE_COUNT:
        raise ValueError(f"Dòng {row_number} có {len(row)} cột; cần đúng {FEATURE_COUNT}.")
    vals = []
    for col, v in enumerate(row, 1):
        s = v.strip()
        if not s:
            raise ValueError(f"Dữ liệu trống ở dòng {row_number}, cột {col}.")
        try:
            n = float(s)
        except ValueError:
            raise ValueError(f"Dữ liệu không phải số ở dòng {row_number}, cột {col} ('{s}').")
        if not np.isfinite(n):
            raise ValueError(f"NaN/Inf ở dòng {row_number}, cột {col}.")
        vals.append(n)
    return vals


def parse_feature_text(text: str) -> np.ndarray:
    text = text.strip()
    if not text:
        raise ValueError("Vui lòng nhập vector đặc trưng 561 số.")
    tokens = [t.strip() for t in (text.split(",") if "," in text else text.split())]
    if tokens and tokens[-1] == "":
        tokens.pop()
    if len(tokens) != FEATURE_COUNT:
        raise ValueError(f"Vector có {len(tokens)} giá trị; yêu cầu đúng {FEATURE_COUNT} đặc trưng.")
    vals = []
    for idx, t in enumerate(tokens, 1):
        if not t:
            raise ValueError(f"Dữ liệu trống / thiếu giá trị ở vị trí đặc trưng thứ {idx}.")
        try:
            n = float(t)
        except ValueError:
            raise ValueError(f"Dữ liệu không phải số ở vị trí đặc trưng thứ {idx} (giá trị: '{t}').")
        if not np.isfinite(n):
            raise ValueError(f"NaN/Inf ở vị trí đặc trưng thứ {idx}.")
        vals.append(n)
    return np.asarray([vals], dtype=np.float64)


def parse_feature_csv(contents: bytes):
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError("CSV phải mã hóa UTF-8.") from e

    try:
        raw = [[c.strip() for c in r]
               for r in csv.reader(io.StringIO(text))
               if any(c.strip() for c in r)]
    except csv.Error as e:
        raise ValueError("Không đọc được cấu trúc CSV.") from e

    if not raw:
        raise ValueError("CSV không có dữ liệu.")

    # Detect header
    first = raw[0]
    has_header = False
    if len(first) >= 2:
        f0_text = not _is_number(first[0])
        f1_text = not _is_number(first[1])
        rest_num = sum(1 for v in first[1:] if _is_number(v))
        if f0_text and f1_text:
            has_header = True              # e.g. "true_label","feature_0",...
        elif f0_text and rest_num >= FEATURE_COUNT:
            has_header = False             # data row with text label at col 0
        else:
            has_header = sum(1 for v in first if _is_number(v)) < len(first) // 2
    else:
        has_header = sum(1 for v in first if _is_number(v)) < len(first) // 2

    data_rows = raw[1:] if has_header else raw
    if not data_rows:
        raise ValueError("CSV chỉ có tiêu đề, chưa có dòng dữ liệu.")
    if len(data_rows) > MAX_ROWS:
        raise ValueError(f"Tối đa {MAX_ROWS:,} dòng mỗi lần tải lên.")

    row_len = len(data_rows[0])
    labels: Optional[List[str]] = None

    if row_len == FEATURE_COUNT + 1:
        f0_num = _is_number(data_rows[0][0])
        fl_num = _is_number(data_rows[0][-1])
        if not f0_num:
            labels    = [r[0] for r in data_rows]
            data_rows = [r[1:] for r in data_rows]
        elif not fl_num:
            labels    = [r[-1] for r in data_rows]
            data_rows = [r[:-1] for r in data_rows]
        else:
            labels    = [r[-1] for r in data_rows]
            data_rows = [r[:-1] for r in data_rows]
    elif row_len != FEATURE_COUNT:
        raise ValueError(
            f"Mỗi dòng cần {FEATURE_COUNT} đặc trưng (hoặc {FEATURE_COUNT+1} nếu kèm nhãn); "
            f"dòng đầu có {row_len} cột.")

    features = [_parse_row(r, i) for i, r in enumerate(data_rows, 1)]
    return np.asarray(features, dtype=np.float64), labels


# ═══════════════════════════════ routes ══════════════════════════════════════

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX_PATH)


@app.get("/outputs/{filename}", include_in_schema=False)
def serve_output(filename: str):
    p = OUTPUTS / filename
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/health")
def health():
    b = get_model_bundles()
    return {
        "status": "ok",
        "available_models": list(b.keys()),
        "classes": CLASS_NAMES,
        "low_confidence_threshold": LOW_CONF_THRESHOLD,
        "xtest_available": XTEST_PATH.is_file(),
    }


@app.get("/api/model-results")
def model_results():
    data = load_json(METRICS_PATH)
    if not data:
        raise HTTPException(404, "Chưa có outputs/model_metrics.json")
    data.setdefault("images", {
        "lr_confusion_matrix":  "/outputs/cm_LogisticRegression.png",
        "mlp_confusion_matrix": "/outputs/cm_MLP.png",
    })
    return data


@app.get("/api/samples")
def get_samples():
    for p in [SAMPLE_DIR / "real_12_rows.csv", APP_DIR / "real_12_rows.csv"]:
        if p.is_file():
            break
    else:
        raise HTTPException(404, "Không tìm thấy real_12_rows.csv")

    with open(p, encoding="utf-8") as f:
        rdr = csv.reader(f)
        next(rdr)  # skip header
        samples = []
        for idx, row in enumerate(rdr, 1):
            lbl   = row[0]
            feats = [float(x) for x in row[1:]]
            samples.append({
                "id": idx, "true_label": lbl,
                "label": f"Mẫu #{idx:02d}  ({lbl})",
                "preview": [round(x, 4) for x in feats[:5]],
                "features": feats,
            })
    return {"samples": samples}


@app.get("/api/sequence")
def get_sequence():
    p = SAMPLE_DIR / "real_20_sequence.json"
    if not p.is_file():
        raise HTTPException(404, "Không tìm thấy real_20_sequence.json")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    bundles  = get_model_bundles()
    features = np.asarray([item["features"] for item in data], dtype=np.float64)
    mlp_res  = predict_bundle(bundles["mlp"], features) if "mlp" in bundles else []
    lr_res   = predict_bundle(bundles["lr"],  features) if "lr"  in bundles else []
    steps = []
    for i, item in enumerate(data):
        steps.append({
            "step": item["step"], "true_label": item["true_label"],
            "mlp": mlp_res[i] if mlp_res else None,
            "lr":  lr_res[i]  if lr_res  else None,
        })
    return {
        "disclaimer": "⚠️ Đây là mô phỏng tuần tự từ 20 mẫu kiểm thử đã thu thập trước, không phải tín hiệu cảm biến thời gian thực.",
        "steps": steps,
    }


# ─────────────── test-row lookup (requires X_test.npz) ───────────────────────

@app.get("/api/test-row/{index}")
def test_row(index: int, model_type: str = "both"):
    if index < 0 or index > 2946:
        raise HTTPException(400, "Chỉ số phải nằm trong [0, 2946].")
    xtest = load_xtest()
    if xtest is None:
        raise HTTPException(503, "X_test.npz chưa được tạo. Hãy chạy generate_analysis.py trước.")
    bundles = get_model_bundles()
    cls     = CLASS_NAMES
    feat    = xtest["X"][index : index + 1].astype(np.float64)
    true_lbl = cls[int(xtest["y"][index])]
    subject  = int(xtest["s"][index])

    resp: Dict[str, Any] = {
        "index": index, "subject": subject, "true_label": true_lbl,
    }
    if model_type in ("mlp", "both") and "mlp" in bundles:
        resp["mlp"] = predict_bundle(bundles["mlp"], feat)[0]
    if model_type in ("lr", "both") and "lr" in bundles:
        resp["lr"] = predict_bundle(bundles["lr"], feat)[0]
    if "mlp" in resp and "lr" in resp:
        resp["agreement"] = resp["mlp"]["prediction"] == resp["lr"]["prediction"]
    return resp


# ─────────────── error analysis ──────────────────────────────────────────────

@app.get("/api/errors")
def get_errors(model: str = "mlp", page: int = 0, page_size: int = 50):
    data = load_json(OUTPUTS / "error_analysis.json")
    if not data:
        raise HTTPException(404, "Chưa có error_analysis.json. Hãy chạy generate_analysis.py.")
    key = "mlp_errors" if model == "mlp" else "lr_errors"
    errors = data.get(key, [])
    start  = page * page_size
    return {
        "model":      model,
        "total":      data.get(f"{model}_total_wrong", len(errors)),
        "page":       page,
        "page_size":  page_size,
        "errors":     errors[start : start + page_size],
        "class_names": data.get("class_names", CLASS_NAMES),
    }


# ─────────────── subject analysis ────────────────────────────────────────────

@app.get("/api/subject-analysis")
def subject_analysis():
    data = load_json(OUTPUTS / "subject_analysis.json")
    if not data:
        raise HTTPException(404, "Chưa có subject_analysis.json. Hãy chạy generate_analysis.py.")
    return data


# ─────────────── hyperparameter search ───────────────────────────────────────

@app.get("/api/hyperparams")
def hyperparams():
    data = load_json(OUTPUTS / "hyperparams.json")
    if not data:
        raise HTTPException(404, "Chưa có hyperparams.json. Hãy chạy generate_analysis.py.")
    return data


# ─────────────── ablation study ──────────────────────────────────────────────

@app.get("/api/ablation")
def ablation():
    data = load_json(OUTPUTS / "ablation.json")
    if not data:
        raise HTTPException(404, "Chưa có ablation.json. Hãy chạy generate_analysis.py.")
    return data


# ─────────────── predict-sample / predict-text / predict (batch) ─────────────

class PredictTextInput(BaseModel):
    text:        str
    model_type:  str = "both"
    true_label:  Optional[str] = None


@app.post("/api/predict-text")
def predict_text(payload: PredictTextInput):
    try:
        feat = parse_feature_text(payload.text)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    bundles = get_model_bundles()
    mode    = payload.model_type.lower()
    resp: Dict[str, Any] = {"true_label": payload.true_label, "model_type": mode}
    if mode in ("mlp", "both") and "mlp" in bundles:
        resp["mlp"] = predict_bundle(bundles["mlp"], feat)[0]
    if mode in ("lr", "both") and "lr" in bundles:
        resp["lr"] = predict_bundle(bundles["lr"], feat)[0]
    if "mlp" in resp and "lr" in resp:
        resp["agreement"] = resp["mlp"]["prediction"] == resp["lr"]["prediction"]
    return resp


class PredictSampleInput(BaseModel):
    sample_id:  int
    model_type: str = "both"


@app.post("/api/predict-sample")
def predict_sample(payload: PredictSampleInput):
    for p in [SAMPLE_DIR / "real_12_rows.csv", APP_DIR / "real_12_rows.csv"]:
        if p.is_file():
            break
    else:
        raise HTTPException(404, "Không tìm thấy tệp mẫu.")

    target_lbl, target_feats = None, None
    with open(p, encoding="utf-8") as f:
        rdr = csv.reader(f)
        next(rdr)
        for idx, row in enumerate(rdr, 1):
            if idx == payload.sample_id:
                target_lbl   = row[0]
                target_feats = [float(x) for x in row[1:]]
                break
    if target_feats is None:
        raise HTTPException(404, f"Không tìm thấy mẫu #{payload.sample_id}")

    feat    = np.asarray([target_feats], dtype=np.float64)
    bundles = get_model_bundles()
    mode    = payload.model_type.lower()
    resp: Dict[str, Any] = {
        "sample_id": payload.sample_id,
        "true_label": target_lbl, "model_type": mode,
    }
    if mode in ("mlp", "both") and "mlp" in bundles:
        resp["mlp"] = predict_bundle(bundles["mlp"], feat)[0]
    if mode in ("lr", "both") and "lr" in bundles:
        resp["lr"]  = predict_bundle(bundles["lr"],  feat)[0]
    if "mlp" in resp and "lr" in resp:
        resp["agreement"] = resp["mlp"]["prediction"] == resp["lr"]["prediction"]
    return resp


@app.post("/api/predict")
async def predict_csv(
    file:        UploadFile = File(...),
    labels_file: Optional[UploadFile] = File(None),
    model_type:  str = Form("mlp"),
):
    if not file.filename or Path(file.filename).suffix.casefold() != ".csv":
        raise HTTPException(400, "Vui lòng tải lên tệp .csv.")

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Tệp vượt quá {MAX_UPLOAD_BYTES//(1024*1024)} MB.")
        features, extracted_labels = parse_feature_csv(contents)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    finally:
        await file.close()

    true_labels = extracted_labels
    if labels_file and labels_file.filename:
        try:
            lc = await labels_file.read(MAX_UPLOAD_BYTES + 1)
            ltext  = lc.decode("utf-8-sig")
            sep_lb = [ln.strip() for ln in ltext.splitlines() if ln.strip()]
            if len(sep_lb) == len(features):
                true_labels = sep_lb
            elif len(sep_lb) == len(features) + 1:
                true_labels = sep_lb[1:]
            else:
                raise HTTPException(422,
                    f"Số nhãn ({len(sep_lb)}) ≠ số dòng đặc trưng ({len(features)}).")
        except UnicodeDecodeError:
            raise HTTPException(422, "Tệp nhãn phải mã hóa UTF-8.")
        finally:
            await labels_file.close()

    bundles  = get_model_bundles()
    cls      = CLASS_NAMES
    mode     = model_type.lower()

    mlp_res = predict_bundle(bundles["mlp"], features) if mode in ("mlp", "both") and "mlp" in bundles else None
    lr_res  = predict_bundle(bundles["lr"],  features) if mode in ("lr",  "both") and "lr"  in bundles else None

    true_norm = None
    if true_labels:
        true_norm = [_normalize_label(l, cls) for l in true_labels]

    results = []
    for i in range(len(features)):
        item: Dict[str, Any] = {"row": i + 1}
        if true_norm:
            item["true_label"] = true_norm[i]
        if mode == "both":
            item["mlp"] = mlp_res[i]
            item["lr"]  = lr_res[i]
            item["agreement"] = mlp_res[i]["prediction"] == lr_res[i]["prediction"]
        elif mode == "lr":
            item.update({k: lr_res[i][k] for k in ("prediction","confidence","low_confidence","probabilities")})
        else:
            item.update({k: mlp_res[i][k] for k in ("prediction","confidence","low_confidence","probabilities")})
        results.append(item)

    resp: Dict[str, Any] = {
        "model_type": mode, "rows": len(results),
        "has_ground_truth": true_norm is not None,
        "results": results,
    }
    if true_norm:
        if mlp_res:
            resp["mlp_metrics"] = calc_metrics(true_norm, [r["prediction"] for r in mlp_res], cls)
        if lr_res:
            resp["lr_metrics"]  = calc_metrics(true_norm, [r["prediction"] for r in lr_res],  cls)
    return resp
