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

APP_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_DIR / "outputs"
SAMPLE_DIR = APP_DIR / "sample_data"
INDEX_PATH = APP_DIR / "index.html"

MLP_MODEL_PATH = OUTPUTS_DIR / "mlp_har.joblib"
LR_MODEL_PATH = OUTPUTS_DIR / "lr_har.joblib"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"

FEATURE_COUNT = 561
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_ROWS = 10_000
LOW_CONFIDENCE_THRESHOLD = 0.60

app = FastAPI(title="UCI HAR Activity Predictor - AI Realtime")


def _is_number(value: str) -> bool:
    try:
        float(value.strip())
        return True
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=1)
def get_model_bundles() -> Dict[str, Any]:
    bundles = {}
    if MLP_MODEL_PATH.is_file():
        bundles["mlp"] = joblib.load(MLP_MODEL_PATH)
    if LR_MODEL_PATH.is_file():
        bundles["lr"] = joblib.load(LR_MODEL_PATH)

    if not bundles:
        raise HTTPException(
            status_code=503,
            detail=f"Không tìm thấy mô hình trong thư mục {OUTPUTS_DIR}",
        )
    return bundles


def get_model_bundle() -> Dict[str, Any]:
    """Compatibility helper returning default MLP model bundle."""
    bundles = get_model_bundles()
    return bundles.get("mlp") or next(iter(bundles.values()))


def get_default_class_names() -> List[str]:
    bundles = get_model_bundles()
    first_bundle = next(iter(bundles.values()))
    return first_bundle["class_names"]


def _parse_numeric_row(row: List[str], row_number: int) -> List[float]:
    if len(row) != FEATURE_COUNT:
        raise ValueError(
            f"Dòng {row_number} có {len(row)} cột; cần đúng {FEATURE_COUNT} cột."
        )

    values = []
    for column_number, value in enumerate(row, start=1):
        clean_val = value.strip()
        if clean_val == "":
            raise ValueError(
                f"Dữ liệu trống / thiếu giá trị ở dòng {row_number}, cột {column_number}."
            )
        try:
            number = float(clean_val)
        except ValueError as error:
            raise ValueError(
                f"Dữ liệu không phải số ở dòng {row_number}, cột {column_number} (giá trị: '{clean_val}')."
            ) from error
        if not np.isfinite(number):
            raise ValueError(
                f"Dữ liệu NaN/Inf ở dòng {row_number}, cột {column_number}."
            )
        values.append(number)
    return values


def parse_feature_text(text: str) -> np.ndarray:
    text = text.strip()
    if not text:
        raise ValueError("Vui lòng nhập hoặc dán vector đặc trưng.")

    # Determine delimiter: comma if present, else whitespace
    if "," in text:
        tokens = [t.strip() for t in text.split(",")]
    else:
        tokens = text.split()

    # Discard empty trailing token caused by trailing comma or newline
    if tokens and tokens[-1] == "":
        tokens.pop()

    if len(tokens) != FEATURE_COUNT:
        raise ValueError(
            f"Vector có {len(tokens)} giá trị; yêu cầu đúng {FEATURE_COUNT} đặc trưng."
        )

    values = []
    for idx, token in enumerate(tokens, start=1):
        if not token:
            raise ValueError(
                f"Dữ liệu trống / thiếu giá trị ở vị trí đặc trưng thứ {idx}."
            )
        try:
            number = float(token)
        except ValueError as error:
            raise ValueError(
                f"Dữ liệu không phải số ở vị trí đặc trưng thứ {idx} (giá trị: '{token}')."
            ) from error
        if not np.isfinite(number):
            raise ValueError(
                f"Dữ liệu NaN/Inf ở vị trí đặc trưng thứ {idx}."
            )
        values.append(number)

    return np.asarray([values], dtype=np.float64)


def parse_feature_csv(contents: bytes):
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV phải được mã hóa UTF-8.") from error

    try:
        raw_rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(io.StringIO(text))
            if any(cell.strip() for cell in row)
        ]
    except csv.Error as error:
        raise ValueError("Không đọc được cấu trúc CSV.") from error

    if not raw_rows:
        raise ValueError("CSV không có dữ liệu.")

    first_row = raw_rows[0]
    has_header = False

    # Determine if first row is a pure header (column names row, not data).
    # Strategy: a header row's cells are mostly non-numeric strings like "feature_0", "true_label".
    # A data row that starts with a text label still has the remaining 561 cells as numbers.
    # So: if first cell is non-numeric AND the SECOND cell is also non-numeric → almost certainly header.
    # If first cell non-numeric but second cell IS numeric → data row with text label (not a header).
    if len(first_row) >= 2:
        first_non_num = not _is_number(first_row[0])
        second_non_num = not _is_number(first_row[1])
        # Count how many cells (excluding first) are numeric
        rest_numeric_count = sum(1 for v in first_row[1:] if _is_number(v))
        if first_non_num and second_non_num:
            # Both first two cells are text → real header row (e.g. "true_label","feature_0","feature_1"...)
            has_header = True
        elif first_non_num and rest_numeric_count >= FEATURE_COUNT:
            # First cell is text label, rest are 561 numeric features → data row with embedded label
            has_header = False
        else:
            # Fallback: majority non-numeric → header
            numeric_count = sum(1 for v in first_row if _is_number(v))
            has_header = numeric_count < len(first_row) // 2
    else:
        numeric_count = sum(1 for v in first_row if _is_number(v))
        has_header = numeric_count < len(first_row) // 2

    data_rows = raw_rows[1:] if has_header else raw_rows
    if not data_rows:
        raise ValueError("CSV chỉ có dòng tiêu đề, chưa có dòng dữ liệu.")
    if len(data_rows) > MAX_ROWS:
        raise ValueError(f"Tối đa {MAX_ROWS:,} dòng dữ liệu mỗi lần tải lên.")

    row_length = len(data_rows[0])
    extracted_labels: Optional[List[str]] = None

    # Check if label column is included
    if row_length == FEATURE_COUNT + 1:
        # Check if label is at column 0 or last column
        first_col_is_num = _is_number(data_rows[0][0])
        last_col_is_num = _is_number(data_rows[0][-1])

        if not first_col_is_num:
            extracted_labels = [r[0] for r in data_rows]
            data_rows = [r[1:] for r in data_rows]
        elif not last_col_is_num:
            extracted_labels = [r[-1] for r in data_rows]
            data_rows = [r[:-1] for r in data_rows]
        else:
            # Both numeric: assume label is last column
            extracted_labels = [r[-1] for r in data_rows]
            data_rows = [r[:-1] for r in data_rows]
    elif row_length != FEATURE_COUNT:
        raise ValueError(
            f"Mỗi dòng cần có {FEATURE_COUNT} đặc trưng (hoặc {FEATURE_COUNT + 1} nếu kèm nhãn); dòng đầu có {row_length} cột."
        )

    features = [
        _parse_numeric_row(row, row_number)
        for row_number, row in enumerate(data_rows, start=1)
    ]
    return np.asarray(features, dtype=np.float64), extracted_labels


def predict_single_model(bundle: Dict[str, Any], features: np.ndarray) -> List[Dict[str, Any]]:
    model = bundle["model"]
    class_names = bundle["class_names"]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    model_classes = model.classes_

    results = []
    for index, prediction in enumerate(predictions):
        predicted_class = int(prediction)
        prob_dict = {
            class_names[int(cls)]: round(float(probabilities[index, col]), 6)
            for col, cls in enumerate(model_classes)
        }
        confidence = round(float(probabilities[index].max()), 6)
        results.append({
            "row": index + 1,
            "prediction": class_names[predicted_class],
            "confidence": confidence,
            "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
            "probabilities": prob_dict,
        })
    return results


def calculate_metrics_and_cm(true_labels: List[str], predicted_labels: List[str], class_names: List[str]):
    # Normalize labels: if labels are integers 1..6, map them
    normalized_true = []
    for val in true_labels:
        val_str = str(val).strip()
        if val_str.isdigit():
            idx = int(val_str)
            if 1 <= idx <= len(class_names):
                normalized_true.append(class_names[idx - 1])
            elif 0 <= idx < len(class_names):
                normalized_true.append(class_names[idx])
            else:
                normalized_true.append(val_str)
        else:
            # Map case-insensitively
            matched = next((c for c in class_names if c.lower() == val_str.lower()), val_str)
            normalized_true.append(matched)

    correct_count = sum(1 for t, p in zip(normalized_true, predicted_labels) if t == p)
    total_count = len(normalized_true)
    accuracy = round(correct_count / total_count, 4) if total_count > 0 else 0.0

    # Macro-F1 computation
    f1_list = []
    for c in class_names:
        tp = sum(1 for t, p in zip(normalized_true, predicted_labels) if t == c and p == c)
        fp = sum(1 for t, p in zip(normalized_true, predicted_labels) if t != c and p == c)
        fn = sum(1 for t, p in zip(normalized_true, predicted_labels) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
    macro_f1 = round(float(np.mean(f1_list)), 4) if f1_list else 0.0

    # 6x6 Confusion Matrix
    cm_grid = []
    for true_c in class_names:
        row = []
        for pred_c in class_names:
            count = sum(1 for t, p in zip(normalized_true, predicted_labels) if t == true_c and p == pred_c)
            row.append(count)
        cm_grid.append(row)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "correct": correct_count,
        "total": total_count,
        "class_names": class_names,
        "confusion_matrix": cm_grid,
    }


# ---------------- API ROUTES ----------------

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX_PATH)


@app.get("/outputs/{filename}", include_in_schema=False)
def get_output_file(filename: str):
    file_path = OUTPUTS_DIR / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp kết quả.")
    return FileResponse(file_path)


@app.get("/api/health")
def health():
    bundles = get_model_bundles()
    class_names = get_default_class_names()
    return {
        "status": "ok",
        "available_models": list(bundles.keys()),
        "classes": class_names,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
    }


@app.get("/api/model-results")
def get_model_results():
    if not METRICS_PATH.is_file():
        raise HTTPException(status_code=404, detail="Chưa có dữ liệu đánh giá mô hình trong outputs.")
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Attach confusion matrix image URLs
    metrics["images"] = {
        "lr_confusion_matrix": "/outputs/cm_LogisticRegression.png",
        "mlp_confusion_matrix": "/outputs/cm_MLP.png",
    }
    return metrics


@app.get("/api/samples")
def get_samples():
    """Returns curated preset samples from real_12_rows.csv."""
    sample_file = SAMPLE_DIR / "real_12_rows.csv"
    if not sample_file.is_file():
        sample_file = APP_DIR / "real_12_rows.csv"
    if not sample_file.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy file mẫu real_12_rows.csv")

    with open(sample_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        samples = []
        for idx, row in enumerate(reader, start=1):
            true_label = row[0]
            features = [float(x) for x in row[1:]]
            samples.append({
                "id": idx,
                "label": f"Mẫu #{idx:02d} ({true_label})",
                "true_label": true_label,
                "preview": [round(x, 4) for x in features[:5]],
                "features": features,
            })
    return {"samples": samples}


@app.get("/api/sequence")
def get_sequence():
    """Returns 20 consecutive samples for sequence replay simulation."""
    json_path = SAMPLE_DIR / "real_20_sequence.json"
    if not json_path.is_file():
        raise HTTPException(status_code=404, detail="Chưa có dữ liệu chuỗi real_20_sequence.json.")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bundles = get_model_bundles()
    features = np.asarray([item["features"] for item in data], dtype=np.float64)

    mlp_results = predict_single_model(bundles["mlp"], features) if "mlp" in bundles else []
    lr_results = predict_single_model(bundles["lr"], features) if "lr" in bundles else []

    combined_steps = []
    for i, item in enumerate(data):
        combined_steps.append({
            "step": item["step"],
            "true_label": item["true_label"],
            "mlp": mlp_results[i] if mlp_results else None,
            "lr": lr_results[i] if lr_results else None,
        })

    return {
        "title": "Mô phỏng chuỗi 20 mẫu kiểm thử liên tiếp",
        "disclaimer": "⚠️ Lưu ý: Đây là mô phỏng tuần tự từ 20 mẫu kiểm thử đã thu thập trước, không phải tín hiệu thời gian thực từ cảm biến sống.",
        "steps": combined_steps,
    }


class PredictTextInput(BaseModel):
    text: str
    model_type: str = "both"  # "mlp", "lr", "both"
    true_label: Optional[str] = None


@app.post("/api/predict-text")
def predict_text(payload: PredictTextInput):
    try:
        features = parse_feature_text(payload.text)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    bundles = get_model_bundles()
    model_mode = payload.model_type.lower()

    response: Dict[str, Any] = {
        "true_label": payload.true_label,
        "model_type": model_mode,
    }

    if model_mode in ("mlp", "both") and "mlp" in bundles:
        response["mlp"] = predict_single_model(bundles["mlp"], features)[0]
    if model_mode in ("lr", "both") and "lr" in bundles:
        response["lr"] = predict_single_model(bundles["lr"], features)[0]

    if "mlp" in response and "lr" in response:
        response["agreement"] = response["mlp"]["prediction"] == response["lr"]["prediction"]

    return response


class PredictSampleInput(BaseModel):
    sample_id: int
    model_type: str = "both"  # "mlp", "lr", "both"


@app.post("/api/predict-sample")
def predict_sample(payload: PredictSampleInput):
    sample_file = SAMPLE_DIR / "real_12_rows.csv"
    if not sample_file.is_file():
        sample_file = APP_DIR / "real_12_rows.csv"
    if not sample_file.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp mẫu.")

    target_row = None
    target_label = None
    with open(sample_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for idx, row in enumerate(reader, start=1):
            if idx == payload.sample_id:
                target_label = row[0]
                target_row = [float(x) for x in row[1:]]
                break

    if target_row is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy mẫu số #{payload.sample_id}")

    features = np.asarray([target_row], dtype=np.float64)
    bundles = get_model_bundles()
    model_mode = payload.model_type.lower()

    response: Dict[str, Any] = {
        "sample_id": payload.sample_id,
        "true_label": target_label,
        "model_type": model_mode,
    }

    if model_mode in ("mlp", "both") and "mlp" in bundles:
        response["mlp"] = predict_single_model(bundles["mlp"], features)[0]
    if model_mode in ("lr", "both") and "lr" in bundles:
        response["lr"] = predict_single_model(bundles["lr"], features)[0]

    if "mlp" in response and "lr" in response:
        response["agreement"] = response["mlp"]["prediction"] == response["lr"]["prediction"]

    return response


@app.post("/api/predict")
async def predict_csv(
    file: UploadFile = File(...),
    labels_file: Optional[UploadFile] = File(None),
    model_type: str = Form("mlp"),
):
    if not file.filename or Path(file.filename).suffix.casefold() != ".csv":
        raise HTTPException(status_code=400, detail="Vui lòng tải lên tệp đặc trưng .csv.")

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Tệp vượt quá giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        features, extracted_labels = parse_feature_csv(contents)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        await file.close()

    # If separate labels file provided
    true_labels = extracted_labels
    if labels_file and labels_file.filename:
        try:
            label_contents = await labels_file.read(MAX_UPLOAD_BYTES + 1)
            label_text = label_contents.decode("utf-8-sig")
            separate_labels = [line.strip() for line in label_text.splitlines() if line.strip()]
            if len(separate_labels) == len(features):
                true_labels = separate_labels
            elif len(separate_labels) == len(features) + 1:
                # Discard possible header
                true_labels = separate_labels[1:]
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"Số nhãn trong tệp nhãn ({len(separate_labels)}) không khớp số dòng đặc trưng ({len(features)}).",
                )
        except UnicodeDecodeError as err:
            raise HTTPException(status_code=422, detail="Tệp nhãn phải được mã hóa UTF-8.") from err
        finally:
            await labels_file.close()

    bundles = get_model_bundles()
    class_names = get_default_class_names()
    model_mode = model_type.lower()

    response: Dict[str, Any] = {
        "model_type": model_mode,
        "rows": len(features),
        "has_ground_truth": true_labels is not None,
    }

    mlp_results = predict_single_model(bundles["mlp"], features) if (model_mode in ("mlp", "both") and "mlp" in bundles) else None
    lr_results = predict_single_model(bundles["lr"], features) if (model_mode in ("lr", "both") and "lr" in bundles) else None

    # Merge results
    results = []
    for i in range(len(features)):
        item: Dict[str, Any] = {"row": i + 1}
        if true_labels:
            item["true_label"] = true_labels[i]

        if model_mode == "both":
            item["mlp"] = mlp_results[i]
            item["lr"] = lr_results[i]
            item["agreement"] = mlp_results[i]["prediction"] == lr_results[i]["prediction"]
        elif model_mode == "lr" and lr_results:
            item["prediction"] = lr_results[i]["prediction"]
            item["confidence"] = lr_results[i]["confidence"]
            item["low_confidence"] = lr_results[i]["low_confidence"]
            item["probabilities"] = lr_results[i]["probabilities"]
        else:
            item["prediction"] = mlp_results[i]["prediction"]
            item["confidence"] = mlp_results[i]["confidence"]
            item["low_confidence"] = mlp_results[i]["low_confidence"]
            item["probabilities"] = mlp_results[i]["probabilities"]
        results.append(item)

    response["results"] = results

    # If ground truth labels are provided, compute evaluation metrics & confusion matrix
    if true_labels:
        if model_mode in ("mlp", "both") and mlp_results:
            pred_mlp = [r["prediction"] for r in mlp_results]
            response["mlp_metrics"] = calculate_metrics_and_cm(true_labels, pred_mlp, class_names)
        if model_mode in ("lr", "both") and lr_results:
            pred_lr = [r["prediction"] for r in lr_results]
            response["lr_metrics"] = calculate_metrics_and_cm(true_labels, pred_lr, class_names)

    return response
