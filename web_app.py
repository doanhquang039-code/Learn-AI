import csv
import io
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get(
    "HAR_MODEL_PATH",
    APP_DIR / "outputs" / "mlp_har.joblib",
))
INDEX_PATH = APP_DIR / "index.html"
FEATURE_COUNT = 561
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 10_000

app = FastAPI(title="UCI HAR Activity Predictor")


def _is_number(value):
    try:
        float(value.strip())
        return True
    except (TypeError, ValueError):
        return False


def _parse_numeric_row(row, row_number):
    if len(row) != FEATURE_COUNT:
        raise ValueError(
            f"Dòng {row_number} có {len(row)} cột; cần đúng {FEATURE_COUNT} cột."
        )

    values = []
    for column_number, value in enumerate(row, start=1):
        try:
            number = float(value.strip())
        except ValueError as error:
            raise ValueError(
                f"Dữ liệu không phải số ở dòng {row_number}, cột {column_number}."
            ) from error
        if not np.isfinite(number):
            raise ValueError(
                f"Dữ liệu NaN/Inf ở dòng {row_number}, cột {column_number}."
            )
        values.append(number)
    return values


def parse_feature_csv(contents):
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV phải được mã hóa UTF-8.") from error

    try:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(io.StringIO(text))
            if any(cell.strip() for cell in row)
        ]
    except csv.Error as error:
        raise ValueError("Không đọc được cấu trúc CSV.") from error

    if not rows:
        raise ValueError("CSV không có dữ liệu.")

    first_row = rows[0]
    if len(first_row) != FEATURE_COUNT:
        raise ValueError(
            f"Mỗi dòng phải có đúng {FEATURE_COUNT} đặc trưng; "
            f"dòng đầu có {len(first_row)} cột."
        )

    if not any(_is_number(value) for value in first_row):
        rows = rows[1:]

    if not rows:
        raise ValueError("CSV chỉ có tiêu đề, chưa có dòng dữ liệu.")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"Tối đa {MAX_ROWS:,} dòng dữ liệu mỗi lần tải lên.")

    features = [
        _parse_numeric_row(row, row_number)
        for row_number, row in enumerate(rows, start=1)
    ]
    return np.asarray(features, dtype=np.float64)


@lru_cache(maxsize=1)
def get_model_bundle():
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Không tìm thấy mô hình đã huấn luyện: {MODEL_PATH}",
        )

    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle or "class_names" not in bundle:
        raise RuntimeError("Tệp mô hình không đúng định dạng HAR.")
    if not hasattr(bundle["model"], "predict_proba"):
        raise RuntimeError("Mô hình HAR không hỗ trợ dự đoán xác suất.")
    return bundle


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX_PATH)


@app.get("/api/health")
def health():
    bundle = get_model_bundle()
    return {"status": "ok", "model": "MLP", "classes": bundle["class_names"]}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.casefold() != ".csv":
        raise HTTPException(status_code=400, detail="Vui lòng tải lên tệp .csv.")

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Tệp vượt quá giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        features = parse_feature_csv(contents)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        await file.close()

    bundle = get_model_bundle()
    model = bundle["model"]
    class_names = bundle["class_names"]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    model_classes = model.classes_

    results = []
    for index, prediction in enumerate(predictions):
        predicted_class = int(prediction)
        if predicted_class < 0 or predicted_class >= len(class_names):
            raise RuntimeError("Mô hình trả về mã lớp không hợp lệ.")

        probability_by_class = {
            class_names[int(model_class)]: round(float(probabilities[index, column]), 6)
            for column, model_class in enumerate(model_classes)
        }
        results.append({
            "row": index + 1,
            "prediction": class_names[predicted_class],
            "confidence": round(float(probabilities[index].max()), 6),
            "probabilities": probability_by_class,
        })

    return {
        "model": "MLP",
        "rows": len(results),
        "results": results,
    }
