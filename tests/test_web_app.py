import csv
import io
import unittest
import zipfile
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from fastapi.testclient import TestClient

from web_app import (
    app,
    get_model_bundle,
    get_model_bundles,
    parse_feature_csv,
    parse_feature_xlsx,
    parse_feature_text,
)


def make_csv(rows, header=False):
    output = io.StringIO()
    writer = csv.writer(output)
    if header:
        writer.writerow([f"feature_{index}" for index in range(len(rows[0]))])
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def make_xlsx(rows, header=False):
    workbook = Workbook()
    worksheet = workbook.active
    if header:
        worksheet.append([f"feature_{index}" for index in range(len(rows[0]))])
    for row in rows:
        worksheet.append(row)
    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


class WebAppTests(unittest.TestCase):
    def test_parser_accepts_header_and_headerless_csv(self):
        samples = [[float(index) / 1000 for index in range(561)]]

        with_header, labels1 = parse_feature_csv(make_csv(samples, header=True))
        without_header, labels2 = parse_feature_csv(make_csv(samples))

        np.testing.assert_array_equal(with_header, samples)
        np.testing.assert_array_equal(without_header, samples)
        self.assertIsNone(labels1)
        self.assertIsNone(labels2)

    def test_parser_accepts_embedded_label_column(self):
        samples = [["WALKING"] + [float(index) / 1000 for index in range(561)]]
        features, labels = parse_feature_csv(make_csv(samples))
        self.assertEqual(features.shape, (1, 561))
        self.assertEqual(labels, ["WALKING"])

    def test_parser_skips_label_header_with_numeric_feature_names(self):
        samples = [["WALKING"] + [float(index) / 1000 for index in range(561)]]
        rows = [["true_label"] + list(range(561))] + samples
        features, labels = parse_feature_csv(make_csv(rows))
        self.assertEqual(features.shape, (1, 561))
        self.assertEqual(labels, ["WALKING"])

    def test_xlsx_parser_accepts_header_and_embedded_label(self):
        samples = [["WALKING"] + [float(index) / 1000 for index in range(561)]]
        rows = [["true_label"] + list(range(561))] + samples
        features, labels = parse_feature_xlsx(make_xlsx(rows))
        self.assertEqual(features.shape, (1, 561))
        self.assertEqual(labels, ["WALKING"])
        np.testing.assert_array_equal(features[0], samples[0][1:])

    def test_xlsx_parser_rejects_wrong_feature_count(self):
        with self.assertRaisesRegex(ValueError, "561"):
            parse_feature_xlsx(make_xlsx([[0.0] * 560]))

    def test_xlsx_parser_rejects_invalid_workbook(self):
        with self.assertRaisesRegex(ValueError, "không hợp lệ"):
            parse_feature_xlsx(b"not an Excel workbook")

    def test_parser_rejects_wrong_feature_count(self):
        with self.assertRaisesRegex(ValueError, "561"):
            parse_feature_csv(make_csv([[0.0] * 560]))

    def test_parser_rejects_empty_cells_and_text(self):
        with self.assertRaises(ValueError):
            parse_feature_csv(make_csv([["0.1"] * 560 + [""]]))
        with self.assertRaises(ValueError):
            parse_feature_csv(make_csv([["0.1"] * 560 + ["abc"]]))

    def test_parse_feature_text_valid_and_invalid(self):
        # Valid comma-separated
        valid_comma = ",".join(["0.5"] * 561)
        res = parse_feature_text(valid_comma)
        self.assertEqual(res.shape, (1, 561))

        # Valid space-separated
        valid_space = " ".join(["0.5"] * 561)
        res_space = parse_feature_text(valid_space)
        self.assertEqual(res_space.shape, (1, 561))

        # Invalid column count
        with self.assertRaisesRegex(ValueError, "561"):
            parse_feature_text("0.1, 0.2, 0.3")

        # Invalid non-numeric token
        tokens = ["0.1"] * 561
        tokens[10] = "error_text"
        with self.assertRaisesRegex(ValueError, "không phải số"):
            parse_feature_text(",".join(tokens))

        # Empty token
        tokens[10] = ""
        with self.assertRaisesRegex(ValueError, "trống"):
            parse_feature_text(",".join(tokens))

    def test_prediction_matches_saved_mlp(self):
        features = np.zeros((1, 561), dtype=np.float64)
        bundle = get_model_bundle()
        model = bundle["model"]
        expected_prediction = int(model.predict(features)[0])
        expected_probabilities = model.predict_proba(features)[0]

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                files={"file": ("sample.csv", make_csv(features.tolist()), "text/csv")},
                data={"model_type": "mlp"},
            )

        self.assertEqual(response.status_code, 200)
        result = response.json()["results"][0]
        self.assertEqual(result["prediction"], bundle["class_names"][expected_prediction])
        self.assertEqual(result["confidence"], round(float(expected_probabilities.max()), 6))

    def test_prediction_accepts_xlsx_upload(self):
        features = np.zeros((1, 561), dtype=np.float64)
        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                files={"file": ("sample.xlsx", make_xlsx(features.tolist()), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                data={"model_type": "mlp"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["rows"], 1)

    def test_predict_sample_and_compare_models(self):
        with TestClient(app) as client:
            # Test getting samples list
            samples_resp = client.get("/api/samples")
            self.assertEqual(samples_resp.status_code, 200)
            samples = samples_resp.json()["samples"]
            self.assertGreaterEqual(len(samples), 12)

            # Test predicting sample 1 with both models
            resp = client.post(
                "/api/predict-sample",
                json={"sample_id": 1, "model_type": "both"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("mlp", data)
            self.assertIn("lr", data)
            self.assertIn("agreement", data)
            self.assertEqual(len(data["mlp"]["probabilities"]), 6)
            self.assertEqual(len(data["lr"]["probabilities"]), 6)

    def test_predict_text_endpoint(self):
        valid_text = ",".join(["0.05"] * 561)
        with TestClient(app) as client:
            resp = client.post(
                "/api/predict-text",
                json={"text": valid_text, "model_type": "both"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("mlp", data)
            self.assertIn("lr", data)

    def test_model_results_endpoint(self):
        with TestClient(app) as client:
            resp = client.get("/api/model-results")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("summary", data)
            self.assertIn("class_f1", data)
            self.assertIn("seed_results", data)
            self.assertIn("images", data)

    def test_sequence_endpoint(self):
        with TestClient(app) as client:
            resp = client.get("/api/sequence")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("steps", data)
            self.assertEqual(len(data["steps"]), 20)

    def test_api_rejects_csv_with_wrong_feature_count(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                files={"file": ("TotalAcceleration.csv", make_csv([[0.0] * 5]), "text/csv")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("561", response.json()["detail"])
        self.assertIn("TotalAcceleration.csv", response.json()["detail"])

    def test_sample_files_have_expected_shapes_and_balanced_labels(self):
        sample_dir = Path(__file__).resolve().parents[1] / "sample_data"
        for row_count in (60, 120, 300, 600):
            for suffix, parser in (
                ("csv", parse_feature_csv),
                ("xlsx", parse_feature_xlsx),
            ):
                sample_path = sample_dir / f"uci_har_{row_count}_rows_labeled.{suffix}"
                self.assertLess(sample_path.stat().st_size, 4 * 1024 * 1024)
                features, labels = parser(sample_path.read_bytes())
                self.assertEqual(features.shape, (row_count, 561))
                self.assertEqual(len(labels), row_count)
                self.assertEqual(
                    {labels.count(label) for label in set(labels)},
                    {row_count // 6},
                )

    def test_sensor_inspection_previews_csv_and_allows_empty_annotation(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/inspect-sensor-files",
                files=[
                    ("files", ("TotalAcceleration.csv", b"time,seconds_elapsed,z,y,x\n1,0.02,9.8,0.1,0.2\n2,0.04,9.7,0.2,0.3\n", "text/csv")),
                    ("files", ("Annotation.csv", b"", "text/csv")),
                ],
            )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["file_count"], 2)
        self.assertEqual(result["total_rows"], 2)
        self.assertEqual(result["files"][0]["sensor_type"], "TotalAcceleration")
        self.assertEqual(result["files"][0]["columns"], [
            "time", "seconds_elapsed", "z", "y", "x",
        ])
        self.assertEqual(result["files"][0]["rows"], 2)
        self.assertEqual(result["files"][1]["sensor_type"], "Annotation")
        self.assertTrue(result["files"][1]["empty"])

    def test_sensor_inspection_rejects_unknown_file_name(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/inspect-sensor-files",
                files={"files": ("custom.csv", b"a,b\n1,2\n", "text/csv")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("6 loại hỗ trợ", response.json()["detail"])

    def test_sensor_inspection_rejects_inconsistent_csv_rows(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/inspect-sensor-files",
                files={"files": ("TotalAcceleration.csv", b"time,x,y,z\n1,2,3,4\n5,6\n", "text/csv")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("dòng 3", response.json()["detail"])

    def test_batch_prediction_compares_both_models_on_labeled_csv_and_xlsx(self):
        sample_dir = Path(__file__).resolve().parents[1] / "sample_data"
        with TestClient(app) as client:
            for row_count in (60, 600):
                for suffix in ("csv", "xlsx"):
                    filename = f"uci_har_{row_count}_rows_labeled.{suffix}"
                    sample_path = sample_dir / filename
                    response = client.post(
                        "/api/predict",
                        files={"file": (filename, sample_path.read_bytes(), "application/octet-stream")},
                        data={"model_type": "both"},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    data = response.json()
                    self.assertEqual(data["rows"], row_count)
                    self.assertTrue(data["has_ground_truth"])
                    self.assertIn("mlp_metrics", data)
                    self.assertIn("lr_metrics", data)

    def test_sample_download_endpoint_serves_csv_and_excel(self):
        with TestClient(app) as client:
            filenames = [
                f"uci_har_{row_count}_rows_labeled.{suffix}"
                for row_count in (60, 120, 300, 600)
                for suffix in ("csv", "xlsx")
            ]
            for filename in filenames:
                response = client.get(f"/sample-data/{filename}")
                self.assertEqual(response.status_code, 200)
                self.assertGreater(len(response.content), 0)

            response = client.get("/sample-data/uci_har_test_samples.zip")
            self.assertEqual(response.status_code, 200)
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(len(archive.namelist()), 9)


if __name__ == "__main__":
    unittest.main()
