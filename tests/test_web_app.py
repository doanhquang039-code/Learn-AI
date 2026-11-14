import csv
import io
import unittest

import numpy as np
from fastapi.testclient import TestClient

from web_app import (
    app,
    get_model_bundle,
    get_model_bundles,
    parse_feature_csv,
    parse_feature_text,
)


def make_csv(rows, header=False):
    output = io.StringIO()
    writer = csv.writer(output)
    if header:
        writer.writerow([f"feature_{index}" for index in range(len(rows[0]))])
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


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
                files={"file": ("bad.csv", make_csv([[0.0] * 560]), "text/csv")},
            )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
