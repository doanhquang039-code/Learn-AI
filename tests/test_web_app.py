import csv
import io
import unittest

import numpy as np
from fastapi.testclient import TestClient

from web_app import app, get_model_bundle, parse_feature_csv


def make_csv(rows, header=False):
    output = io.StringIO()
    writer = csv.writer(output)
    if header:
        writer.writerow([f"feature_{index}" for index in range(561)])
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


class WebAppTests(unittest.TestCase):
    def test_parser_accepts_header_and_headerless_csv(self):
        samples = [[float(index) / 1000 for index in range(561)]]

        with_header = parse_feature_csv(make_csv(samples, header=True))
        without_header = parse_feature_csv(make_csv(samples))

        np.testing.assert_array_equal(with_header, samples)
        np.testing.assert_array_equal(without_header, samples)

    def test_parser_rejects_wrong_feature_count(self):
        with self.assertRaisesRegex(ValueError, "561"):
            parse_feature_csv(make_csv([[0.0] * 560]))

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
            )

        self.assertEqual(response.status_code, 200)
        result = response.json()["results"][0]
        self.assertEqual(result["prediction"], bundle["class_names"][expected_prediction])
        self.assertEqual(result["confidence"], round(float(expected_probabilities.max()), 6))

    def test_api_rejects_csv_with_wrong_feature_count(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                files={"file": ("bad.csv", make_csv([[0.0] * 560]), "text/csv")},
            )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
