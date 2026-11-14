import requests, json

BASE = "http://127.0.0.1:8000"
PASS = []
FAIL = []

def chk(label, cond, detail=""):
    if cond:
        PASS.append(label)
        print(f"  ✅ {label} {detail}")
    else:
        FAIL.append(label)
        print(f"  ❌ {label} {detail}")

print("=" * 60)
print("TEST 1: Health check")
r = requests.get(f"{BASE}/api/health")
d = r.json()
chk("Status 200", r.status_code == 200)
chk("Both models available", set(d.get("available_models", [])) == {"mlp", "lr"})
chk("6 classes", len(d.get("classes", [])) == 6)
chk("Low conf threshold = 0.60", d.get("low_confidence_threshold") == 0.6)

print("\nTEST 2: Samples list (12 real test rows)")
r = requests.get(f"{BASE}/api/samples")
d = r.json()
samples = d.get("samples", [])
chk("Status 200", r.status_code == 200)
chk("12 samples loaded", len(samples) == 12)
chk("Each sample has 561 features", all(len(s["features"]) == 561 for s in samples))
chk("Each sample has true_label", all(s.get("true_label") for s in samples))
classes_in_samples = set(s["true_label"] for s in samples)
chk("All 6 classes represented", len(classes_in_samples) == 6, str(classes_in_samples))

print("\nTEST 3: Predict sample #1 WALKING (both models)")
r = requests.post(f"{BASE}/api/predict-sample", json={"sample_id": 1, "model_type": "both"})
d = r.json()
chk("Status 200", r.status_code == 200)
chk("Has true_label", "true_label" in d)
chk("MLP prediction present", "mlp" in d)
chk("LR prediction present", "lr" in d)
chk("MLP has 6-class probs", len(d["mlp"]["probabilities"]) == 6)
chk("LR has 6-class probs", len(d["lr"]["probabilities"]) == 6)
chk("MLP probs sum ~= 1.0", abs(sum(d["mlp"]["probabilities"].values()) - 1.0) < 0.01)
chk("Agreement field present", "agreement" in d)
chk("MLP low_confidence field", "low_confidence" in d["mlp"])
print(f"     > WALKING -> MLP: {d['mlp']['prediction']} ({d['mlp']['confidence']:.3f}) | LR: {d['lr']['prediction']} ({d['lr']['confidence']:.3f})")

print("\nTEST 4: Predict SITTING sample (sample #7) - check SITTING/STANDING confusion visibility")
r = requests.post(f"{BASE}/api/predict-sample", json={"sample_id": 7, "model_type": "both"})
d = r.json()
chk("Status 200", r.status_code == 200)
true_lbl = d.get("true_label", "?")
mlp_probs = d["mlp"]["probabilities"]
lr_probs = d["lr"]["probabilities"]
print(f"     > true={true_lbl} | MLP: {d['mlp']['prediction']} ({d['mlp']['confidence']:.3f}) | LR: {d['lr']['prediction']} ({d['lr']['confidence']:.3f})")
print(f"     > MLP sitting={mlp_probs.get('SITTING',0):.3f} standing={mlp_probs.get('STANDING',0):.3f}")
print(f"     > LR  sitting={lr_probs.get('SITTING',0):.3f} standing={lr_probs.get('STANDING',0):.3f}")

print("\nTEST 5: Text/paste vector input - valid 561 numbers")
valid_text = ",".join(["0.05"] * 561)
r = requests.post(f"{BASE}/api/predict-text", json={"text": valid_text, "model_type": "both"})
d = r.json()
chk("Status 200", r.status_code == 200)
chk("MLP result present", "mlp" in d)
chk("LR result present", "lr" in d)

print("\nTEST 6: Text input - wrong column count (550 instead of 561)")
bad_cols = ",".join(["0.1"] * 550)
r = requests.post(f"{BASE}/api/predict-text", json={"text": bad_cols, "model_type": "mlp"})
chk("Returns 422", r.status_code == 422)
err_detail = r.json().get("detail", "")
chk("Error mentions 561", "561" in err_detail, f'("{err_detail[:80]}")')

print("\nTEST 7: Text input - contains non-numeric text")
tokens = ["0.123"] * 561
tokens[10] = "invalid_abc"
bad_text = ",".join(tokens)
r = requests.post(f"{BASE}/api/predict-text", json={"text": bad_text, "model_type": "mlp"})
chk("Returns 422", r.status_code == 422)
err_detail = r.json().get("detail", "")
chk("Error mentions 'không phải số'", "không phải số" in err_detail, f'("{err_detail[:80]}")')

print("\nTEST 8: Text input - empty cell")
tokens2 = ["0.123"] * 561
tokens2[8] = ""
bad_empty = ",".join(tokens2)
r = requests.post(f"{BASE}/api/predict-text", json={"text": bad_empty, "model_type": "mlp"})
chk("Returns 422", r.status_code == 422)
err_detail = r.json().get("detail", "")
chk("Error mentions 'trống'", "trống" in err_detail, f'("{err_detail[:80]}")')

print("\nTEST 9: Batch CSV upload - no labels (features only)")
with open("sample_data/real_12_rows_features_only.csv", "rb") as f:
    r = requests.post(f"{BASE}/api/predict",
                      files={"file": ("test.csv", f, "text/csv")},
                      data={"model_type": "mlp"})
d = r.json()
chk("Status 200", r.status_code == 200)
chk("12 rows returned", d.get("rows") == 12)
chk("No ground truth", d.get("has_ground_truth") == False)
chk("No mlp_metrics (no labels)", "mlp_metrics" not in d)
chk("low_confidence field present in result", "low_confidence" in d["results"][0])
chk("6-class probabilities present", len(d["results"][0]["probabilities"]) == 6)
print(f"     > Low conf rows: {sum(1 for r2 in d['results'] if r2['low_confidence'])}/12")

print("\nTEST 10: Batch CSV upload WITH labels (mini-Colab mode)")
with open("sample_data/real_12_rows.csv", "rb") as f:
    r = requests.post(f"{BASE}/api/predict",
                      files={"file": ("test.csv", f, "text/csv")},
                      data={"model_type": "both"})
d = r.json()
chk("Status 200", r.status_code == 200)
chk("12 rows returned", d.get("rows") == 12)
chk("has_ground_truth = True", d.get("has_ground_truth") == True)
chk("mlp_metrics present", "mlp_metrics" in d)
chk("lr_metrics present", "lr_metrics" in d)
m = d["mlp_metrics"]
chk("Accuracy in mlp_metrics", "accuracy" in m)
chk("macro_f1 in mlp_metrics", "macro_f1" in m)
chk("confusion_matrix in mlp_metrics", "confusion_matrix" in m)
chk("Confusion matrix is 6x6", len(m["confusion_matrix"]) == 6 and all(len(row) == 6 for row in m["confusion_matrix"]))
print(f"     > MLP accuracy on 12 samples: {m['accuracy']:.4f} | Macro-F1: {m['macro_f1']:.4f}")
print(f"     > LR  accuracy on 12 samples: {d['lr_metrics']['accuracy']:.4f} | Macro-F1: {d['lr_metrics']['macro_f1']:.4f}")

print("\nTEST 11: Sequence replay (20 steps)")
r = requests.get(f"{BASE}/api/sequence")
d = r.json()
chk("Status 200", r.status_code == 200)
chk("20 steps", len(d.get("steps", [])) == 20)
chk("Disclaimer present", "Lưu ý" in d.get("disclaimer", "") or "mô phỏng" in d.get("disclaimer", "").lower())
step1 = d["steps"][0]
chk("Each step has MLP result", step1.get("mlp") is not None)
chk("Each step has LR result", step1.get("lr") is not None)
chk("Each step has true_label", step1.get("true_label") is not None)
print(f"     > Step 1: true={step1['true_label']} MLP={step1['mlp']['prediction']} LR={step1['lr']['prediction']}")

print("\nTEST 12: Model results from outputs/")
r = requests.get(f"{BASE}/api/model-results")
d = r.json()
chk("Status 200", r.status_code == 200)
chk("summary present", "summary" in d)
chk("class_f1 present", "class_f1" in d)
chk("seed_results present", "seed_results" in d)
chk("images present", "images" in d)
chk("LR CM image URL", "/outputs/cm_LogisticRegression.png" in d.get("images", {}).values())
chk("MLP CM image URL", "/outputs/cm_MLP.png" in d.get("images", {}).values())
chk("5 seed results", len(d.get("seed_results", [])) == 5)
chk("6 class F1 rows", len(d.get("class_f1", [])) == 6)
chk("seed_stats present", "seed_stats" in d)

print("\nTEST 13: Static output images serve correctly")
for img in ["cm_LogisticRegression.png", "cm_MLP.png"]:
    r = requests.get(f"{BASE}/outputs/{img}")
    chk(f"{img} serves 200", r.status_code == 200)
    chk(f"{img} is PNG", r.headers.get("content-type", "").startswith("image/png"))
    chk(f"{img} size > 0", len(r.content) > 10000, f"({len(r.content)} bytes)")

print("\nTEST 14: Bad CSV upload - wrong column count")
bad_csv = b"0.1,0.2,0.3\n"
r = requests.post(f"{BASE}/api/predict",
                  files={"file": ("bad.csv", bad_csv, "text/csv")},
                  data={"model_type": "mlp"})
chk("Returns 422 for bad col count", r.status_code == 422)

print("\nTEST 15: bad_contains_text.csv file")
with open("sample_data/bad_contains_text.csv", "rb") as f:
    r = requests.post(f"{BASE}/api/predict",
                      files={"file": ("bad.csv", f, "text/csv")},
                      data={"model_type": "mlp"})
chk("bad_contains_text returns 422", r.status_code == 422)

print("\nTEST 16: bad_empty_cell.csv file")
with open("sample_data/bad_empty_cell.csv", "rb") as f:
    r = requests.post(f"{BASE}/api/predict",
                      files={"file": ("bad.csv", f, "text/csv")},
                      data={"model_type": "mlp"})
chk("bad_empty_cell returns 422", r.status_code == 422)

print()
print("=" * 60)
print(f"TỔNG KẾT: ✅ {len(PASS)} PASS  |  ❌ {len(FAIL)} FAIL")
if FAIL:
    print("Các test bị lỗi:")
    for f in FAIL:
        print(f"  ❌ {f}")
else:
    print("🎉 Toàn bộ tests PASS!")

