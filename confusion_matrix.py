import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Path to your log folder
log_dir = "YOLO_Confusion_Snaps"

# Initialize totals
total_expected = {}
total_expected_not_ok = {}
total_ok_detected = {}
total_not_ok_detected = {}

# Aggregate data from all JSON files
for filename in os.listdir(log_dir):
    if filename.endswith(".json"):
        with open(os.path.join(log_dir, filename), 'r') as f:
            data = json.load(f)
            for key in data["expected"]:
                for d, name in zip(
                    [total_expected, total_expected_not_ok, total_ok_detected, total_not_ok_detected],
                    ["expected", "expected_not_ok", "ok_detected", "not_ok_detected"]
                ):
                    d[key] = d.get(key, 0) + data[name].get(key, 0)

# Confusion matrix: [[TP, FN], [FP, TN]]
conf_matrix = np.zeros((2, 2), dtype=int)
labels = sorted(total_expected.keys())

for label in labels:
    expected_ok = total_expected.get(label, 0)
    expected_not_ok = total_expected_not_ok.get(label, 0)
    detected_ok = total_ok_detected.get(label, 0)
    detected_not_ok = total_not_ok_detected.get(label, 0)

    # True Positive: correctly detected OK
    conf_matrix[0, 0] += min(detected_ok, expected_ok)

    # False Negative: missed OK
    conf_matrix[0, 1] += max(0,expected_ok - detected_ok)

    # False Positive: wrongly detected NOT OK
    conf_matrix[1, 0] += max(0,detected_not_ok - expected_not_ok)

    # True Negative: correctly detected NOT OK
    conf_matrix[1, 1] += min(detected_not_ok, expected_not_ok)

# Plotting using matplotlib only
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(conf_matrix, cmap='Oranges')

# Labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["OK", "NOT OK"])
ax.set_yticklabels(["OK (Expected)", "NOT OK (Expected)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Overall Confusion Matrix")

# Annotate values
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="black")

plt.colorbar(im)
plt.tight_layout()
plt.show()
