import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

from disaggregate import run_disaggregation
from preprocessing import preprocess_house
from refit_metadata import get_appliance_column, parse_house_number

train_house_csv = "../Processed_Data_CSV/House_9.csv"
test_house_csv = "../Processed_Data_CSV/House_3.csv"
apps = ["kettle", "microwave", "fridge", "tv"]
limit = 200

train_house_number = parse_house_number(train_house_csv)
test_house_number = parse_house_number(test_house_csv)

results = run_disaggregation(
    house_csv=test_house_csv,
    target_appliances=apps,
    nilm_mode=False,
    limit=limit,
    plot=False,
    train_house_number=train_house_number,
)

df = preprocess_house(test_house_csv).iloc[:limit]

metrics = []
for appliance in apps:
    state_col = f"{appliance}_state_label"
    if state_col not in results.columns:
        continue

    col = get_appliance_column(test_house_number, appliance)
    if col is None or col not in df.columns:
        continue

    gt_on = (df[col] > 10).astype(int)
    pred_on = (results[state_col] != "OFF").astype(int)

    common_idx = results.index.intersection(df.index)
    y_true = gt_on.loc[common_idx].values
    y_pred = pred_on.loc[common_idx].values

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics.append((appliance, precision, recall, f1))

print("METRICS_START")
for app, p, r, f in metrics:
    print(f"{app}: precision={p:.4f}, recall={r:.4f}, f1={f:.4f}")

labels = [m[0] for m in metrics]
precisions = [m[1] for m in metrics]
recalls = [m[2] for m in metrics]
f1s = [m[3] for m in metrics]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width, precisions, width, label="Precision", color="#4e79a7")
ax.bar(x, recalls, width, label="Recall", color="#f28e2b")
ax.bar(x + width, f1s, width, label="F1-score", color="#59a14f")
ax.set_xticks(x)
ax.set_xticklabels([lbl.capitalize() for lbl in labels])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title(f"Precision / Recall / F1 (small dataset, limit={limit})")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper right")
plt.tight_layout()

plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(plots_dir, exist_ok=True)
out_path = os.path.join(plots_dir, f"metrics_prf_small_{limit}.png")
plt.savefig(out_path, dpi=130)
plt.close(fig)

print(f"PLOT_PATH={out_path}")
print("METRICS_END")
