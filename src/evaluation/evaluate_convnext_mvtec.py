from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Tiny_Weights
from sklearn.metrics import roc_auc_score

from src.data.dataset import MVTecDataset
from src.models.convnext_features import ConvNeXtTinyFeatureExtractor
from src.models.anomaly_detector import FeatureDistanceDetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_FILE = PROJECT_ROOT / "data" / "mvtec_index.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

device = torch.device("cpu")

weights = ConvNeXt_Tiny_Weights.DEFAULT
transform = weights.transforms()

model = ConvNeXtTinyFeatureExtractor().to(device)
model.eval()


categories = sorted(
    pd.read_csv(INDEX_FILE)["category"].unique()
)

results = []


for category in categories:
    print(f"\nEvaluating: {category}")

    train_dataset = MVTecDataset(
        INDEX_FILE,
        split="train",
        transform=transform,
    )

    test_dataset = MVTecDataset(
        INDEX_FILE,
        split="test",
        transform=transform,
    )

    train_dataset.data = train_dataset.data[
        train_dataset.data["category"] == category
    ].reset_index(drop=True)

    test_dataset.data = test_dataset.data[
        test_dataset.data["category"] == category
    ].reset_index(drop=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
    )

    train_features = []

    with torch.no_grad():
        for batch in train_loader:
            images = batch["image"].to(device)
            features = model(images)
            train_features.append(features)

    train_features = torch.cat(train_features, dim=0)

    detector = FeatureDistanceDetector()
    detector.fit(train_features)

    test_features = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)

            features = model(images)

            scores = detector.score(features)

            test_features.append(scores)
            test_labels.append(batch["label"])

    scores = torch.cat(test_features).numpy()
    labels = torch.cat(test_labels).numpy()

    auroc = roc_auc_score(labels, scores)

    print(f"AUROC: {auroc:.4f}")

    results.append(
        {
            "category": category,
            "auroc": auroc,
        }
    )


results_df = pd.DataFrame(results)

output_file = RESULTS_DIR / "baseline_convnext_tiny_mvtec.csv"
results_df.to_csv(output_file, index=False)

mean_auroc = results_df["auroc"].mean()

print("\n" + "=" * 50)
print("ConvNeXt-Tiny MVTec Baseline")
print("=" * 50)
print(results_df.to_string(index=False))
print(f"\nMean AUROC: {mean_auroc:.4f}")
print(f"Results saved to: {output_file}")