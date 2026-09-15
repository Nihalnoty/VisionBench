import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from sklearn.metrics import roc_auc_score

from src.data.dataset import MVTecDataset
from src.models.resnet_features import ResNet18FeatureExtractor
from src.models.anomaly_detector import FeatureDistanceDetector


def extract_features(model, loader):
    features = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            features.append(model(batch["image"]))
            labels.append(batch["label"])

    return torch.cat(features), torch.cat(labels)


def main():
    index_file = Path("data/mvtec_index.csv")

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    full_index = pd.read_csv(index_file)
    categories = sorted(full_index["category"].unique())

    model = ResNet18FeatureExtractor()
    model.eval()

    results = []

    for category in categories:
        print(f"\nEvaluating: {category}")

        train_dataset = MVTecDataset(
            index_file,
            split="train",
            transform=transform,
        )

        test_dataset = MVTecDataset(
            index_file,
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
            batch_size=16,
            shuffle=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
        )

        train_features, _ = extract_features(model, train_loader)
        test_features, test_labels = extract_features(model, test_loader)

        detector = FeatureDistanceDetector()
        detector.fit(train_features)

        scores = detector.score(test_features)

        auc = roc_auc_score(
            test_labels.numpy(),
            scores.numpy(),
        )

        results.append({
            "category": category,
            "train_images": len(train_dataset),
            "test_images": len(test_dataset),
            "auroc": auc,
        })

        print(f"AUROC: {auc:.4f}")

    results_df = pd.DataFrame(results)

    output_path = Path("results/baseline_resnet18_mvtec.csv")
    results_df.to_csv(output_path, index=False)

    print("\n=== VisionBench Baseline #1 ===")
    print(results_df.to_string(index=False))
    print(f"\nMean AUROC: {results_df['auroc'].mean():.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()