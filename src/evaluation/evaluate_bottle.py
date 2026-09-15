import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
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
            batch_features = model(batch["image"])
            features.append(batch_features)
            labels.append(batch["label"])

    return torch.cat(features), torch.cat(labels)


def main():
    index_file = Path("data/mvtec_index.csv")

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

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
        train_dataset.data["category"] == "bottle"
    ].reset_index(drop=True)

    test_dataset.data = test_dataset.data[
        test_dataset.data["category"] == "bottle"
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

    model = ResNet18FeatureExtractor()
    model.eval()

    print("Extracting normal training features...")
    train_features, _ = extract_features(model, train_loader)

    print("Extracting test features...")
    test_features, test_labels = extract_features(model, test_loader)

    detector = FeatureDistanceDetector()
    detector.fit(train_features)

    scores = detector.score(test_features)

    auc = roc_auc_score(
        test_labels.numpy(),
        scores.numpy(),
    )

    print()
    print("=== VisionBench Bottle Baseline ===")
    print("Training images:", len(train_dataset))
    print("Test images:", len(test_dataset))
    print("Feature dimension:", train_features.shape[1])
    print("AUROC:", round(auc, 4))


if __name__ == "__main__":
    main()