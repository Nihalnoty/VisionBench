from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from src.data.dataset import MVTecDataset
from src.models.patchcore import PatchCoreFeatureExtractor


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_FILE = PROJECT_ROOT / "data" / "mvtec_index.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

device = torch.device("cpu")

BATCH_SIZE = 8
MEMORY_BANK_SIZE = 5000
RANDOM_SEED = 42


# ---------------------------------------------------------
# Preprocessing and model
# ---------------------------------------------------------

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

model = PatchCoreFeatureExtractor().to(device)
model.eval()


# ---------------------------------------------------------
# Load categories
# ---------------------------------------------------------

index_df = pd.read_csv(INDEX_FILE)

categories = sorted(
    index_df["category"].unique()
)

results = []


# ---------------------------------------------------------
# Evaluate every MVTec category
# ---------------------------------------------------------

for category in categories:

    print("\n" + "=" * 60)
    print(f"Evaluating PatchCore: {category}")
    print("=" * 60)

    # Training data: normal images only
    train_dataset = MVTecDataset(
        INDEX_FILE,
        split="train",
        transform=transform,
    )

    train_dataset.data = train_dataset.data[
        train_dataset.data["category"] == category
    ].reset_index(drop=True)

    # Test data: normal + anomalous images
    test_dataset = MVTecDataset(
        INDEX_FILE,
        split="test",
        transform=transform,
    )

    test_dataset.data = test_dataset.data[
        test_dataset.data["category"] == category
    ].reset_index(drop=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Normal training images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")


    # -----------------------------------------------------
    # Build normal patch memory bank
    # -----------------------------------------------------

    memory_bank = []

    with torch.no_grad():

        for batch in train_loader:

            images = batch["image"].to(device)

            patch_features = model(images)

            patch_features = patch_features.reshape(
                -1,
                patch_features.shape[-1],
            )

            memory_bank.append(
                patch_features.cpu()
            )

    memory_bank = torch.cat(
        memory_bank,
        dim=0
    )

    print(
        f"Extracted normal patches: "
        f"{memory_bank.shape[0]}"
    )


    # -----------------------------------------------------
    # Coreset sampling
    # -----------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    if memory_bank.shape[0] > MEMORY_BANK_SIZE:

        indices = torch.randperm(
            memory_bank.shape[0],
            generator=generator,
        )[:MEMORY_BANK_SIZE]

        memory_bank = memory_bank[indices]

    print(
        f"Memory bank size: "
        f"{memory_bank.shape[0]}"
    )


    # -----------------------------------------------------
    # Nearest-neighbor search
    # -----------------------------------------------------

    nearest_neighbor = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
        n_jobs=-1,
    )

    nearest_neighbor.fit(
        memory_bank.numpy()
    )


    # -----------------------------------------------------
    # Score test images
    # -----------------------------------------------------

    image_scores = []
    image_labels = []

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(device)

            patch_features = model(images)

            batch_size = patch_features.shape[0]
            num_patches = patch_features.shape[1]
            feature_dim = patch_features.shape[2]

            patch_features_flat = patch_features.reshape(
                -1,
                feature_dim,
            )

            distances, _ = nearest_neighbor.kneighbors(
                patch_features_flat.cpu().numpy(),
                return_distance=True,
            )

            distances = torch.tensor(
                distances,
                dtype=torch.float32,
            )

            distances = distances.reshape(
                batch_size,
                num_patches,
            )

            # Most anomalous patch determines
            # the image-level anomaly score.
            scores = distances.max(dim=1).values

            image_scores.extend(
                scores.numpy().tolist()
            )

            image_labels.extend(
                batch["label"].numpy().tolist()
            )


    # -----------------------------------------------------
    # Calculate AUROC
    # -----------------------------------------------------

    auroc = roc_auc_score(
        image_labels,
        image_scores,
    )

    print(f"AUROC: {auroc:.4f}")

    results.append(
        {
            "category": category,
            "auroc": auroc,
        }
    )


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

output_file = (
    RESULTS_DIR /
    "baseline_patchcore_mvtec.csv"
)

results_df.to_csv(
    output_file,
    index=False,
)


# ---------------------------------------------------------
# Final results
# ---------------------------------------------------------

mean_auroc = results_df["auroc"].mean()

print("\n" + "=" * 60)
print("PATCHCORE MVTec BASELINE")
print("=" * 60)

print(
    results_df.to_string(index=False)
)

print(
    f"\nMean AUROC: {mean_auroc:.4f}"
)

print(
    f"Results saved to: {output_file}"
)