from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from sklearn.metrics import roc_auc_score

from src.data.dataset import MVTecDataset
from src.models.padim import PaDiMFeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_FILE = PROJECT_ROOT / "data" / "mvtec_index.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

device = torch.device("cpu")

BATCH_SIZE = 8

# Maximum number of feature dimensions retained.
FEATURE_DIM = 100

# Small regularization value for covariance stability.
EPSILON = 0.01


weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

model = PaDiMFeatureExtractor().to(device)
model.eval()


index_df = pd.read_csv(INDEX_FILE)

categories = sorted(
    index_df["category"].unique()
)

results = []


for category in categories:

    print("\n" + "=" * 60)
    print(f"Evaluating PaDiM: {category}")
    print("=" * 60)

    train_dataset = MVTecDataset(
        INDEX_FILE,
        split="train",
        transform=transform,
    )

    train_dataset.data = train_dataset.data[
        train_dataset.data["category"] == category
    ].reset_index(drop=True)

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
    # Extract normal training patch features
    # -----------------------------------------------------

    train_features = []

    with torch.no_grad():

        for batch in train_loader:

            images = batch["image"].to(device)

            features = model(images)

            train_features.append(
                features.cpu()
            )

    train_features = torch.cat(
        train_features,
        dim=0
    )

    # [N, patches, dimensions]

    num_images = train_features.shape[0]
    num_patches = train_features.shape[1]
    original_dim = train_features.shape[2]

    print(
        f"Feature shape: "
        f"{train_features.shape}"
    )


    # -----------------------------------------------------
    # Random feature-dimension selection
    # -----------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(42)

    selected_dim = min(
        FEATURE_DIM,
        original_dim
    )

    selected_indices = torch.randperm(
        original_dim,
        generator=generator,
    )[:selected_dim]

    train_features = train_features[
        :, :, selected_indices
    ]


    # -----------------------------------------------------
    # Learn mean and covariance for every patch location
    # -----------------------------------------------------

    means = train_features.mean(dim=0)

    means_np = means.numpy()

    train_np = train_features.numpy()

    # Covariance matrices for each spatial patch.
    covariance_matrices = []

    for patch_idx in range(num_patches):

        patch_data = train_np[:, patch_idx, :]

        covariance = np.cov(
            patch_data,
            rowvar=False,
        )

        if covariance.ndim == 0:
            covariance = np.array(
                [[float(covariance)]]
            )

        covariance = covariance + (
            EPSILON *
            np.eye(selected_dim)
        )

        covariance_matrices.append(
            np.linalg.pinv(covariance)
        )

    inverse_covariances = np.stack(
        covariance_matrices,
        axis=0
    )


    # -----------------------------------------------------
    # Score test images
    # -----------------------------------------------------

    image_scores = []
    image_labels = []

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(device)

            features = model(images)

            features = features[
                :, :, selected_indices
            ]

            features_np = features.cpu().numpy()

            batch_size = features_np.shape[0]

            scores = np.zeros(
                (batch_size, num_patches)
            )

            for patch_idx in range(num_patches):

                difference = (
                    features_np[:, patch_idx, :]
                    -
                    means_np[patch_idx]
                )

                covariance_inverse = (
                    inverse_covariances[patch_idx]
                )

                mahalanobis_distance = np.einsum(
                    "bi,ij,bj->b",
                    difference,
                    covariance_inverse,
                    difference,
                )

                scores[:, patch_idx] = (
                    mahalanobis_distance
                )

            # Most anomalous spatial location
            # determines image-level score.

            image_batch_scores = scores.max(
                axis=1
            )

            image_scores.extend(
                image_batch_scores.tolist()
            )

            image_labels.extend(
                batch["label"].numpy().tolist()
            )


    # -----------------------------------------------------
    # AUROC
    # -----------------------------------------------------

    auroc = roc_auc_score(
        image_labels,
        image_scores,
    )

    print(
        f"AUROC: {auroc:.4f}"
    )

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
    "baseline_padim_mvtec.csv"
)

results_df.to_csv(
    output_file,
    index=False,
)

mean_auroc = results_df["auroc"].mean()


print("\n" + "=" * 60)
print("PADIM MVTec BASELINE")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print(
    f"\nMean AUROC: {mean_auroc:.4f}"
)

print(
    f"Results saved to: {output_file}"
)