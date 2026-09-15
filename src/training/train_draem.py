from pathlib import Path

import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from src.data.dataset import MVTecDataset
from src.models.draem import DRAEM


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_FILE = PROJECT_ROOT / "data" / "mvtec_index.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

device = torch.device("cpu")

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4

IMAGE_SIZE = 224


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

dataset = MVTecDataset(
    INDEX_FILE,
    split="train",
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = DRAEM().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ---------------------------------------------------------
# Synthetic anomaly generation
# ---------------------------------------------------------

def create_synthetic_anomaly(images):
    """
    Create simple synthetic defects on normal images.

    Returns:
        corrupted_images
        anomaly_masks
    """

    corrupted = images.clone()

    masks = torch.zeros(
        images.shape[0],
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    for i in range(images.shape[0]):

        # Random rectangle location
        x1 = random.randint(20, 150)
        y1 = random.randint(20, 150)

        width = random.randint(15, 60)
        height = random.randint(15, 60)

        x2 = min(x1 + width, IMAGE_SIZE)
        y2 = min(y1 + height, IMAGE_SIZE)

        # Random synthetic defect type
        defect_type = random.choice(
            ["noise", "dark", "bright"]
        )

        if defect_type == "noise":

            noise = torch.randn(
                3,
                y2 - y1,
                x2 - x1,
            ) * 0.5

            corrupted[
                i,
                :,
                y1:y2,
                x1:x2
            ] += noise

        elif defect_type == "dark":

            corrupted[
                i,
                :,
                y1:y2,
                x1:x2
            ] *= 0.15

        else:

            corrupted[
                i,
                :,
                y1:y2,
                x1:x2
            ] = 1.0

        masks[
            i,
            :,
            y1:y2,
            x1:x2
        ] = 1.0

    corrupted = torch.clamp(
        corrupted,
        0.0,
        1.0,
    )

    return corrupted, masks


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

print("=" * 60)
print("DRAEM TRAINING")
print("=" * 60)

print(f"Training images: {len(dataset)}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Device: {device}")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch in loader:

        normal_images = batch["image"].to(device)

        # Create synthetic defective images
        anomalous_images, anomaly_masks = (
            create_synthetic_anomaly(
                normal_images
            )
        )

        anomalous_images = anomalous_images.to(device)
        anomaly_masks = anomaly_masks.to(device)

        # Forward pass
        reconstruction, anomaly_logits = model(
            anomalous_images
        )

        # Reconstruction loss
        reconstruction_loss = F.l1_loss(
            reconstruction,
            normal_images,
        )

        # Segmentation loss
        segmentation_loss = F.binary_cross_entropy_with_logits(
            anomaly_logits,
            anomaly_masks,
        )

        # Combined DRAEM-style objective
        loss = (
            reconstruction_loss
            +
            segmentation_loss
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss /
        len(loader)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )


# ---------------------------------------------------------
# Save trained model
# ---------------------------------------------------------

model_path = (
    MODEL_DIR /
    "draem_mvtec.pth"
)

torch.save(
    model.state_dict(),
    model_path,
)

print("\nTraining complete.")
print(
    f"Model saved to: {model_path}"
)