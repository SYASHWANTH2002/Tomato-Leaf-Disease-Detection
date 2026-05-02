import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

from src.dataset.tomato_dataset import TomatoDataset
from src.models.hybrid_backbone import HybridModel
from src.utils.transforms import train_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():

    dataset = TomatoDataset(
        root_dir=r"src/dataset/Unified_Tomato_Dataset",
        transform=train_transform()
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    model = HybridModel(10).to(DEVICE)

    # ===== CLASS WEIGHTS =====
    labels_list = [label for _, label in dataset.samples]
    counts = Counter(labels_list)

    weights = [1.0 / counts[i] for i in range(len(counts))]
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    best_loss = float("inf")
    patience = 5
    no_improve = 0

    EPOCHS = 15

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        loop = tqdm(loader)

        for images, labels, _ in loop:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            outputs = torch.clamp(outputs, -10, 10)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)

        print(f"\nEpoch {epoch+1} Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "results/best_model.pth")
            print("✅ Best model saved")
            no_improve = 0
        else:
            no_improve += 1
            print(f"⚠️ No improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            print("🛑 Early stopping triggered")
            break

    print("\n🔥 TRAINING COMPLETE")


if __name__ == "__main__":
    train()