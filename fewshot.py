import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset.tomato_dataset import TomatoDataset
from src.utils.transforms import train_transform, test_transform
from src.models.hybrid_backbone import HybridModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= FEWSHOT SUBSET =================
def create_fewshot_subset(dataset, shots=5):
    class_indices = {}

    for idx, label in enumerate(dataset.labels):
        class_indices.setdefault(label, []).append(idx)

    selected_indices = []

    for label, indices in class_indices.items():
        random.shuffle(indices)
        selected_indices.extend(indices[:shots])

    return Subset(dataset, selected_indices)


# ================= TRAIN =================
def train():

    print("\n🔥 FEW-SHOT TRAINING STARTED\n")

    dataset_path = "D:/tomato/src/dataset/Unified_Tomato_Dataset"

    dataset = TomatoDataset(dataset_path, transform=train_transform())
    print(f"✅ Dataset loaded: {len(dataset)} samples")

    # 🔥 INCREASE SHOTS
    fewshot_dataset = create_fewshot_subset(dataset, shots=50)

    loader = DataLoader(fewshot_dataset, batch_size=16, shuffle=True)

    model = HybridModel(num_classes=len(set(dataset.labels))).to(DEVICE)

    # ================= PARTIAL FINE-TUNING =================
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # ================= LOSS + OPT =================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # 🔥 MORE EPOCHS
    epochs = 25

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels, _ in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

        # 🔥 SAVE BEST MODEL
        if total_loss < best_loss:
            best_loss = total_loss
            os.makedirs("results", exist_ok=True)
            torch.save(model.state_dict(), "results/fewshot_model.pth")
            print("✅ Best Few-Shot Model Saved")

    return model


# ================= TEST =================
def test(model):

    print("\n🔥 FEW-SHOT TESTING\n")

    dataset_path = "D:/tomato/src/dataset/Unified_Tomato_Dataset"

    dataset = TomatoDataset(dataset_path, transform=test_transform())

    shots_list = [1, 5, 10]
    results = {}

    for shots in shots_list:

        subset = create_fewshot_subset(dataset, shots=shots)
        loader = DataLoader(subset, batch_size=16, shuffle=False)

        correct = 0
        total = 0

        model.eval()

        with torch.no_grad():
            for images, labels, _ in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = (correct / total) * 100
        results[f"{shots}-shot"] = acc

        print(f"✅ {shots}-shot Accuracy: {acc:.2f}%")

    return results


# ================= PLOT =================
def plot_results(results):

    os.makedirs("results", exist_ok=True)

    shots = list(results.keys())
    accs = list(results.values())

    plt.figure()

    plt.plot(shots, accs, marker='o')
    plt.title("Few-Shot Learning Performance")
    plt.xlabel("Shots")
    plt.ylabel("Accuracy (%)")

    plt.grid(True)

    save_path = "results/fewshot_performance.png"
    plt.savefig(save_path)

    print(f"\n📊 Graph saved: {save_path}")

    plt.close()


# ================= MAIN =================
def main():

    model = train()

    results = test(model)

    plot_results(results)


if __name__ == "__main__":
    main()