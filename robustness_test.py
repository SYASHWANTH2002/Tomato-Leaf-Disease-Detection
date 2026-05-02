import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter

from src.dataset.tomato_dataset import TomatoDataset
from src.models.hybrid_backbone import HybridModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= NOISE TRANSFORMS =================

class GaussianNoise:
    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, img):
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, self.std, arr.shape)
        arr = np.clip(arr + noise, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


class GaussianBlur:
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=1.5))


class BrightnessChange:
    def __init__(self, factor=1.2):
        self.factor = factor

    def __call__(self, img):
        return ImageEnhance.Brightness(img).enhance(self.factor)


class ContrastChange:
    def __init__(self, factor=1.2):
        self.factor = factor

    def __call__(self, img):
        return ImageEnhance.Contrast(img).enhance(self.factor)


# ================= EVALUATION =================

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels, _ in tqdm(loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return (correct / total) * 100


# ================= MAIN =================

def main():

    print("\n🔥 ROBUSTNESS TEST (UNIFIED DATASET)\n")

    # ================= LOAD MODEL =================
    model = HybridModel(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load("results/best_model.pth", map_location=DEVICE))
    model.eval()
    print("✅ Model loaded")

    # ================= DATASET PATH =================
    DATASET_PATH = r"D:/tomato/src/dataset/Unified_Tomato_Dataset"

    # ================= TRANSFORMS =================
    transforms_dict = {

        "Original": transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),

        "Gaussian Noise": transforms.Compose([
            transforms.Resize((224,224)),
            GaussianNoise(0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),

        "Gaussian Blur": transforms.Compose([
            transforms.Resize((224,224)),
            GaussianBlur(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),

        "Brightness Change": transforms.Compose([
            transforms.Resize((224,224)),
            BrightnessChange(1.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),

        "Contrast Change": transforms.Compose([
            transforms.Resize((224,224)),
            ContrastChange(1.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    }

    results = {}

    # ================= RUN TEST =================
    for name, transform in transforms_dict.items():

        print(f"\n🔹 Testing: {name}")

        dataset = TomatoDataset(
            root_dir=DATASET_PATH,
            transform=transform
        )

        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        acc = evaluate(model, loader)

        print(f"✅ {name} Accuracy: {acc:.2f}%")

        results[name] = acc

    # ================= PLOT GRAPH =================

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(10,6))

    names = list(results.keys())
    values = list(results.values())

    plt.bar(names, values)

    # 🔥 FIX LABEL CUT ISSUE
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    plt.ylabel("Accuracy (%)")
    plt.title("Robustness Test (Unified Dataset)")

    save_path = "results/robustness_unified.png"
    plt.savefig(save_path)

    print(f"\n📊 Graph saved: {save_path}")


if __name__ == "__main__":
    main()