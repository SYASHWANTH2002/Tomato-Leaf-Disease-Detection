import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset.tomato_dataset import TomatoDataset
from src.models.hybrid_backbone import HybridModel
from src.utils.transforms import val_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CLASSES = [
    "Bacterial_spot",
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Mosaic_virus",
    "YellowLeafCurl"
]


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def main():

    print("\n🔥 UNCERTAINTY + OVERALL CONFIDENCE\n")

    dataset = TomatoDataset(
        root_dir=r"D:/tomato/data/combined_test_dataset",
        transform=val_transform()
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = HybridModel(len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load("results/best_model.pth", map_location=DEVICE))
    model.eval()

    enable_dropout(model)

    T = 10

    uncertainties = []
    confidences = []
    image_paths = []

    print("\n🔍 Running MC Dropout...\n")

    for images, labels, paths in tqdm(loader):

        images = images.to(DEVICE)

        outputs_list = []

        for _ in range(T):
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            outputs_list.append(probs.detach().cpu().numpy())

        outputs_stack = np.stack(outputs_list, axis=0)

        mean_probs = outputs_stack.mean(axis=0)[0]

        # ===== ENTROPY =====
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))

        confidence = np.max(mean_probs)

        uncertainties.append(entropy)
        confidences.append(confidence)

        image_paths.append(str(paths[0]))

    uncertainties = np.array(uncertainties)
    confidences = np.array(confidences)

    os.makedirs("results/uncertainty/high_uncertainty", exist_ok=True)
    os.makedirs("results/uncertainty/low_confidence", exist_ok=True)

    # ===== SAVE ARRAYS =====
    np.save("results/uncertainty.npy", uncertainties)
    np.save("results/confidence.npy", confidences)

    print("\n✅ Arrays saved")

    # ===== 🔥 OVERALL CONFIDENCE =====
    overall_confidence = np.mean(confidences)
    overall_uncertainty = np.mean(uncertainties)

    print("\n📊 OVERALL METRICS")
    print(f"Average Confidence   : {overall_confidence:.4f}")
    print(f"Average Uncertainty  : {overall_uncertainty:.4f}")

    # ===== TOP UNCERTAIN =====
    idx_uncertain = np.argsort(-uncertainties)[:20]

    print("\n⚠️ Top Uncertain Samples:")
    for i in idx_uncertain:
        print(f"{i} → Uncertainty: {uncertainties[i]:.4f}, Confidence: {confidences[i]:.4f}")

        path = image_paths[i]
        if os.path.exists(path):
            img = cv2.imread(path)
            cv2.imwrite(f"results/uncertainty/high_uncertainty/{i}.png", img)

    # ===== LOW CONFIDENCE =====
    idx_low_conf = np.argsort(confidences)[:20]

    print("\n❌ Lowest Confidence Samples:")
    for i in idx_low_conf:
        print(f"{i} → Confidence: {confidences[i]:.4f}, Uncertainty: {uncertainties[i]:.4f}")

        path = image_paths[i]
        if os.path.exists(path):
            img = cv2.imread(path)
            cv2.imwrite(f"results/uncertainty/low_confidence/{i}.png", img)

    # ===== PLOT =====
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(confidences, bins=30)
    plt.title("Confidence Distribution")

    plt.subplot(1, 2, 2)
    plt.hist(uncertainties, bins=30)
    plt.title("Uncertainty Distribution")

    plt.tight_layout()
    plt.savefig("results/uncertainty_analysis.png")
    plt.close()

    print("\n📊 Plot saved")

    # ===== REJECTION SYSTEM =====
    threshold = 0.5
    rejected = np.sum(confidences < threshold)

    print("\n🚨 Rejection System")
    print(f"Rejected samples: {rejected}")


if __name__ == "__main__":
    main()