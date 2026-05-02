import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import seaborn as sns

from src.dataset.tomato_dataset import TomatoDataset
from src.models.hybrid_backbone import HybridModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# ================= LOAD DATA =================
dataset = TomatoDataset(
    root_dir=r"D:/tomato/src/dataset/Unified_Tomato_Dataset",
    transform=transform
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)
classes = dataset.classes


# ================= LOAD MODEL =================
model = HybridModel(num_classes=10).to(DEVICE)
model.load_state_dict(torch.load("results/best_model.pth", map_location=DEVICE))
model.eval()

print("✅ Model loaded")


# ================= EVALUATION =================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels, _ in tqdm(loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


y_true = np.array(all_labels)
y_pred = np.array(all_preds)


# ================= METRICS =================
print("\n📊 CLASSIFICATION REPORT\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=classes,
    digits=4,
    zero_division=0
)

print(report)

acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Overall Accuracy: {acc*100:.2f}%")


# ================= CONFUSION MATRIX =================
print("\n📊 Generating Confusion Matrix...")

cm = confusion_matrix(y_true, y_pred)

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    annot=True,                # 🔥 show numbers
    fmt='d',
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.tight_layout()

save_path = "results/confusion_matrix.png"
plt.savefig(save_path)

print(f"📊 Confusion matrix saved: {save_path}")


print("\n✅ Evaluation Completed")