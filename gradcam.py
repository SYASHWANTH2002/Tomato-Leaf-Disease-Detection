import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from src.dataset.tomato_dataset import TomatoDataset
from src.models.hybrid_backbone import HybridModel
from src.utils.transforms import val_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===== CLASSES =====
CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Tomato_YellowLeaf_Curl_Virus"
]


# ===== GRADCAM =====
class GradCAM:

    def __init__(self, model):
        self.model = model
        self.feature_map = None

    def save_feature(self, module, input, output):
        # take last feature map
        self.feature_map = output[-1]
        self.feature_map.retain_grad()

    def generate(self, x, class_idx):

        handle = self.model.convnext.register_forward_hook(self.save_feature)

        output = self.model(x)

        self.model.zero_grad()

        loss = output[:, class_idx]
        loss.backward()

        handle.remove()

        gradients = self.feature_map.grad[0]   # [C,H,W]
        activations = self.feature_map[0]

        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(DEVICE)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)

        cam = cam.cpu().detach().numpy()

        # 🔥 better resize
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)

        # normalize
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        # 🔥 sharpen
        cam = np.power(cam, 0.5)

        return cam


# ===== OVERLAY FUNCTION =====
def save_overlay(image_tensor, cam, save_path):

    img = image_tensor[0].permute(1, 2, 0).cpu().numpy()

    # normalize image
    img = (img - img.min()) / (img.max() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap / 255.0

    # 🔥 balanced blending
    overlay = heatmap * 0.4 + img * 0.6

    overlay = overlay / overlay.max()

    cv2.imwrite(save_path, np.uint8(255 * overlay))


# ===== MAIN =====
def main():

    print("\n🔥 GRADCAM STARTED\n")

    dataset = TomatoDataset(
        root_dir=r"D:/tomato/data/combined_test_dataset",
        transform=val_transform()
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = HybridModel(len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load("results/best_model.pth", map_location=DEVICE))
    model.eval()

    cam_generator = GradCAM(model)

    base_dir = "results/gradcam"
    os.makedirs(base_dir, exist_ok=True)

    class_counts = {cls: 0 for cls in CLASSES}

    print("\n🔍 Generating GradCAM overlays...\n")

    for image, label, _ in tqdm(loader):

        image = image.to(DEVICE)
        label = label.item()
        class_name = CLASSES[label]

        if class_counts[class_name] >= 5:
            continue

        output = model(image)
        pred = torch.argmax(output, dim=1).item()

        cam = cam_generator.generate(image, pred)

        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, f"{class_counts[class_name]}.png")

        save_overlay(image, cam, save_path)

        class_counts[class_name] += 1

        if all(v >= 5 for v in class_counts.values()):
            break

    print("\n✅ DONE: 5 GradCAM overlays per class saved")


if __name__ == "__main__":
    main()