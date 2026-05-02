import os
from PIL import Image
from torch.utils.data import Dataset


class TomatoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # ================= STANDARD CLASS ORDER (TRAINING) =================
        self.classes = [
            'Tomato_Bacterial_spot',
            'Tomato_Early_blight',
            'Tomato_healthy',
            'Tomato_Late_blight',
            'Tomato_Leaf_Mold',
            'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites',
            'Tomato_Target_Spot',
            'Tomato_mosaic_virus',
            'Tomato_YellowLeaf_Curl_Virus'
        ]

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # ================= CLASS NAME FIX (IMPORTANT) =================
        self.class_map = {
            'Tomato_Bacterial_spot': 'Tomato_Bacterial_spot',
            'Tomato_Early_blight': 'Tomato_Early_blight',
            'Tomato_healthy': 'Tomato_healthy',
            'Tomato_Late_blight': 'Tomato_Late_blight',
            'Tomato_Leaf_Mold': 'Tomato_Leaf_Mold',
            'Tomato_Septoria_leaf_spot': 'Tomato_Septoria_leaf_spot',

            # 🔥 FIXED DIFFERENT NAMES
            'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato_Spider_mites',
            'Tomato_Target_Spot': 'Tomato_Target_Spot',
            'Tomato_Tomato_mosaic_virus': 'Tomato_mosaic_virus',
            'Tomato_Tomato_YellowLeaf_Curl_Virus': 'Tomato_YellowLeaf_Curl_Virus'
        }

        self.image_paths = []
        self.labels = []

        # ================= LOAD DATA =================
        for folder_name in os.listdir(root_dir):

            folder_path = os.path.join(root_dir, folder_name)

            if not os.path.isdir(folder_path):
                continue

            if folder_name not in self.class_map:
                print(f"⚠️ Skipping unknown folder: {folder_name}")
                continue

            mapped_class = self.class_map[folder_name]
            label = self.class_to_idx[mapped_class]

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)

                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        # ================= DEBUG =================
        print(f"\n✅ Dataset loaded: {len(self.image_paths)} images")
        print("📊 Classes:", self.classes)

        # Show class distribution
        from collections import Counter
        print("\n📊 Class Distribution:")
        print(Counter(self.labels))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, img_path