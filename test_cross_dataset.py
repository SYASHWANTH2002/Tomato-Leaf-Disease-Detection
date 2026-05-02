import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset.tomato_dataset import TomatoDataset
from src.models.hybrid_backbone import HybridModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def main():

    print("\n🔥 CROSS DATASET TEST\n")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = TomatoDataset(
        root_dir=r"D:/tomato/data/combined_test_dataset",
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = HybridModel(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load("results/best_model.pth", map_location=DEVICE))
    model.eval()

    acc = evaluate(model, loader)

    print(f"\n✅ Cross Dataset Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()