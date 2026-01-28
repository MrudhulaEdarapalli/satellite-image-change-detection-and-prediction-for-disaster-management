import torch
from torch.utils.data import DataLoader
from model import SiameseUNet
from dataset import OneraDataset
import os

def test_model():
    root_dir = "c:/Users/mrudh/OneDrive/Documents/satellite b & a"
    batch_size = 8
    patch_size = 96
    
    # Test cities (using validation cities for now as a proxy for test)
    test_cities = ["rennes", "saclay_e", "aguasclaras"]
    
    test_ds = OneraDataset(root_dir, test_cities, patch_size=patch_size, stride=48, transform=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseUNet().to(device)
    
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    def get_metrics(preds, targets):
        preds = torch.sigmoid(preds) > 0.5
        
        # IoU
        intersection = (preds.bool() & (targets > 0.5).bool()).float().sum()
        union = (preds.bool() | (targets > 0.5).bool()).float().sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        # Accuracy
        correct = (preds == (targets > 0.5)).float().sum()
        total = torch.numel(targets)
        acc = correct / total
        
        return iou.item(), acc.item()

    total_iou = 0
    total_acc = 0
    
    print(f"Starting testing on {len(test_ds)} patches...")
    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output = model(img1, img2)
            iou, acc = get_metrics(output, label)
            total_iou += iou
            total_acc += acc
            
    avg_iou = total_iou / len(test_loader)
    avg_acc = total_acc / len(test_loader)
    
    print("\nTest Results:")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Pixel Accuracy: {avg_acc*100:.2f}%")
    
    if avg_acc > 0.90:
        print("Success! Accuracy is above 90%.")
    else:
        print("Accuracy is below 90%. Consider more training or model adjustments.")

if __name__ == "__main__":
    test_model()
