import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from backend.model.model import SiameseUNet
from backend.model.dataset import OneraDataset
import os
try:
    import torch_directml
except ImportError:
    torch_directml = None

import sys

import datetime

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("training.log", "a", encoding="utf-8")

    def write(self, message):
        if message.strip():
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"[{timestamp}] {message}"
            self.terminal.write(formatted_message)
            self.log.write(formatted_message)
        else:
            self.terminal.write(message)
            self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()
sys.stderr = sys.stdout

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

def train_model():
    # Parameters
    root_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    batch_size = 8
    epochs = 30
    lr = 0.0001
    patch_size = 96
    stride = 48

    # Dataset
    train_cities = ["abudhabi", "beihai", "beirut", "bercy", "bordeaux", "cupertino", "hongkong", "mumbai", "nantes", "paris", "pisa"]
    val_cities = ["rennes", "saclay_e", "aguasclaras"]

    train_ds = OneraDataset(root_dir, train_cities, patch_size=patch_size, stride=stride, transform=True)
    val_ds = OneraDataset(root_dir, val_cities, patch_size=patch_size, stride=stride, transform=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model, Optimizer, Loss
    if torch_directml and torch_directml.is_available():
        device = torch_directml.device()
        print(f"Using DirectML (AMD GPU) device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using standard device: {device}")

    model = SiameseUNet().to(device)
    
    # Load checkpoint if exists
    checkpoint_path = "best_model.pth"
    start_epoch = 0
    best_val_iou = 0.0

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_iou = checkpoint.get('best_val_iou', 0.0)
                print(f"Successfully loaded checkpoint at epoch {start_epoch}.")
            else:
                # Fallback for old state_dict only files
                model.load_state_dict(checkpoint)
                print("Successfully loaded model weights (no epoch info found).")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    def get_iou(preds, targets):
        preds = torch.sigmoid(preds) > 0.5
        intersection = (preds.bool() & (targets > 0.5).bool()).float().sum()
        union = (preds.bool() | (targets > 0.5).bool()).float().sum()
        return (intersection + 1e-6) / (union + 1e-6)

    def get_pixel_accuracy(preds, targets):
        preds = torch.sigmoid(preds) > 0.5
        correct = (preds == (targets > 0.5)).float().sum()
        total = torch.numel(targets)
        return correct / total

    print(f"Starting training on {device} from epoch {start_epoch+1}...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        train_iou = 0
        train_acc = 0
        for i, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(img1, img2)
            
            loss = bce_loss(output, label) + dice_loss(output, label)
            loss.backward()
            optimizer.step()
            
            iou = get_iou(output, label).item()
            acc = get_pixel_accuracy(output, label).item()
            
            train_loss += loss.item()
            train_iou += iou
            train_acc += acc
            
            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}, IoU: {iou:.4f}, Acc: {acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        val_acc = 0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output = model(img1, img2)
                loss = bce_loss(output, label) + dice_loss(output, label)
                
                val_loss += loss.item()
                val_iou += get_iou(output, label).item()
                val_acc += get_pixel_accuracy(output, label).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train -> Loss: {avg_train_loss:.4f}, IoU: {avg_train_iou:.4f}, Acc: {avg_train_acc:.4f}")
        print(f"  Val   -> Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}, Acc: {avg_val_acc:.4f}")

        scheduler.step(avg_val_iou)

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
            }, os.path.join(os.path.dirname(__file__), "best_model.pth"))
            print(f"Saved best model with Val IoU: {avg_val_iou:.4f}")
    
    print("Training process completed successfully.")

if __name__ == "__main__":
    train_model()
