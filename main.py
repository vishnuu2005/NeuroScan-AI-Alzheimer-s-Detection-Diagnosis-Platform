import os

# =======================
# Force all caches to D: BEFORE importing torch
# =======================
BASE_DIR = r"D:/AlzeigmersChatBot/AlzeigmersChatBot"
TORCH_CACHE_DIR = os.path.join(BASE_DIR, "torch_cache")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(TORCH_CACHE_DIR, exist_ok=True)

# Environment variables must be set BEFORE importing torch
os.environ['TORCH_HOME'] = TORCH_CACHE_DIR
os.environ['MATPLOTLIBRC'] = os.path.join(BASE_DIR, "matplotlib_cache")

# =======================
# Imports
# =======================
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_preprocessing import get_data_loaders
from models.resnet_model import get_resnet2d

# =======================
# Hyperparameters
# =======================
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_DECAY = 1e-4
PATIENCE = 5
DROPOUT_PROB = 0.6
USE_CLASS_WEIGHT = True  # Set True if dataset is imbalanced

# =======================
# Training / Evaluation Functions
# =======================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100 * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return val_loss / total, 100 * correct / total

# =======================
# Main
# =======================
def main():
    # Load Data
    train_loader, test_loader, classes = get_data_loaders(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        use_augmented=True
    )
    print(f"Classes: {classes}")

    # Compute class weights if needed
    class_weights = None
    if USE_CLASS_WEIGHT:
        counts = [0]*len(classes)
        for _, labels in train_loader:
            for l in labels:
                counts[l] += 1
        total = sum(counts)
        class_weights = [total/c for c in counts]
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        print(f"Class weights: {class_weights}")

    # Model
    model = get_resnet2d(num_classes=len(classes), dropout_prob=DROPOUT_PROB, freeze_layers=True)
    model = model.to(DEVICE)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Training loop
    best_val_acc = 0.0
    counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "resnet2d_alzheimers_best.pth"))
            print(f"Saved best model at epoch {epoch+1} with Val Acc: {val_acc:.2f}%")
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print("Early stopping triggered!")
                break

    # Plot curves
    epochs_range = range(1, len(train_losses)+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs_range, val_accs, 'r-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "training_curves.png"))
    plt.close()

    # Confusion Matrix
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
    plt.close()

    print(f"Training complete. Best model saved in {MODEL_DIR}/resnet2d_alzheimers_best.pth")

if __name__ == "__main__":
    main()

