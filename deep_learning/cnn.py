import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from tqdm import tqdm

config = {
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 0.001,
}

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
    ]
)

train_data = datasets.MNIST(
    "./data", train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    "./data", train=False, download=True, transform=transform
)

print(f"Train data shape: {train_data.data.shape}")
print(f"Test data shape: {test_data.data.shape}")

train_loader = DataLoader(
    train_data, batch_size=config["batch_size"], shuffle=True, drop_last=False
)
test_loader = DataLoader(
    test_data, batch_size=config["batch_size"], shuffle=False, drop_last=False
)

print(f"Train Loader size: {len(train_loader)}")
print(f"Test Loader size: {len(test_loader)}")


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input dimensions: [batch_size, 1, 28, 28] - MNIST images are 28x28 with 1 channel (grayscale)
        
        # First convolutional layer
        # Conv2d(in_channels, out_channels, kernel_size)
        # Output dimensions: [batch_size, 32, 26, 26]
        # Calculation: 28 - 3 + 1 = 26 (no padding)
        self.conv1 = nn.Conv2d(1, 32, 3)  
        
        # First pooling layer
        # MaxPool2d(kernel_size)
        # Output dimensions: [batch_size, 32, 13, 13]
        # Calculation: 26 / 2 = 13
        self.pool1 = nn.MaxPool2d(2)

        # Second convolutional layer
        # Output dimensions: [batch_size, 64, 10, 10]
        # Calculation: 13 - 4 + 1 = 10 (no padding)
        self.conv2 = nn.Conv2d(32, 64, 4)
        
        # Second pooling layer
        # Output dimensions: [batch_size, 64, 5, 5]
        # Calculation: 10 / 2 = 5
        self.pool2 = nn.MaxPool2d(2)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # Fully connected layers
        # After flattening, dimensions: [batch_size, 64*5*5] = [batch_size, 1600]
        self.fc1 = nn.Linear(64*5*5, 128)  # Output: [batch_size, 128]
        self.fc2 = nn.Linear(128, 64)      # Output: [batch_size, 64]
        self.fc3 = nn.Linear(64, 10)       # Output: [batch_size, 10] - 10 classes for MNIST

    def forward(self, x):
        # x starts with shape: [batch_size, 1, 28, 28]
        
        # First conv + activation + pooling
        x = self.pool1(self.relu(self.conv1(x)))  # Shape: [batch_size, 32, 13, 13]
        x = self.dropout(x)
        
        # Second conv + activation + pooling
        x = self.pool2(self.relu(self.conv2(x)))  # Shape: [batch_size, 64, 5, 5]

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64*5*5)  # Shape: [batch_size, 1600]
        
        # Fully connected layers
        x = self.dropout(self.fc1(x))  # Shape: [batch_size, 128]
        x = self.fc2(x)               # Shape: [batch_size, 64]
        return self.fc3(x)            # Shape: [batch_size, 10]

# ==================================================
# Training Loop
# ==================================================
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

def train_model():
    model.train()
    for epoch in range(config["epochs"]):
        train_accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
        running_loss = 0.0
        # Add tqdm progress bar for each epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
            preds = torch.argmax(output, dim=1)
            train_accuracy.update(preds, target)
            loss.backward()
            optimizer.step()
            
            # Update progress bar with current loss
            progress_bar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = train_accuracy.compute() * 100
        print(f'Epoch {epoch+1}/{config["epochs"]}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%')

# ==================================================
# Testing Loop
# ==================================================
def evaluate_model():
    model.eval()
    test_loss = 0.0
    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
    precision = Precision(task="multiclass", num_classes=10, average="macro").to(device)
    recall = Recall(task="multiclass", num_classes=10, average="macro").to(device)
    f1 = F1Score(task="multiclass", num_classes=10, average="macro").to(device)
    conf_matrix = ConfusionMatrix(task="multiclass", num_classes=10).to(device)

    # Add tqdm progress bar for evaluation
    progress_bar = tqdm(test_loader, desc="Evaluating")
    
    with torch.no_grad():
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            accuracy.update(preds, target)
            precision.update(preds, target)
            recall.update(preds, target)
            f1.update(preds, target)
            conf_matrix.update(preds, target)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

    test_loss /= len(test_loader.dataset)
    final_accuracy = accuracy.compute() * 100
    final_precision = precision.compute()
    final_recall = recall.compute()
    final_f1 = f1.compute()
    final_conf_matrix = conf_matrix.compute()

    return {
        "test_loss": test_loss,
        "accuracy": final_accuracy.item(),
        "precision": final_precision.item(),
        "recall": final_recall.item(),
        "f1_score": final_f1.item(),
        "confusion_matrix": final_conf_matrix,
    }

# ==================================================
# Run Training and Evaluation
# ==================================================
train_model()
print("\nEvaluating model on test data...")
metrics = evaluate_model()

print(f"\nTest Loss: {metrics['test_loss']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.2f}%")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")