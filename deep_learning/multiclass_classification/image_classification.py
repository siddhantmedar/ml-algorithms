# ==================================================
# Imports
# ==================================================
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import wandb

# ==================================================
# Configurations
# ==================================================
config = {
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 0.001,
    "architecture": "MLP",
    "dataset": "MNIST",
}

# ==================================================
# Load Dataset / Apply Transformations
# ==================================================
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

# ==================================================
# Model Architecture
# ==================================================
class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        return self.fc3(out)

# ==================================================
# Training Loop
# ==================================================
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model = ClassificationModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

def train_model():
    model.train()
    train_accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
    for epoch in range(config["epochs"]):
        running_loss = 0.0
        train_accuracy.reset()  # Reset accuracy per epoch
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
            preds = torch.argmax(output, dim=1)
            train_accuracy.update(preds, target)
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = train_accuracy.compute() * 100
        wandb.log({"train_loss": epoch_loss, "train_accuracy": epoch_acc, "epoch": epoch + 1})
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

    with torch.no_grad():
        for data, target in test_loader:
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

    test_loss /= len(test_loader.dataset)
    final_accuracy = accuracy.compute() * 100
    final_precision = precision.compute()
    final_recall = recall.compute()
    final_f1 = f1.compute()
    final_conf_matrix = conf_matrix.compute()

    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": final_accuracy.item(),
        "test_precision": final_precision.item(),
        "test_recall": final_recall.item(),
        "test_f1_score": final_f1.item(),
    })

    conf_matrix_np = final_conf_matrix.cpu().numpy()
    class_names = [str(i) for i in range(10)]
    wandb.log({
        "confusion_matrix": wandb.Table(
            columns=class_names,
            rows=class_names,
            data=conf_matrix_np.tolist()
        )
    })

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
wandb.init(
    project="mnist-mlp",
    config=config,
    name=f"mlp_run_{wandb.util.generate_id()}"
)

train_model()
metrics = evaluate_model()

model_path = f'model_{wandb.run.id}.pth'
torch.save(model.state_dict(), model_path)
wandb.save(model_path)

print(f"Test Loss: {metrics['test_loss']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.2f}%")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

wandb.finish()