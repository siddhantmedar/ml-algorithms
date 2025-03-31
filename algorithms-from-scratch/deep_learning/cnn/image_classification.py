from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
    ]
)

config = {"batch_size": 64}
train_data = datasets.MNIST(
    "../mnist_dataset/train_data", train=True, download=True, transform=transform
)

test_data = datasets.MNIST(
    "../mnist_dataset/test_data", train=False, download=True, transform=transform
)

print(f"Train data shape: {train_data.data.shape}")
print(f"Test data shape: {test_data.data.shape}")

train_loader = DataLoader(
    train_data, batch_size=config["batch_size"], shuffle=True, drop_last=False
)


test_loader = DataLoader(
    test_data, batch_size=config["batch_size"], shuffle=True, drop_last=False
)

print(f"Train Loader size: {len(train_loader)}")
print(f"Test Loader size: {len(test_loader)}")
