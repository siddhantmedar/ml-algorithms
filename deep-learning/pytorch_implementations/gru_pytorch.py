import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


generator = torch.Generator().manual_seed(313)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device found:", device)

class MyDataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index],self.y[index]


class GRUModel(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim,hidden_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim,hidden_dim)

    def forward(self,x):
        output,hn = self.gru(x)
        return self.fc(output)


if __name__ == "__main__":
    n_samples = 1524
    seq_length = 10
    batch_size = 16

    input_dim = 3
    hidden_dim = 5

    learning_rate = 0.001
    epochs = 20
    
    X = torch.randn(n_samples, seq_length, input_dim)
    y = torch.randint(low=0, high=2, size=(X.shape[0], X.shape[1], hidden_dim)).float()

    print("X shape:", X.shape)
    print("y_true shape:", y.shape)

    dataset = MyDataset(X,y)

    train_size = int(0.90*len(dataset))
    test_size = int(len(dataset)-train_size)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0,
                            drop_last=False)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        drop_last=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Train dataloader batches: {len(train_dataloader)}")
    print(f"Test dataloader batches: {len(test_dataloader)}")

    model = GRUModel(input_dim,hidden_dim)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

    epoch_losses = []

    for epoch in range(epochs):
        model.train()

        batch_losses = []

        for batch_index,(batch_data,batch_target) in enumerate(train_dataloader):
            optimizer.zero_grad()

            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            print(f"Batch {batch_index}: Input batch shape: {batch_data.shape}, Target batch shape: {batch_target.shape}, Device: {batch_data.device}")

            batch_pred = model(batch_data)
            loss = criterion(batch_pred,batch_target)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_losses.append(torch.mean(torch.tensor(batch_losses)))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_losses[-1]:.4f}")

    # plot the loss
    plt.plot(epoch_losses)
    plt.title('Training Loss (Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'./plots/training_loss_{epochs}_epochs.png')  # Save the plot to a relative path
    plt.show()

    # inference - to be implemented
    with torch.no_grad():
        model.eval()

        test_losses = []

        for batch_index,(batch_data,batch_target) in enumerate(test_dataloader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            print(f"Batch {batch_index}: Input batch shape: {batch_data.shape}, Target batch shape: {batch_target.shape}, Device: {batch_data.device}")

            batch_pred = model(batch_data)
            loss = criterion(batch_pred,batch_target)

            test_losses.append(loss.item())
                
        print(f"Test Loss: {torch.mean(torch.tensor(test_losses))}")

 