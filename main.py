import CNN
import ChessDataset as CD
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN.CNN().to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    loss = nn.MSELoss()
    #205,353,531 is the limit
    data = CD.ChessDataset("Dataset/cleaned_dataset", 205353531)
    loader = DataLoader(data, batch_size=3000, shuffle=True, num_workers=1)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_id, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            difference = loss(pred, y)
            optimizer.zero_grad()
            difference.backward()
            optimizer.step()
            total_loss += difference.item()

            if batch_id % 1 == 0:
                avg_loss = total_loss / (batch_id + 1)
                print(f"Epoch {epoch + 1}, batch-id {batch_id}, avg loss {avg_loss}")
    torch.save(model.state_dict(), f"results.pth")

if __name__ == "__main__":
    main()