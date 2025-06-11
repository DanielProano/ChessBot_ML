import CNN
import torch
from FEN_conversion import fen_to_tensor
import json
import torch.nn as nn

model = CNN.CNN()
model.load_state_dict(torch.load("results_200mil_epoch10_mse50.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    with open("Dataset/test_dataset") as file:
        total_actual = []
        total_pred = []
        for line in file:
            data = json.loads(line)
            fen = data["fen"]
            cp = data["cp"]
            tensor = fen_to_tensor(fen).to(device)
            tensor = tensor.unsqueeze(0)
            cp_final = torch.tensor([data["cp"] / 100.0], dtype=torch.float32).to(device)
            pred = model(tensor)

            print(f"Prediction: {pred.item(): 0.2f}, Actual {cp_final.item(): 0.2f}")

            total_pred.append(pred.item())
            total_actual.append(cp_final.item())
        MSE = nn.MSELoss()
        final_prediction = torch.tensor(total_pred)
        final_actual = torch.tensor(total_actual)
        loss = MSE(final_prediction, final_actual)
        print(f"Congrats! You're error rate: {loss.item()}")