import json
import torch
from torch.utils.data import Dataset
from FEN_conversion import fen_to_tensor


class ChessDataset(Dataset):
    def __init__(self, filepath, limit=None):
        self.filepath = filepath
        self.offsets = []

        with open(filepath, "r") as f:
            offset = f.tell()
            count = 0
            while True:
                line = f.readline()
                if not line or (limit is not None and count >= limit):
                    break
                self.offsets.append(offset)
                offset = f.tell()
                count += 1

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with open(self.filepath, "r") as f:
            f.seek(self.offsets[index])
            line = f.readline()
            data = json.loads(line)
            fen = fen_to_tensor(data["fen"])
            cp = torch.tensor([data["cp"] / 100.0], dtype=torch.float32)
            return fen, cp
