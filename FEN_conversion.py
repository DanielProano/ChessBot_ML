import torch
import json
import zstandard
from io import TextIOWrapper

def fen_to_tensor(fen):
    tensor_list = []
    pieces = ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P']
    parts = fen.split("/")
    index = 0
    temp = ""
    while parts[7][index] != ' ':
        temp += parts[7][index]
        index += 1
    parts[7] = temp
    for i in range(12):
        matrix = []
        for rows in range(8):
            row = []
            for square in range(8):
                if square >= len(parts[rows]):
                    row.append(0)
                elif parts[rows][square] == pieces[i]:
                    row.append(1)
                else:
                    row.append(0)
            matrix.append(row)
        tensor_list.append(torch.tensor(matrix))
    print(tensor_list)
    return torch.stack(tensor_list)

def read_file(file, limit):
    count = 0
    with open(file, "rb") as file:
        zst = zstandard.ZstdDecompressor()
        with zst.stream_reader(file) as reader:
            text = TextIOWrapper(reader)
            for obj in text:
                if count == limit:
                    break
                try:
                    py_obj = json.loads(obj)
                    tensor = fen_to_tensor(py_obj["fen"])
                    print(tensor)
                    print(py_obj["evals"])
                except json.JSONDecodeError:
                    pass
                count += 1

read_file("Dataset/lichess_evas.zst", 2)