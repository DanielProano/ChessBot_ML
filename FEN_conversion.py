import torch

def fen_to_tensor(fen):
    tensor_list = []
    pieces = ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P']
    parts = fen.split(" ")[0].split("/")
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
        tensor_list.append(torch.tensor(matrix, dtype=torch.float32))
    return torch.stack(tensor_list)