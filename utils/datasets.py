from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.model_utils import hex_to_rgb
import torch

class ColorDataset(Dataset):
    def __init__(self, path_to_file: str):
        self.data = self._read_file(path_to_file)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.data.iloc[index]
        input_rgb = row['input_color']
        output_rgb = row['output_color']

        # нормализация и преобразование в tensor
        input_tensor = torch.tensor(input_rgb, dtype=torch.float32) / 255.0
        output_tensor = torch.tensor(output_rgb, dtype=torch.float32) / 255.0

        return input_tensor, output_tensor
    
    def get_df(self):
        return self.data

    def _read_file(self, path_to_file: str) -> pd.DataFrame:
        df = pd.read_csv(path_to_file)
        
        for row in df.columns:
            df[row] = df[row].apply(hex_to_rgb)

        return df
    

# dataset = ColorDataset('./data/colors.csv')
# print(dataset.__getitem__(2))