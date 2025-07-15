from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.model_utils import hex_to_hsv
import torch
import logging

logger = logging.getLogger(__name__)

class ColorDataset(Dataset):
    """
    Датасет для загрузки и предобработки цветов из CSV-файла.

    Каждый элемент датасета представляет пару: входной и целевой цвет в формате HSV.
    Используется для обучения и тестирования моделей генерации цветов.

    Args:
        path_to_file (str): путь к CSV-файлу с цветами, цвета записаны как шестнадцатеричные строки (#RRGGBB).
    
    Attributes:
        data (pd.DataFrame): хранит всё загруженное и преобразованное содержимое.
    """
    def __init__(self, path_to_file: str):
        logger.info("Загрузка датасета из файла: %s", path_to_file)
        self.data = self._read_file(path_to_file)
        logger.info("Датасет загружен, всего %d элементов.", len(self.data))

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        logger.debug("Запрошен элемент по индексу: %d", index)
        row = self.data.iloc[index]
        input_rgb = row['input_color']
        output_rgb = row['output_color']
        input_tensor = torch.tensor(input_rgb, dtype=torch.float32)
        output_tensor = torch.tensor(output_rgb, dtype=torch.float32)
        return input_tensor, output_tensor
    
    def get_df(self):
        return self.data

    def _read_file(self, path_to_file: str) -> pd.DataFrame:
        logger.debug("Чтение CSV-файла: %s", path_to_file)
        df = pd.read_csv(path_to_file)
        logger.info("Файл успешно прочитан: %d строк.", len(df))
        for row in df.columns:
            df[row] = df[row].apply(hex_to_hsv)
        logger.info("Цвета переведены в HSV.")
        return df
