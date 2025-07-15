import torch
import os


def hex_to_rgb(hex):
  """Переводит цвет из hex в rgb"""
  if hex[0] == '#':
    hex = hex[1:]

  rgb = []
  for i in (0, 2, 4):
    decimal = int(hex[i:i+2], 16)
    rgb.append(decimal)
  
  return tuple(rgb)


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(
        path: str, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        epoch: int
    ):
    """Сохраняет модель"""
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)