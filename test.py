from matplotlib import pyplot as plt
import numpy as np
import torch
from models.LinearModels import LR_ReLU
from utils.model_utils import hex_to_hsv
import matplotlib.colors as colors


def hsv_to_rgb_arr(hsv_arr):
    rgb = colors.hsv_to_rgb(hsv_arr)
    return (np.array(rgb) * 255).astype(np.uint8)

if __name__ == '__main__':
    model = LR_ReLU(in_features=3, hidden_size=16, out_features=3)
    state_dict = torch.load('./results/color_model_hsv.pth', map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.train()

    color = "#7000DF"
    input_hsv = hex_to_hsv(color)
    input_tensor = torch.tensor(input_hsv, dtype=torch.float32).unsqueeze(0) 

    with torch.no_grad():
        output_hsv = model(input_tensor).squeeze().cpu().numpy()
        output_hsv = np.clip(output_hsv, 0, 1)
        output_rgb = hsv_to_rgb_arr(output_hsv)

    input_rgb = hsv_to_rgb_arr(input_hsv)

    print("Входной HSV:", input_hsv)
    print("Сгенерированный HSV:", output_hsv)
    print("Входной RGB:", input_rgb)
    print("Сгенерированный моделью RGB:", output_rgb)

    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    axes[0].imshow(np.ones((10, 10, 3), dtype=np.uint8) * input_rgb)
    axes[0].set_title('Входной цвет')
    axes[0].axis('off')

    axes[1].imshow(np.ones((10, 10, 3), dtype=np.uint8) * output_rgb)
    axes[1].set_title('Сгенерированный')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('./plots/result_model.png')