from matplotlib import pyplot as plt
import numpy as np
import torch
from models.LinearModels import LR_ReLU

model = LR_ReLU(in_features=3, hidden_size=16, out_features=3)

state_dict = torch.load('./results/color_model.pth', map_location='cpu')
model.load_state_dict(state_dict['model'])
model.eval()

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0

color = "#9A918B" 
input_rgb = hex_to_rgb(color)
input_tensor = torch.tensor(input_rgb, dtype=torch.float32).unsqueeze(0) 

with torch.no_grad():
    output_tensor = model(input_tensor)
    output_rgb = (output_tensor.squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)

print("Исходный цвет RGB:", (input_rgb * 255).astype(int))
print("Сгенерированный моделью RGB:", output_rgb)

fig, axes = plt.subplots(1, 2, figsize=(4, 2))
axes[0].imshow(np.ones((10, 10, 3), dtype=np.uint8) * (input_rgb * 255).astype(np.uint8))
axes[0].set_title('Входной цвет')
axes[0].axis('off')

axes[1].imshow(np.ones((10, 10, 3), dtype=np.uint8) * output_rgb)
axes[1].set_title('Сгенерированный')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('./plots/result_model.png')