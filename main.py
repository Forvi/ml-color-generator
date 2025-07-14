import torch
import torch.nn as nn
import torch.optim as optim
from models.LinearModels import LR_ReLU, LinearRegression
from torch.utils.data import DataLoader, random_split
from utils.datasets import ColorDataset
from utils.prints import plot_training_history
from utils.trainer import train_model

if __name__ == '__main__':
    PATH = './data/colors.csv'
    LR = 0.008
    EPOCHS = 10
    
    dataset = ColorDataset(PATH)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f'Размер train: {len(train_dataset)}, Размер test: {len(test_dataset)}')

    model = LR_ReLU(in_features=3)
    optimizer = optim.LBFGS(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    history = train_model(
        model=model, 
        test_loader=dataloader, 
        train_loader=train_loader, 
        optimizer=optimizer, 
        criterion=criterion, 
        lr=LR, 
        epochs=EPOCHS
    )
    
    plot_training_history(history, f'./plots/linear_reg_5.png', 'LinearRegression, Adam, LR=0.008, ReLU')