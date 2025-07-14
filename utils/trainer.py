import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from utils.model_utils import save_model


def train_model(model, train_loader, optimizer, criterion, test_loader, epochs:int = 10, lr: float = 0.001, device:str = 'cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, optimizer, criterion, is_test=False)
        test_loss = _run_epoch(model, test_loader, optimizer, criterion, is_test=True)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}')
        print('-' * 50)
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
    } 


def _run_epoch(model, data_loader, optimizer=None, criterion=None, is_test=False, device='cpu'):
    l1_lambda = 1e-5 
    
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    for batch_X, batch_y in tqdm(data_loader):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device) 

        if is_test:
            with torch.no_grad():
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)
        else:
            def closure():
                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)
                
                l1_norm = 0
                for param in model.parameters():
                    l1_norm += torch.norm(param, 1)
                loss = loss + l1_lambda * l1_norm

                loss.backward()
                return loss

            loss = optimizer.step(closure)

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

