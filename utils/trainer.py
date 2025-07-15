import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from utils.model_utils import count_parameters
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('./results/logs.log')

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def train_model(model, train_loader, optimizer, criterion, test_loader, epochs: int = 10, lr: float = 0.001, device: str = 'cpu'):
    """
    Тренирует модель на train_loader и оценивает её на test_loader.
    """

    train_losses = []
    test_losses = []

    logger.info("Старт обучения модели на %d эпох(и)", epochs)
    logger.info("Параметров в модели: %d", count_parameters(model))

    for epoch in range(epochs):
        logger.info("=== Эпоха %d/%d ===", epoch + 1, epochs)

        try:
            train_loss = _run_epoch(model, train_loader, optimizer, criterion, is_test=False, device=device)
            logger.info("Train loss: %.4f", train_loss)
        except Exception as e:
            logger.error("Ошибка во время тренировки на эпохе %d: %s", epoch + 1, str(e))
            raise

        try:
            test_loss = _run_epoch(model, test_loader, optimizer, criterion, is_test=True, device=device)
            logger.info("Test loss: %.4f", test_loss)
        except Exception as e:
            logger.error("Ошибка во время валидации на эпохе %d: %s", epoch + 1, str(e))
            raise

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        logger.info('-' * 50)

    logger.info("Обучение завершено. Всего эпох: %d", epochs)
    logger.info("Финальное значение Test Loss: %.4f", test_losses[-1])

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
    }

def _run_epoch(model, data_loader, optimizer=None, criterion=None, is_test=False, device='cpu'):
    """
    Запускает одну эпоху обучения для модели.
    """
    import torch

    l1_lambda = 1e-5
    mode = 'Валидация' if is_test else 'Тренировка'
    logger.debug("[%s] Режим: %s", mode, ('eval' if is_test else 'train'))

    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    batches = len(data_loader)
    logger.debug("[%s] Всего батчей: %d", mode, batches)

    for i, (batch_X, batch_y) in enumerate(tqdm(data_loader)):
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
        if (i + 1) % 10 == 0 or (i + 1) == batches:
            logger.debug("[%s] Batch %d/%d: loss=%.4f", mode, i + 1, batches, loss.item())

    avg_loss = total_loss / batches
    logger.info("[%s] Средний Loss за эпоху: %.4f", mode, avg_loss)
    return avg_loss
