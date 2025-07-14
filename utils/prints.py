import matplotlib.pyplot as plt
import os


def log_epoch(epoch, loss, **metrics):
    msg = f"Epoch {epoch}: loss={loss:.4f}"
    for k, v in metrics.items():
        msg += f", {k}={v:.4f}"
    print(msg)


def plot_training_history(history, save_path, title):
    """Сохраняет график с историей обучения"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    ax.plot(history['train_losses'], label='Train Loss')
    ax.plot(history['test_losses'], label='Test Loss')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def plot_color_predictions(true_colors, pred_colors, epoch=None, save_path=None):
    """
    Визуализация предсказанных и истинных цветов в RGB пространстве.
    
    true_colors, pred_colors — numpy массивы формы (N, 3) с значениями в [0,1]
    epoch — номер эпохи (для заголовка)
    save_path — путь для сохранения графика (если None, график показывается)
    """
    plt.figure(figsize=(6,6))
    plt.title(f'Цвета: Истинные (круги) и Предсказанные (крестики)' + (f', Эпоха {epoch}' if epoch is not None else ''))
    
    # Истинные цвета — точки с цветом true_colors
    plt.scatter(true_colors[:,0], true_colors[:,1], color=true_colors, marker='o', label='Истинные цвета', edgecolor='k', s=100, alpha=0.7)
    # Предсказанные цвета — крестики с цветом pred_colors
    plt.scatter(pred_colors[:,0], pred_colors[:,1], color=pred_colors, marker='x', label='Предсказанные цвета', s=100, alpha=0.7)
    
    plt.xlabel('Red канал')
    plt.ylabel('Green канал')
    plt.legend()
    plt.grid(True)
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()