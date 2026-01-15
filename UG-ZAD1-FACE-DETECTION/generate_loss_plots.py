import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(csv_path):
    metrics = pd.read_csv(csv_path)

    # Lightning loguje train i val w różnych krokach, trzeba je zagregować po epokach
    train_loss = metrics[['epoch', 'train_loss']].dropna()
    val_loss = metrics[['epoch', 'val_loss']].dropna()

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss['epoch'], train_loss['train_loss'], label='Train Loss')
    plt.plot(val_loss['epoch'], val_loss['val_loss'], label='Val Loss')
    plt.xlabel('Epoka')
    plt.ylabel('Loss')
    plt.title('Krzywa uczenia (Loss)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_chart.png')
    plt.show()

# do uruchomienia tam gdzie mamy metryki z logami
# plot_metrics("lightning_logs/gender_v1/metrics.csv")
# plot_metrics("lightning_logs/glasses_v2/metrics.csv")