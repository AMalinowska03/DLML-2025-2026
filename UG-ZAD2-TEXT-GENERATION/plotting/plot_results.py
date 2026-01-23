import pandas as pd
import matplotlib.pyplot as plt

FILE_NAME = 'plotting/transformer_2_v1_val_loss'

if __name__ == '__main__':
    # 1. Wczytanie danych
    file_path = f'{FILE_NAME}.csv'
    df = pd.read_csv(file_path)

    # 2. Funkcja wygładzająca
    def smooth_curve(points, factor=0.6):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    # 3. Zastosowanie mniejszego wygładzania (0.6 zamiast 0.9)
    smoothing_factor = 0.6
    df['Value_Smooth'] = smooth_curve(df['Value'], factor=smoothing_factor)

    # 4. Styl: Białe tło i "kwadratowe" proporcje
    plt.style.use('default')
    plt.figure(figsize=(8, 7)) # Zmiana z (12, 6) na (8, 7)

    plt.ylim(1, 7.1)

    # Surowe dane (jasnoszare)
    plt.plot(df['Step'], df['Value'], color='lightskyblue', alpha=0.5, label='Surowe dane')
    # Wygładzony trend
    plt.plot(df['Step'], df['Value_Smooth'], color='#1f77b4', linewidth=2, label='Trend')

    # Opisy
    plt.title('Przebieg funkcji straty (Loss)', fontsize=14)
    plt.xlabel('Krok (Step)', fontsize=11)
    plt.ylabel('Wartość (Value)', fontsize=11)

    # Kratka i legenda
    plt.grid(True, linestyle='--', color='gray', alpha=0.4) # Delikatna kratka
    plt.legend(frameon=True, facecolor='white', framealpha=1)

    plt.savefig(f'{FILE_NAME}.png', dpi=300, bbox_inches='tight')
    plt.show()