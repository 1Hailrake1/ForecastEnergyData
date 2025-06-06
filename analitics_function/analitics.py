import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import plotly.graph_objs as go
import os

def load_data(file_path, end_date='2025-06-05 17:12:00'):
    """Загрузка данных из CSV с ограничением до текущей даты (17:12 PM CEST, 05.06.2025)."""
    data = pd.read_csv(file_path)
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds'] <= pd.to_datetime(end_date)]
    return data

def calculate_statistics(data):
    """Расчет статистических показателей."""
    y = data['y']

    stats = {
        'mean': y.mean(),
        'variance': y.var(),
        'std': y.std(),
        'median': y.median(),
        'min': y.min(),
        'max': y.max(),
        'q25': y.quantile(0.25),
        'q75': y.quantile(0.75),
        'coeff_variation': y.std() / y.mean() * 100,  # Коэффициент вариации в процентах
        'skewness': skew(y),
        'kurtosis': kurtosis(y)
    }
    return stats

def save_statistics(stats, output_path):
    """Сохранение статистики в текстовый файл."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Статистические показатели данных:\n")
        f.write(f"Математическое ожидание (среднее): {stats['mean']:.2f} МВт\n")
        f.write(f"Дисперсия: {stats['variance']:.2f} (МВт²)\n")
        f.write(f"Стандартное отклонение: {stats['std']:.2f} МВт\n")
        f.write(f"Медиана: {stats['median']:.2f} МВт\n")
        f.write(f"Минимум: {stats['min']:.2f} МВт\n")
        f.write(f"Максимум: {stats['max']:.2f} МВт\n")
        f.write(f"25-й квантиль: {stats['q25']:.2f} МВт\n")
        f.write(f"75-й квантиль: {stats['q75']:.2f} МВт\n")
        f.write(f"Коэффициент вариации: {stats['coeff_variation']:.2f}%\n")
        f.write(f"Асимметрия (skewness): {stats['skewness']:.2f}\n")
        f.write(f"Эксцесс (kurtosis): {stats['kurtosis']:.2f}\n")

def plot_data(data, output_path):
    """Построение графика временного ряда."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Выработка ветроэнергии', line=dict(color='green')))
    fig.update_layout(
        title='Выработка ветроэнергии в Нидерландах (2020–2025)',
        xaxis_title='Дата',
        yaxis_title='Выработка (МВт)',
        showlegend=True
    )
    fig.write_html(output_path)

def analyze_data(file_path):
    """Основная функция для анализа данных."""
    # Загрузка данных
    data = load_data(file_path)

    # Расчет статистики
    stats = calculate_statistics(data)

    # Вывод статистики в консоль
    print("Статистические показатели данных:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")

    # Сохранение статистики
    stats_output_path = os.path.join('case3', 'data_stats.txt')
    os.makedirs('case3', exist_ok=True)
    save_statistics(stats, stats_output_path)
    print(f"\nСтатистика сохранена в: {stats_output_path}")

    # Построение графика
    plot_output_path = os.path.join('case3', 'data_plot.html')
    plot_data(data, plot_output_path)
    print(f"График сохранен в: {plot_output_path}")

if __name__ == '__main__':
    file_path = 'wind_forecast_clean.csv'
    analyze_data(file_path)