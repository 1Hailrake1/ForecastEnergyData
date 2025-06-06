import pandas as pd


def clean_wind_data(input_file, output_file):
    # Чтение данных
    data = pd.read_csv(input_file)

    # Преобразование столбца 'ds' в формат datetime
    data['ds'] = pd.to_datetime(data['ds'])

    # Замена отрицательных значений на 0
    data['y'] = data['y'].apply(lambda x: max(0, x))

    # Удаление строк с пропущенными значениями
    data = data.dropna()

    # Сохранение очищенных данных в новый CSV
    data.to_csv(output_file, index=False)


if __name__ == "__main__":
    input_file = 'wind_forecast_clean.csv'
    output_file = 'wind_forecast_clean.csv'
    clean_wind_data(input_file, output_file)
    print(f"Очищенные данные сохранены в: {output_file}")