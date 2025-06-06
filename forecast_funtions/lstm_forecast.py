import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plotly.graph_objs as go
import time
from typing import Tuple, List

def load_and_preprocess_data(file_path: str, start_date: str = '2020-01-01', end_date: str = '2025-06-05') -> Tuple[
    pd.DataFrame, np.ndarray, MinMaxScaler, List[str]]:
    """Загружает и предварительно обрабатывает данные с добавлением циклических признаков."""
    data = pd.read_csv(file_path, parse_dates=['ds'])
    data = data[(data['ds'] >= pd.to_datetime(start_date)) & (data['ds'] <= pd.to_datetime(end_date))]
    if data.empty:
        raise ValueError("Нет данных в указанном диапазоне дат.")

    data.set_index('ds', inplace=True)
    data['y'] = data['y'].replace(0, np.nan).interpolate(method='time')
    data['y'].fillna(method='bfill', inplace=True)
    data['y'].fillna(method='ffill', inplace=True)

    # Добавление временных и циклических признаков
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
    data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    data['y_scaled'] = scaler_y.fit_transform(data['y'].values.reshape(-1, 1))

    feature_cols = ['hour', 'day_of_week', 'month', 'sin_hour', 'cos_hour', 'sin_month', 'cos_month']
    for col in feature_cols:
        data[col] = data[col] / data[col].max()  # Нормализация

    scaled_features = data[['y_scaled'] + feature_cols].values

    return data, scaled_features, scaler_y, feature_cols

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Создание последовательностей."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def calculate_metrics(y_true_inv: np.ndarray, y_pred_inv: np.ndarray) -> Tuple[float, float, float]:
    """Расчет метрик с защитой от деления на малые значения в MAPE."""
    y_true_inv = np.clip(y_true_inv, 0, None)
    y_pred_inv = np.clip(y_pred_inv, 0, None)
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    mask = y_true_inv > 1
    mape = np.mean(np.abs((y_true_inv[mask] - y_pred_inv[mask]) / y_true_inv[mask])) * 100 if np.sum(mask) > 0 else 0
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    return mae, mape, rmse

def build_and_train_fast_model(X_train: np.ndarray, y_train: np.ndarray) -> Sequential:
    """Создание, компиляция и быстрое обучение LSTM модели с улучшенной архитектурой."""
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),  # Второй слой для лучшего моделирования зависимостей
        Dropout(0.2),
        Dense(1, activation='relu')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]

    print("\n--- Начало быстрого обучения ---")
    start_time = time.time()

    model.fit(
        X_train, y_train,
        epochs=20,  # Уменьшено для ускорения
        batch_size=128,  # Увеличено для ускорения
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
        shuffle=False
    )

    end_time = time.time()
    print(f"--- Обучение завершено за {end_time - start_time:.2f} секунд ---")
    return model

def forecast_future(model: Sequential, initial_sequence: np.ndarray, horizon: int, scaler_y: MinMaxScaler,
                    feature_cols: List[str], last_date: pd.Timestamp) -> np.ndarray:
    """Прогноз на будущее с динамическим обновлением временных признаков."""
    future_predictions_scaled = []
    current_sequence = initial_sequence.copy()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=horizon, freq='h')

    for future_date in future_dates:
        predicted_scaled_value = model.predict(current_sequence[np.newaxis, :, :], verbose=0)[0, 0]
        future_predictions_scaled.append(predicted_scaled_value)

        # Обновление временных признаков для следующего шага
        new_features = np.array([
            future_date.hour / 23.0,  # Нормализация
            future_date.dayofweek / 6.0,
            future_date.month / 12.0,
            np.sin(2 * np.pi * future_date.hour / 24),  # Циклические признаки
            np.cos(2 * np.pi * future_date.hour / 24),
            np.sin(2 * np.pi * future_date.month / 12),
            np.cos(2 * np.pi * future_date.month / 12)
        ])
        next_step_features = np.insert(new_features, 0, predicted_scaled_value)
        current_sequence = np.vstack([current_sequence[1:], next_step_features])

    future_predictions = scaler_y.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    return np.clip(future_predictions, 0, None)

def plot_results(data: pd.DataFrame, y_test_inv: np.ndarray, y_pred_test_inv: np.ndarray,
                 future_dates: pd.DatetimeIndex, y_pred_future: np.ndarray, test_dates_start_index: int,
                 mae: float, mape: float, rmse: float, file_path: str) -> str:
    """Визуализация результатов в стиле XGBoost."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['y'], name='Реальные данные', line=dict(color='blue')))
    test_dates = data.index[test_dates_start_index:test_dates_start_index + len(y_test_inv)]
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred_test_inv.flatten(), name='Прогноз (тест)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=future_dates, y=y_pred_future.flatten(), name='Прогноз (будущее)', line=dict(color='green', dash='dash')))
    fig.update_layout(
        title='Прогноз LSTM (Выработка ветроэнергии)',
        xaxis_title='Дата',
        yaxis_title='Выработка (МВт)',
        showlegend=True,
        annotations=[
            dict(
                text=f"MAE={mae:.2f} | MAPE={mape:.2f}% | RMSE={rmse:.2f}",
                xref="paper", yref="paper", x=0, y=1.05,
                showarrow=False, font=dict(size=12)
            )
        ]
    )

    output_file = file_path.replace('.csv', '_lstm_forecast.html')
    fig.write_html(output_file)
    return output_file

def main(file_path: str, forecast_horizon: int = 4380):
    seq_length = 168  # Увеличено до 1 недели

    # Загрузка и предобработка
    data, scaled_features, scaler_y, feature_cols = load_and_preprocess_data(file_path)

    # Создание последовательностей
    X, y = create_sequences(scaled_features, seq_length)

    # Разделение на train/test (85% для обучения, 15% для теста)
    train_size = int(len(X) * 0.85)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Длина последовательности: {seq_length} шагов ({seq_length / 24:.1f} дня)")
    print(f"Размер обучающей выборки: {len(X_train)}, тестовой: {len(X_test)}")

    # Обучение модели
    model = build_and_train_fast_model(X_train, y_train)

    # Прогноз на тесте
    y_pred_test_scaled = model.predict(X_test, verbose=0)
    y_pred_test_inv = scaler_y.inverse_transform(y_pred_test_scaled)
    y_pred_test_inv = np.clip(y_pred_test_inv, 0, None)
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # Метрики
    mae, mape, rmse = calculate_metrics(y_test_inv.flatten(), y_pred_test_inv.flatten())
    print(f"\nМетрики на тестовых данных: MAE={mae:.2f}, MAPE={mape:.2f}%, RMSE={rmse:.2f}")

    # Прогноз на будущее
    last_sequence = scaled_features[-seq_length:]
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=forecast_horizon, freq='h')
    y_pred_future = forecast_future(model, last_sequence, forecast_horizon, scaler_y, feature_cols, last_date)

    # Визуализация
    test_dates_start_index = train_size + seq_length
    output_file = plot_results(data, y_test_inv, y_pred_test_inv, future_dates, y_pred_future,
                               test_dates_start_index, mae, mape, rmse, file_path)
    print(f"График сохранен в: {output_file}")

    return (mae, mape, rmse), output_file

if __name__ == "__main__":
    file_path = 'wind_forecast_clean.csv'  # Обновлено для ветровой энергии
    try:
        metrics, output_file = main(file_path)
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути '{file_path}'. Убедитесь, что файл существует.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")