import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def load_data(file_path):
    """Загрузка данных из CSV."""
    data = pd.read_csv(file_path)
    data['ds'] = pd.to_datetime(data['ds'])
    return data

def calculate_metrics(y_true, y_pred):
    """Расчет метрик MAE, MAPE, RMSE."""
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, mape, rmse

def create_sequences(data, seq_length=24):
    """Создание последовательностей для LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def lstm_forecast(file_path, forecast_horizon=24, seq_length=24):
    """Прогнозирование с использованием LSTM."""
    # Загрузка данных
    data = load_data(file_path)
    
    # Разделение на тренировочную и тестовую выборки
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Нормализация данных
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data[['y']])
    test_scaled = scaler.transform(test_data[['y']])
    
    # Подготовка последовательностей
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)
    
    # Создание и обучение модели LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Прогноз
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)
    
    # Расчет метрик
    metrics = calculate_metrics(y_test_inv, y_pred)
    
    # Визуализация
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Реальные данные', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data['ds'][-len(y_pred):], y=y_pred.flatten(), name='Прогноз LSTM', line=dict(color='red')))
    fig.update_layout(title='Прогноз LSTM', xaxis_title='Дата', yaxis_title='Выработка/Потребление', showlegend=True)
    
    # Сохранение графика
    output_file = file_path.replace('.csv', '_lstm_forecast.html')
    fig.write_html(output_file)
    
    return metrics, output_file

