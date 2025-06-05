import pandas as pd
import numpy as np
import xgboost as xgb
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

def xgboost_forecast(file_path, forecast_horizon=24):
    """Прогнозирование с использованием XGBoost."""
    # Загрузка данных
    data = load_data(file_path)
    
    # Разделение на тренировочную и тестовую выборки
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Подготовка данных
    X_train = np.arange(len(train_data)).reshape(-1, 1)
    y_train = train_data['y']
    X_test = np.arange(len(train_data), len(data)).reshape(-1, 1)
    
    # Обучение модели XGBoost
    model = xgb.XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Прогноз
    y_pred = model.predict(X_test)
    
    # Расчет метрик
    metrics = calculate_metrics(test_data['y'], y_pred)
    
    # Визуализация
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Реальные данные', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data['ds'], y=y_pred, name='Прогноз XGBoost', line=dict(color='red')))
    fig.update_layout(title='Прогноз XGBoost', xaxis_title='Дата', yaxis_title='Выработка/Потребление', showlegend=True)
    
    # Сохранение графика
    output_file = file_path.replace('.csv', '_xgboost_forecast.html')
    fig.write_html(output_file)
    
    return metrics, output_file

