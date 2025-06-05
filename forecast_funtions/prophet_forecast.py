import pandas as pd
import numpy as np
from fbprophet import Prophet
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

def prophet_forecast(file_path, forecast_horizon=24):
    """Прогнозирование с использованием Prophet."""
    # Загрузка данных
    data = load_data(file_path)
    
    # Разделение на тренировочную и тестовую выборки
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Обучение модели Prophet
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(train_data)
    
    # Прогноз
    future = model.make_future_dataframe(periods=len(test_data), freq='H')
    forecast = model.predict(future)
    y_pred = forecast['yhat'][-len(test_data):]
    
    # Расчет метрик
    metrics = calculate_metrics(test_data['y'], y_pred)
    
    # Визуализация
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Реальные данные', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data['ds'], y=y_pred, name='Прогноз Prophet', line=dict(color='red')))
    fig.update_layout(title='Прогноз Prophet', xaxis_title='Дата', yaxis_title='Выработка/Потребление', showlegend=True)
    
    # Сохранение графика
    output_file = file_path.replace('.csv', '_prophet_forecast.html')
    fig.write_html(output_file)
    
    return metrics, output_file

