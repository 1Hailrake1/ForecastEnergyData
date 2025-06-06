import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import plotly.graph_objs as go

def load_data(file_path, start_date='2020-01-01 00:00:00', end_date='2025-06-05 23:59:00'):
    """Загрузка данных из CSV с ограничением по датам."""
    data = pd.read_csv(file_path)
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[(data['ds'] >= pd.to_datetime(start_date)) & (data['ds'] <= pd.to_datetime(end_date))]
    if data.empty:
        raise ValueError("После обрезки данных нет записей. Проверьте файл или даты.")
    return data

def create_features(data):
    """Создание дополнительных признаков для временного ряда с улучшенными лагами."""
    data = data.copy()
    data['hour'] = data['ds'].dt.hour
    data['day'] = data['ds'].dt.day
    data['month'] = data['ds'].dt.month
    data['year'] = data['ds'].dt.year
    data['dayofweek'] = data['ds'].dt.dayofweek

    # Расширенные лаги для улавливания пиков
    for lag in [24, 48, 72, 168, 336]:
        data[f'lag_{lag}'] = data['y'].shift(lag).astype(float)  # Приведение к float

    # Скользящее среднее с разными окнами
    data['rolling_mean_24'] = data['y'].rolling(window=24).mean().shift(1).astype(float)
    data['rolling_max_24'] = data['y'].rolling(window=24).max().shift(1).astype(float)

    # Удаление строк с NaN
    data = data.dropna()

    return data

def calculate_metrics(y_true, y_pred):
    """Расчет метрик с защитой от деления на малые значения в MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true > 1  # Порог 1 МВт для защиты от деления на слишком малые значения
    if np.sum(mask) == 0:
        mape = 0
    else:
        mape = np.mean(100 * np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, mape, rmse

def xgboost_forecast(file_path, forecast_horizon=4380):
    """Прогноз временного ряда с использованием XGBoost с улучшенной точностью на пиках."""
    # Загрузка данных
    data = load_data(file_path)

    # Создание признаков
    data = create_features(data)

    # Подготовка признаков и целевой переменной
    features = ['hour', 'day', 'month', 'year', 'dayofweek', 'lag_24', 'lag_48', 'lag_72', 'lag_168', 'lag_336', 'rolling_mean_24', 'rolling_max_24']
    X = data[features]
    y = data['y']

    # Приведение типов данных
    X = X.astype(float)

    # Деление на train/test (70% для обучения, 30% для теста)
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Весовые коэффициенты для акцента на пиках
    weights = np.ones(len(y_train))
    peak_threshold = y_train.quantile(0.95)  # Порог для пиковых значений (95-й процентиль)
    weights[y_train > peak_threshold] = 5.0  # Увеличение веса для пиков в 5 раз

    # Обучение XGBoost с оптимизированными параметрами
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=weights)

    # Прогноз на тесте
    y_pred_test = model.predict(X_test)
    y_pred_test = np.clip(y_pred_test, 0, None)

    # Подготовка данных для будущего прогноза с динамическим обновлением
    last_data = data.iloc[-1]
    future_dates = pd.date_range(start=data['ds'].iloc[-1] + pd.Timedelta(hours=1), periods=forecast_horizon, freq='h')
    future_data = pd.DataFrame(index=range(forecast_horizon), columns=features)
    future_data['hour'] = [d.hour for d in future_dates]
    future_data['day'] = [d.day for d in future_dates]
    future_data['month'] = [d.month for d in future_dates]
    future_data['year'] = [d.year for d in future_dates]
    future_data['dayofweek'] = [d.dayofweek for d in future_dates]

    # Динамическое обновление лагов и скользящего среднего
    future_y = np.concatenate([y.values, y_pred_test])  # Комбинируем реальные и предсказанные данные
    for i in range(forecast_horizon):
        for lag in [24, 48, 72, 168, 336]:
            if i + lag < len(future_y):
                future_data.iloc[i, future_data.columns.get_loc(f'lag_{lag}')] = float(future_y[i + lag - 1])  # Приведение к float
            else:
                future_data.iloc[i, future_data.columns.get_loc(f'lag_{lag}')] = float(last_data[f'lag_{lag}'])  # Используем последнее значение
        future_data.iloc[i, future_data.columns.get_loc('rolling_mean_24')] = float(np.mean(future_y[max(0, i-23):i+1]))
        future_data.iloc[i, future_data.columns.get_loc('rolling_max_24')] = float(np.max(future_y[max(0, i-23):i+1]))

    # Приведение типов данных для future_data
    future_data = future_data.astype(float)

    # Прогноз на будущее
    y_pred_future = model.predict(future_data)
    y_pred_future = np.clip(y_pred_future, 0, None)

    # Метрики
    mae, mape, rmse = calculate_metrics(y_test.values, y_pred_test)

    # График
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Реальные данные', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['ds'][train_size:], y=y_pred_test, name='Прогноз (тест)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=future_dates, y=y_pred_future, name='Прогноз (будущее)', line=dict(color='green', dash='dash')))
    fig.update_layout(
        title='Прогноз XGBoost (Производство энергии)',
        xaxis_title='Дата',
        yaxis_title='Производство (МВт)',
        showlegend=True,
        annotations=[
            dict(
                text=f"MAE={mae:.2f} | MAPE={mape:.2f}% | RMSE={rmse:.2f}",
                xref="paper", yref="paper", x=0, y=1.05,
                showarrow=False, font=dict(size=12)
            )
        ]
    )

    # Сохранение графика
    output_file = file_path.replace('.csv', '_xgboost_forecast.html')
    fig.write_html(output_file)

    return (mae, mape, rmse), output_file

if __name__ == "__main__":
    file_path = 'wind_forecast_clean.csv'
    metrics, output_file = xgboost_forecast(file_path)
    print(f"XGBoost Metrics: MAE={metrics[0]:.2f}, MAPE={metrics[1]:.2f}%, RMSE={metrics[2]:.2f}")
    print(f"График сохранен в: {output_file}")