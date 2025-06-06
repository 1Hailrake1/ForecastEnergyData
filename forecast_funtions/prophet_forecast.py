import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objs as go

def load_data(file_path, start_date='2020-01-01 00:00:00', end_date='2025-06-05 23:59:00'):
    """Загрузка данных из CSV с ограничением по датам."""
    data = pd.read_csv(file_path)
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[(data['ds'] >= pd.to_datetime(start_date)) & (data['ds'] <= pd.to_datetime(end_date))]
    if data.empty:
        raise ValueError("После обрезки данных нет записей. Проверьте файл или даты.")
    return data

def preprocess_data(data):
    """Предобработка данных: интерполяция нулевых значений и мягкая обработка выбросов."""
    data['y'] = data['y'].replace(0, np.nan).interpolate(method='linear')
    mean = data['y'].mean()
    std = data['y'].std()
    # Мягкая обрезка выбросов (±5 стандартных отклонений вместо 3)
    data['y'] = data['y'].clip(mean - 5 * std, mean + 5 * std)
    print(f"Min y after preprocessing: {data['y'].min()}, Max y: {data['y'].max()}")
    return data

def calculate_metrics(y_true, y_pred):
    """Расчет метрик с защитой от деления на малые значения в MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true > 1  # Порог 1 МВт для защиты от деления на слишком малые значения
    if np.sum(mask) == 0:
        mape = 0  # Если все значения меньше порога, MAPE = 0
    else:
        mape = np.mean(100 * np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, mape, rmse

def prophet_forecast(file_path, forecast_horizon=4380):
    """Максимально точный прогноз временного ряда с использованием Prophet."""
    # Загрузка и предобработка данных
    data = load_data(file_path)
    data = preprocess_data(data)

    # Подготовка данных для Prophet
    df = data[['ds', 'y']].copy()

    # Логарифмическое преобразование для стабилизации дисперсии
    df['y'] = np.log1p(df['y'])

    # Деление на train/test (вернем 70% для обучения)
    train_size = int(len(df) * 0.7)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    # Обучение Prophet с оптимизированными параметрами
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,  # Вернем 10.0 для лучшей сезонности
        seasonality_mode='multiplicative'
    )
    model.fit(train_data)

    # Прогноз на тест
    future_test = model.make_future_dataframe(periods=len(test_data), freq='h')
    forecast_test = model.predict(future_test)
    y_pred_test_log = forecast_test['yhat'].iloc[train_size:].values
    y_pred_test = np.expm1(y_pred_test_log)  # Обратное преобразование
    y_pred_test = np.clip(y_pred_test, 0, None)

    # Прогноз на будущее
    future = model.make_future_dataframe(periods=len(test_data) + forecast_horizon, freq='h')
    forecast_future = model.predict(future)
    y_pred_future_log = forecast_future['yhat'].iloc[-forecast_horizon:].values
    y_pred_future = np.expm1(y_pred_future_log)  # Обратное преобразование
    y_pred_future = np.clip(y_pred_future, 0, None)

    # Даты для будущего прогноза
    future_dates = pd.date_range(start=data['ds'].iloc[-1] + pd.Timedelta(hours=1), periods=forecast_horizon, freq='h')

    # Метрики
    y_test_true = np.expm1(test_data['y'].values)  # Обратное преобразование тестовых данных
    mae, mape, rmse = calculate_metrics(y_test_true, y_pred_test)

    # График
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=np.expm1(df['y']), name='Реальные данные', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data['ds'], y=y_pred_test, name='Прогноз (тест)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=future_dates, y=y_pred_future, name='Прогноз (будущее)', line=dict(color='green', dash='dash')))
    fig.update_layout(
        title='Прогноз Prophet (Потребление электроэнергии)',
        xaxis_title='Дата',
        yaxis_title='Потребление (МВт)',
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
    output_file = file_path.replace('.csv', '_prophet_forecast.html')
    fig.write_html(output_file)

    return (mae, mape, rmse), output_file

if __name__ == "__main__":
    file_path = 'energy_consumption.csv'
    metrics, output_file = prophet_forecast(file_path)
    print(f"Prophet Metrics: MAE={metrics[0]:.2f}, MAPE={metrics[1]:.2f}%, RMSE={metrics[2]:.2f}")
    print(f"График сохранен в: {output_file}")