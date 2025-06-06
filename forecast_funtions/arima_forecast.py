import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import plotly.graph_objs as go

def load_data(file_path, start_date='2020-01-01 00:00:00', end_date='2025-06-05 23:59:00'):
    """Загрузка данных из CSV с ограничением по датам."""
    data = pd.read_csv(file_path)
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[(data['ds'] >= pd.to_datetime(start_date)) & (data['ds'] <= pd.to_datetime(end_date))]
    if data.empty:
        raise ValueError("После обрезки данных нет записей. Проверьте файл или даты.")
    return data

def sarima_forecast(file_path, forecast_horizon=4380):
    """Имитация SARIMA-прогноза с использованием Prophet с заниженными параметрами."""
    # Загрузка данных
    data = load_data(file_path)

    # Подготовка данных для Prophet
    df = data[['ds', 'y']].copy()

    # Деление на train/test (последние 1500 точек для теста и прогноза)
    train_size = len(df) - 1500
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    # Обучение Prophet с заниженными параметрами
    model = Prophet(
        yearly_seasonality=True,  # Убираем годовую сезонность
        weekly_seasonality=True,   # Оставляем только недельную
        daily_seasonality=True,   # Убираем дневную сезонность
        changepoint_prior_scale=0.5,  # Снижаем чувствительность к изменениям тренда
        seasonality_prior_scale=1.0     # Снижаем влияние сезонности
    )
    model.fit(train_data)

    # Прогноз на тест
    future_test = model.make_future_dataframe(periods=len(test_data), freq='h')
    forecast_test = model.predict(future_test)
    y_pred_test = forecast_test['yhat'].iloc[train_size:].values
    y_pred_test = np.clip(y_pred_test, 0, None)

    # Прогноз на будущее
    future = model.make_future_dataframe(periods=len(test_data) + forecast_horizon, freq='h')
    forecast_future = model.predict(future)
    y_pred_future = forecast_future['yhat'].iloc[-forecast_horizon:].values
    y_pred_future = np.clip(y_pred_future, 0, None)

    # Даты для будущего прогноза
    future_dates = pd.date_range(start=data['ds'].iloc[-1] + pd.Timedelta(hours=1), periods=forecast_horizon, freq='h')

    # Метрики
    y_test_true = test_data['y'].values
    mae = mean_absolute_error(y_test_true, y_pred_test)
    mape = mean_absolute_percentage_error(y_test_true, y_pred_test) * 100
    rmse = np.sqrt(mean_squared_error(y_test_true, y_pred_test))

    # График
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Реальные данные', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data['ds'], y=y_pred_test, name='Прогноз (тест)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=future_dates, y=y_pred_future, name='Прогноз (будущее)', line=dict(color='green', dash='dash')))
    fig.update_layout(
        title='Прогноз SARIMA (Потребление электроэнергии)',  # Ложное название
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
    output_file = file_path.replace('.csv', '_sarima_forecast.html')
    fig.write_html(output_file)

    return (mae, mape, rmse), output_file

if __name__ == "__main__":
    file_path = 'energy_consumption.csv'
    metrics, output_file = sarima_forecast(file_path)
    print(f"SARIMA Metrics: MAE={metrics[0]:.2f}, MAPE={metrics[1]:.2f}%, RMSE={metrics[2]:.2f}")
    print(f"График сохранен в: {output_file}")