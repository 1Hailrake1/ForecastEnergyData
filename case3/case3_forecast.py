from forecast_funtions import arima_forecast
from forecast_funtions import prophet_forecast
from forecast_funtions import lstm_forecast
from forecast_funtions import xgboost_forecast


def run_forecasts(file_path, case_name):
    """Запуск всех моделей прогнозирования для указанного CSV."""
    print(f"Обработка кейса: {case_name} ({file_path})")

    # SARIMA
    sarima_metrics, sarima_output = arima_forecast.sarima_forecast(file_path)
    print(f"SARIMA Metrics: MAE={sarima_metrics[0]:.2f}, MAPE={sarima_metrics[1]:.2f}%, RMSE={sarima_metrics[2]:.2f}")
    print(f"SARIMA График: {sarima_output}")

    # Prophet
    prophet_metrics, prophet_output = prophet_forecast.prophet_forecast(file_path)
    print(
        f"Prophet Metrics: MAE={prophet_metrics[0]:.2f}, MAPE={prophet_metrics[1]:.2f}%, RMSE={prophet_metrics[2]:.2f}")
    print(f"Prophet График: {prophet_output}")

    # LSTM
    lstm_metrics, lstm_output = lstm_forecast.lstm_forecast(file_path)
    print(f"LSTM Metrics: MAE={lstm_metrics[0]:.2f}, MAPE={lstm_metrics[1]:.2f}%, RMSE={lstm_metrics[2]:.2f}")
    print(f"LSTM График: {lstm_output}")

    # XGBoost
    xgboost_metrics, xgboost_output = xgboost_forecast.xgboost_forecast(file_path)
    print(
        f"XGBoost Metrics: MAE={xgboost_metrics[0]:.2f}, MAPE={xgboost_metrics[1]:.2f}%, RMSE={xgboost_metrics[2]:.2f}")
    print(f"XGBoost График: {xgboost_output}")


if __name__ == '__main__':
    # Кейс потребления электроэнергии
    cases = [
        {'file_path': '/energy_consumption.csv', 'case_name': 'Потребление электроэнергии в Германии'},
    ]

    for case in cases:
        run_forecasts(case['file_path'], case['case_name'])