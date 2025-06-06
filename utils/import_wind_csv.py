import sqlite3
import pandas as pd

def import_wind_forecast_csv(csv_path, db_path='../load_data.db'):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract start of time interval and convert to datetime
    df['ds'] = df['MTU (CET/CEST)'].str.extract(r'^(.+?)\s-\s')[0]
    df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)
    df['ds'] = df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Select wind forecast columns (onshore and offshore)
    onshore_col = "Generation - Wind Onshore [MW] Day Ahead/ BZN|NL"
    offshore_col = "Generation - Wind Offshore [MW] Day Ahead/ BZN|NL"

    # Filter out non-numeric values and convert to float
    df = df[df[onshore_col] != "n/e"]
    df = df[df[offshore_col] != "n/e"]
    df[onshore_col] = pd.to_numeric(df[onshore_col], errors='coerce')
    df[offshore_col] = pd.to_numeric(df[offshore_col], errors='coerce')

    # Drop rows where wind values are NaN
    df = df.dropna(subset=[onshore_col, offshore_col])

    # Calculate total wind generation
    df['total_wind_mw'] = df[onshore_col] + df[offshore_col]

    # Prepare final dataframe
    df_final = df[['ds', 'total_wind_mw']].rename(columns={'total_wind_mw': 'y'})

    # Connect to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if not exists (Netherlands Wind Generation)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS netherlands_wind_generation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            value_mw REAL NOT NULL
        )
    ''')

    # Check for duplicates
    cursor.execute("SELECT datetime FROM netherlands_wind_generation")
    existing_dates = {row[0] for row in cursor.fetchall()}
    new_records = 0

    # Insert data
    for _, row in df_final.iterrows():
        if pd.notna(row['y']) and row['ds'] not in existing_dates:
            cursor.execute('INSERT INTO netherlands_wind_generation (datetime, value_mw) VALUES (?, ?)',
                           (row['ds'], float(row['y'])))
            new_records += 1
            existing_dates.add(row['ds'])

    conn.commit()
    conn.close()
    print(f"✅ Данные успешно загружены в таблицу netherlands_wind_generation. Добавлено {new_records} новых записей.")

def select_wind_forecast():
    conn = sqlite3.connect('../load_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='netherlands_wind_generation'")
    if cursor.fetchone() is None:
        print("Таблица 'netherlands_wind_generation' не найдена.")
        conn.close()
        return
    cursor.execute("SELECT * FROM netherlands_wind_generation ORDER BY datetime")
    rows = cursor.fetchall()
    print("ID | Дата и время         | Прогноз ветра (MW)")
    print("---|----------------------|----------------------")
    for row in rows:
        print(f"{row[0]:>2} | {row[1]} | {row[2]:.2f}")
    conn.close()

def export_wind_to_csv(db_path, output_csv, aggregate_to_hourly=False):
    """Экспорт данных о ветре в CSV."""
    conn = sqlite3.connect(db_path)
    query = "SELECT datetime, value_mw FROM netherlands_wind_generation ORDER BY datetime"
    df = pd.read_sql_query(query, conn)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Validate data range
    if df.empty:
        print("❌ Нет данных для экспорта.")
        conn.close()
        return None
    print(f"Диапазон данных: {df['datetime'].min()} - {df['datetime'].max()}")

    df = df.rename(columns={'datetime': 'ds', 'value_mw': 'y'})

    if aggregate_to_hourly:
        df = df.set_index('ds').resample('H').sum().reset_index()  # Sum total wind generation
        df = df.dropna()

    df.to_csv(output_csv, index=False)
    print(f"✅ Ветроданные экспортированы в {output_csv}")
    conn.close()
    return df

# Пример использования
if __name__ == "__main__":
    #import_wind_forecast_csv('Generation Forecasts for Wind and Solar_202501010000-202601010000.csv')
    #select_wind_forecast()
    export_wind_to_csv('../load_data.db', 'wind_forecast_clean.csv', aggregate_to_hourly=True)