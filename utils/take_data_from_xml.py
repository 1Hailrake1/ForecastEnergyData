import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import sqlite3
import pandas as pd

def parse_xml():
    tree = ET.parse('ACTUAL_TOTAL_LOAD_202501010000-202601010000.xml')
    root = tree.getroot()
    ns = {'ns': 'urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0'}
    conn = sqlite3.connect('../load_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS actual_load (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            quantity REAL NOT NULL
        )
    ''')
    for timeseries in root.findall('ns:TimeSeries', ns):
        period = timeseries.find('ns:Period', ns)
        start_str = period.find('ns:timeInterval/ns:start', ns).text
        resolution = period.find('ns:resolution', ns).text
        start_time = datetime.strptime(start_str, '%Y-%m-%dT%H:%MZ')
        step = timedelta(minutes=15) if resolution == 'PT15M' else timedelta(hours=1)
        for point in period.findall('ns:Point', ns):
            position = int(point.find('ns:position', ns).text)
            quantity = float(point.find('ns:quantity', ns).text)
            dt = start_time + (position - 1) * step
            cursor.execute('INSERT INTO actual_load (datetime, quantity) VALUES (?, ?)', (dt.isoformat(), quantity))

    conn.commit()
    conn.close()


def select_all():
    conn = sqlite3.connect('../load_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='actual_load'")
    if cursor.fetchone() is None:
        print("Таблица 'actual_load' не найдена.")
        conn.close()
        return
    cursor.execute("SELECT * FROM actual_load ORDER BY datetime")  # Ограничим до 100 строк
    rows = cursor.fetchall()
    print("ID | Дата и время         | Значение (MW)")
    print("---|----------------------|----------------")
    for row in rows:
        print(f"{row[0]:>2} | {row[1]} | {row[2]:.2f}")
    conn.close()

def export_sqlite_to_csv(db_path, output_csv, aggregate_to_hourly=False):
    """Экспорт данных из SQLite в CSV."""
    conn = sqlite3.connect(db_path)
    query = "SELECT datetime, quantity FROM actual_load ORDER BY datetime"
    df = pd.read_sql_query(query, conn)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.rename(columns={'datetime': 'ds', 'quantity': 'y'})

    if aggregate_to_hourly:
        df = df.set_index('ds').resample('H').mean().reset_index()
        df = df.dropna()

    df.to_csv(output_csv, index=False)
    print(f"Данные экспортированы в {output_csv}")

    conn.close()
    return df

if __name__ == "__main__":
    export_sqlite_to_csv('../load_data.db', 'energy_consumption.csv', aggregate_to_hourly=True)
    #print("Данные успешно сохранены в load_data.db")
    #parse_xml()
    #select_all()