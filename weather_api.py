import requests
from datetime import datetime, timedelta
import pandas as pd

API_KEY = "#########################"   

def get_lat_lon(city):
    """Get latitude and longitude using OpenWeatherMap Geocoding API."""
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    response = requests.get(geo_url)
    data = response.json()

    if not data:
        print("❌ Invalid city name or API key.")
        return None, None

    lat = data[0]["lat"]
    lon = data[0]["lon"]
    print(f"📍 Location: {city} -> lat: {lat}, lon: {lon}")
    return lat, lon


def get_monthly_weather(lat, lon):
    """Fetch monthly rainfall, temperature, and humidity for the past 1 year."""
    end = datetime.now().date() - timedelta(days=1)
    start = end.replace(year=end.year - 1)

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean"
        f"&timezone=auto&start_date={start}&end_date={end}"
    )

    response = requests.get(url).json()
    if "daily" not in response:
        print("❌ No data found for this location.")
        return None

    df = pd.DataFrame(response["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.to_period("M")

    # Aggregate daily to monthly averages/sums
    monthly = (
        df.groupby("month")[["precipitation_sum", "temperature_2m_max", "temperature_2m_min", "relative_humidity_2m_mean"]]
        .agg({
            "precipitation_sum": "sum",
            "temperature_2m_max": "mean",
            "temperature_2m_min": "mean",
            "relative_humidity_2m_mean": "mean"
        })
        .reset_index()
        .rename(columns={
            "precipitation_sum": "rainfall_mm",
            "relative_humidity_2m_mean": "humidity_pct"
        })
    )

    monthly["month_name"] = monthly["month"].dt.strftime("%B").str.lower()

    print("\n📆 Monthly Weather Data (Rainfall, Temp & Humidity):")
    print(monthly[["month_name", "rainfall_mm", "humidity_pct"]])

    return monthly


def get_three_month_average(monthly_df, start_month):
    """Compute 3-month average rainfall, temperature, and humidity."""
    start_month = start_month.lower()
    if start_month not in monthly_df["month_name"].values:
        print("❌ Invalid month name. Please try again (e.g., May, July, September).")
        return None

    start_idx = monthly_df[monthly_df["month_name"] == start_month].index[0]
    indices = [(start_idx + i) % len(monthly_df) for i in range(3)]  # wrap around
    subset = monthly_df.loc[indices]

    avg_rainfall = subset["rainfall_mm"].mean()
    avg_temp = (subset["temperature_2m_max"].mean() + subset["temperature_2m_min"].mean()) / 2
    avg_humidity = subset["humidity_pct"].mean()

    print(f"\n🌾 Selected Months: {', '.join(subset['month_name'].str.capitalize())}")
    print(f"🌡️ Avg Temp (3M): {avg_temp:.2f} °C")
    print(f"🌧️ Avg Rainfall (3M): {avg_rainfall:.2f} mm")
    print(f"💧 Avg Humidity (3M): {avg_humidity:.2f}%")

    return avg_temp, avg_rainfall, avg_humidity

if __name__ == "__main__":
    city = input("Enter city name: ")
    lat, lon = get_lat_lon(city)

    if lat and lon:
        monthly_data = get_monthly_weather(lat, lon)
        if monthly_data is not None:
            user_month = input("\nEnter the month (e.g., May, July, September): ")
            get_three_month_average(monthly_data, user_month)
