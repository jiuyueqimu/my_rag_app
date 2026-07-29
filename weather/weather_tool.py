import requests
import time
from langchain.tools import tool

# ---------- 辅助：地理编码 ----------
def geocode_with_fallback(location: str):
    """将地名转为经纬度（Open-Meteo + Nominatim 备选）"""
    # 1. Open-Meteo
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=zh&format=json"
    try:
        response = requests.get(geo_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                result = data['results'][0]
                return {
                    'latitude': result['latitude'],
                    'longitude': result['longitude'],
                    'name': result.get('name', location)
                }
    except Exception:
        pass

    # 2. Nominatim 备选
    print(f"⚠️ Open-Meteo 未找到 '{location}'，尝试 Nominatim...")
    nominatim_url = "https://nominatim.openstreetmap.org/search"
    params = {'q': location, 'format': 'json', 'limit': 1}
    headers = {'User-Agent': 'WeatherApp/1.0 (your-email@example.com)'}
    try:
        response = requests.get(nominatim_url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                return {
                    'latitude': float(data[0]['lat']),
                    'longitude': float(data[0]['lon']),
                    'name': data[0].get('display_name', location)
                }
    except Exception as e:
        print(f"❌ Nominatim 失败: {e}")
    return None

def fetch_weather(lat, lon, city_name=None):
    """根据经纬度获取天气数据，返回友好字符串"""
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true&timezone=auto"
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(weather_url, timeout=10)
            response.raise_for_status()
            weather_data = response.json()
            current = weather_data.get('current_weather', {})
            if not current:
                return None
            temp = current.get('temperature')
            wind_speed = current.get('windspeed')
            weather_code = current.get('weathercode', 0)
            weather_desc = {
                0: "☀️ 晴", 1: "⛅ 多云", 2: "☁️ 阴", 3: "🌧️ 雨",
                45: "🌫️ 雾", 48: "🌫️ 雾", 51: "🌦️ 小雨", 61: "🌧️ 中雨"
            }.get(weather_code, "🌡️ 未知")
            name_part = f"**{city_name}**" if city_name else f"纬度 {lat}, 经度 {lon}"
            return (
                f"📍 {name_part} 的当前天气：\n"
                f"{weather_desc}\n"
                f"🌡️ 温度：{temp}°C\n"
                f"💨 风速：{wind_speed} km/h"
            )
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None
    return None

# ---------- 原有的工具（通过城市名） ----------
@tool
def get_current_weather(location: str) -> str:
    """通过城市名称查询该地当前的天气情况。"""
    location_data = geocode_with_fallback(location)
    if not location_data:
        return f"❌ 抱歉，未能找到城市 '{location}' 的天气信息。"
    result = fetch_weather(location_data['latitude'], location_data['longitude'], location_data['name'])
    return result if result else f"⚠️ 未能获取到 '{location}' 的天气数据。"

# ---------- 新增工具：直接通过经纬度 ----------
@tool
def get_weather_by_coords(latitude: float, longitude: float) -> str:
    """
    通过经纬度坐标直接查询当前天气。
    当用户提供了精确经纬度或通过定位获取到坐标时使用此工具。
    """
    result = fetch_weather(latitude, longitude)
    return result if result else f"⚠️ 未能获取到该坐标（{latitude}, {longitude}）的天气数据。"