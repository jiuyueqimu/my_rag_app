import requests

API_KEY = "sk-62f62426ef14428b8ab2098626475b91"  # ← 务必替换成真实的新 Key
BASE_URL = "https://api.deepseek.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "1+1等于几？"}],
    "temperature": 0
}

try:
    response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=10)
    print("HTTP 状态码:", response.status_code)
    print("响应内容:", response.text)
    if response.status_code == 200:
        print("✅ API 连接成功！")
    else:
        print("❌ API 返回错误，请检查 Key 或网络。")
except Exception as e:
    print("❌ 请求失败:", repr(e))