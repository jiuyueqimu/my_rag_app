import streamlit as st
from streamlit_javascript import st_javascript
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os
from weather_tool import get_current_weather, get_weather_by_coords

# ---------- 页面配置 ----------
st.set_page_config(page_title="天气查询助手", page_icon="🌤️")
st.title("🌤️ AI 天气查询助手")
st.markdown("输入城市名称，或点击下方按钮自动定位获取天气。")

# ---------- API 密钥 ----------
# 请替换为你的真实密钥
os.environ["OPENAI_API_KEY"] = "sk-78de76ffe49e4e058f7b7f1054387046"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# ---------- 初始化 Agent ----------
llm = ChatOpenAI(model="deepseek-chat", temperature=0)
tools = [get_current_weather, get_weather_by_coords]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个天气助手。如果用户提供城市名，使用 get_current_weather；如果用户提供了经纬度，使用 get_weather_by_coords。"
)

# ---------- 状态管理 ----------
if "query_triggered" not in st.session_state:
    st.session_state.query_triggered = False
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

# ---------- 定位按钮 ----------
if st.button("📍 定位当前位置"):
    with st.spinner("正在获取位置..."):
        # 调用 st_javascript，返回 Promise 解析的经纬度对象
        coords = st_javascript("""
            if (navigator.geolocation) {
                return new Promise((resolve) => {
                    navigator.geolocation.getCurrentPosition(
                        (pos) => resolve({lat: pos.coords.latitude, lon: pos.coords.longitude}),
                        (err) => resolve(null)  // 定位失败返回 null
                    );
                });
            } else {
                return null;
            }
        """, key="geolocation")  # 增加 key 避免缓存

        if coords:
            lat = coords['lat']
            lon = coords['lon']
            st.session_state.coords = (lat, lon)
            st.session_state.query_triggered = True
            st.session_state.query_text = f"我的当前位置经纬度是 {lat}, {lon}，请告诉我天气情况。"
            st.success(f"✅ 定位成功！纬度 {lat:.4f}, 经度 {lon:.4f}")
        else:
            st.error("❌ 定位失败。请检查浏览器是否授予位置权限，或手动输入城市名称。")
            st.session_state.query_triggered = False

# ---------- 手动输入 ----------
city_input = st.text_input("或输入城市名称", placeholder="例如：北京、上海、London")

if st.button("查询天气"):
    if city_input:
        st.session_state.query_triggered = True
        st.session_state.query_text = f"{city_input}的天气怎么样？"
    else:
        st.warning("请输入城市名称或使用定位。")

# ---------- 执行查询（由定位或手动按钮触发） ----------
if st.session_state.query_triggered and st.session_state.query_text:
    with st.spinner("正在获取天气信息..."):
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": st.session_state.query_text}]}
            )
            reply = response["messages"][-1].content
            st.success("查询成功！")
            st.info(reply)
        except Exception as e:
            st.error(f"❌ 出错了: {e}")
        finally:
            # 重置触发标志，避免重复查询
            st.session_state.query_triggered = False
            st.session_state.query_text = ""