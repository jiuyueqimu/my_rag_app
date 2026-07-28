import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import tempfile
import time
import streamlit as st
import pdfplumber
import docx2txt
import textract
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.documents import Document
from datetime import datetime
import json
import shutil

# ========== 页面配置（必须在最前面） ==========
st.set_page_config(page_title="个人知识库问答", page_icon="📚")

# ========== 用户登录 ==========
if "username" not in st.session_state or not st.session_state.username:
    st.title("👤 欢迎使用个人知识库问答")
    st.markdown("请输入你的用户名，系统会为你独立保存所有文档和对话历史。")
    username_input = st.text_input("用户名", placeholder="例如：张三")
    if st.button("进入"):
        if username_input.strip():
            st.session_state.username = username_input.strip()
            st.rerun()
        else:
            st.warning("用户名不能为空")
    st.stop()  # 阻止后续代码执行，只显示登录界面

# ========== 配置（根据用户名动态设置） ==========
username = st.session_state.get("username", "default")
BASE_DB_DIR = f"./chroma_dbs_{username}"   # 不同用户使用不同目录
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# =========================

os.makedirs(BASE_DB_DIR, exist_ok=True)
MANIFEST_FILE = os.path.join(BASE_DB_DIR, "manifest.json")

# ---------- 全局变量 ----------
_CURRENT_DB_PATH = None

# ---------- manifest 操作 ----------
def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return sorted(data, key=lambda x: x.get('created_at', ''), reverse=True)
            except:
                return []
    return []

def save_manifest(manifest):
    with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def add_document_to_manifest(doc_id, doc_name):
    manifest = load_manifest()
    manifest = [item for item in manifest if item['id'] != doc_id]
    manifest.append({'id': doc_id, 'name': doc_name, 'created_at': datetime.now().isoformat()})
    save_manifest(manifest)

def remove_document_from_manifest(doc_id):
    manifest = load_manifest()
    manifest = [item for item in manifest if item['id'] != doc_id]
    save_manifest(manifest)

def delete_document(doc_id):
    doc_path = os.path.join(BASE_DB_DIR, doc_id)
    if os.path.exists(doc_path):
        shutil.rmtree(doc_path, ignore_errors=True)
    remove_document_from_manifest(doc_id)

# ---------- 核心函数 ----------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
    )

@st.cache_resource
def get_llm():
    # 尝试从 st.secrets 读取，如果失败则使用硬编码（仅本地测试用）
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        # 本地开发时可以使用硬编码，但不要上传到 GitHub
        api_key = "sk-你的本地测试Key"  # 请替换，或者使用环境变量
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
    return ChatOpenAI(model="deepseek-chat", temperature=0, timeout=60)

def create_new_db(chunks, embeddings):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    new_db_path = os.path.join(BASE_DB_DIR, f"db_{timestamp}")
    os.makedirs(new_db_path, exist_ok=True)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=new_db_path)
    vectorstore.persist()
    return os.path.basename(new_db_path)

def process_uploaded_file(uploaded_file, status_placeholder):
    global _CURRENT_DB_PATH
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        status_placeholder.info(f"⏳ 正在解析 {uploaded_file.name} ...")
        text = ""
        if suffix == ".pdf":
            progress_bar = st.progress(0, text="正在解析 PDF...")
            try:
                with pdfplumber.open(tmp_path) as pdf:
                    total = len(pdf.pages)
                    for i, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                        progress_bar.progress((i + 1) / total, text=f"解析第 {i+1}/{total} 页")
                progress_bar.empty()
            except Exception as e:
                status_placeholder.error(f"❌ PDF 解析失败：{str(e)}")
                return False, f"PDF 解析失败：{str(e)}"
            if not text.strip():
                status_placeholder.error("❌ PDF 文件不包含可提取的文本。")
                return False, "PDF 文件不包含可提取的文本。"

        elif suffix == ".docx":
            try:
                text = docx2txt.process(tmp_path)
            except Exception as e:
                status_placeholder.error(f"❌ DOCX 解析失败：{str(e)}")
                return False, f"DOCX 解析失败：{str(e)}"
            if not text.strip():
                status_placeholder.error("❌ DOCX 文件不包含可提取的文本。")
                return False, "DOCX 文件不包含可提取的文本。"

        elif suffix == ".doc":
            try:
                text = textract.process(tmp_path).decode('utf-8', errors='ignore')
            except Exception as e:
                status_placeholder.error(f"❌ DOC 解析失败（请另存为 DOCX 再上传）：{str(e)}")
                return False, f"DOC 解析失败（请另存为 DOCX 再上传）：{str(e)}"
            if not text.strip():
                status_placeholder.error("❌ DOC 文件不包含可提取的文本。")
                return False, "DOC 文件不包含可提取的文本。"

        elif suffix == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()
            text = docs[0].page_content
            if not text.strip():
                status_placeholder.error("❌ TXT 文件为空。")
                return False, "TXT 文件为空。"
        else:
            status_placeholder.error("❌ 不支持的文件格式，请上传 PDF, DOC, DOCX 或 TXT。")
            return False, "不支持的文件格式，请上传 PDF, DOC, DOCX 或 TXT。"

        status_placeholder.info(f"⏳ 正在切分并存入向量库...")
        docs = [Document(page_content=text)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embeddings = get_embeddings()
        doc_id = create_new_db(chunks, embeddings)

        add_document_to_manifest(doc_id, uploaded_file.name)
        new_path = os.path.join(BASE_DB_DIR, doc_id)
        st.session_state.current_db_path = new_path
        _CURRENT_DB_PATH = new_path
        if "chat_histories" not in st.session_state:
            st.session_state.chat_histories = {}
        st.session_state.chat_histories[doc_id] = []

        status_placeholder.success(f"✅ 成功添加「{uploaded_file.name}」，共 {len(chunks)} 个片段。")
        return True, f"✅ 成功添加「{uploaded_file.name}」，共 {len(chunks)} 个片段。"
    except Exception as e:
        status_placeholder.error(f"❌ 处理失败：{str(e)}")
        return False, f"❌ 处理失败：{str(e)}"
    finally:
        os.unlink(tmp_path)

# ---------- 检索工具 ----------
@tool(description="""【必用工具】当用户问及任何关于文档、文件、文章、内容、信息的问题时，必须调用此工具。包括但不限于：文档讲了什么、这篇文章在说什么、帮我总结、提取关键信息、文档里提到了什么、作者是谁、主要观点等。""")
def query_knowledge(question: str) -> str:
    db_path = _CURRENT_DB_PATH
    if not db_path or not os.path.exists(db_path) or not os.listdir(db_path):
        return "📭 知识库为空，请先上传文档。"
    embeddings = get_embeddings()
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    if not docs:
        return "📭 未找到相关内容，请换个问题再试。"
    snippet_parts = []
    for idx, doc in enumerate(docs, 1):
        snippet = doc.page_content[:300].replace('\n', ' ')
        snippet_parts.append(f"[片段{idx}] {snippet}...")
    snippets = "\n\n".join(snippet_parts)
    context = "\n---\n".join([doc.page_content for doc in docs])
    return f"📚 检索到的相关资料（共{len(docs)}个片段）：\n{snippets}\n\n完整上下文：\n{context}"

def stream_text(content: str):
    for char in content:
        yield char
        time.sleep(0.015)

llm = get_llm()

# ---------- Agent 系统提示 ----------
SYSTEM_PROMPT = """你是一个知识库问答助手。你的任务是基于用户上传的文档内容回答问题。

**⚠️ 调用规则（必须严格遵守）：**
1. 只要用户的问题涉及任何文档、文章、内容、信息，你**必须**调用 query_knowledge 工具来检索。
2. **禁止**在没有调用工具的情况下直接回答涉及文档内容的问题。
3. 如果工具返回 "知识库为空"，则告诉用户"知识库为空，请先上传文档"。
4. 如果工具返回了检索结果，请基于检索结果回答，不要添加检索结果以外的信息。

**示例：**
- 用户："这个文档讲的是什么" → 调用 query_knowledge("这个文档讲的是什么")
- 用户："帮我总结一下内容" → 调用 query_knowledge("总结文档内容")
- 用户："有什么关键信息" → 调用 query_knowledge("关键信息")
- 用户："你好" → 可以不用工具，直接回答

请严格遵循以上规则。"""

agent = create_agent(
    model=llm,
    tools=[query_knowledge],
    system_prompt=SYSTEM_PROMPT
)

# ---------- 初始化 session_state ----------
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
if "current_db_path" not in st.session_state:
    manifest = load_manifest()
    if manifest:
        latest = manifest[0]
        st.session_state.current_db_path = os.path.join(BASE_DB_DIR, latest['id'])
    else:
        st.session_state.current_db_path = None

# ---------- 同步全局变量 ----------
_CURRENT_DB_PATH = st.session_state.current_db_path

def get_current_doc_id():
    if st.session_state.current_db_path:
        return os.path.basename(st.session_state.current_db_path)
    return None

def ensure_current_history():
    doc_id = get_current_doc_id()
    if doc_id and doc_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[doc_id] = []

# ---------- 界面 ----------
st.title("📚 个人知识库问答助手")
st.markdown(f"当前用户：**{username}**")
st.markdown("上传文档后，在下方输入问题，AI 会基于文档内容回答，并记住对话上下文。")

with st.sidebar:
    st.header("📁 文档管理")
    uploaded_files = st.file_uploader(
        "上传文档（支持 PDF / DOC / DOCX / TXT）",
        type=["pdf", "doc", "docx", "txt"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            if f"uploaded_{file.name}" not in st.session_state:
                status_placeholder = st.empty()
                success, msg = process_uploaded_file(file, status_placeholder)
                if success:
                    st.session_state[f"uploaded_{file.name}"] = True

    st.divider()

    st.subheader("📄 文档库")
    manifest = load_manifest()
    if manifest:
        options = {item['name']: item['id'] for item in manifest}
        current_id = get_current_doc_id()
        default_index = 0
        for idx, (name, doc_id) in enumerate(options.items()):
            if doc_id == current_id:
                default_index = idx
                break
        selected_name = st.radio(
            "选择文档",
            options=list(options.keys()),
            index=default_index,
            key="doc_selector",
            label_visibility="collapsed"
        )
        selected_id = options[selected_name]

        if selected_id != current_id:
            new_path = os.path.join(BASE_DB_DIR, selected_id)
            if os.path.exists(new_path) and os.listdir(new_path):
                st.session_state.current_db_path = new_path
                _CURRENT_DB_PATH = new_path
                ensure_current_history()
                st.rerun()

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️", help="删除此文档"):
                delete_document(selected_id)
                if st.session_state.current_db_path == os.path.join(BASE_DB_DIR, selected_id):
                    new_manifest = load_manifest()
                    if new_manifest:
                        st.session_state.current_db_path = os.path.join(BASE_DB_DIR, new_manifest[0]['id'])
                    else:
                        st.session_state.current_db_path = None
                if selected_id in st.session_state.chat_histories:
                    del st.session_state.chat_histories[selected_id]
                _CURRENT_DB_PATH = st.session_state.current_db_path
                st.rerun()

        current_path = st.session_state.current_db_path
        if current_path and os.path.exists(current_path):
            st.caption(f"📂 当前: {selected_name}")
    else:
        st.info("暂无文档，请上传。")

    st.divider()

    if st.button("🔄 重置所有数据（清空全部文档）"):
        shutil.rmtree(BASE_DB_DIR, ignore_errors=True)
        os.makedirs(BASE_DB_DIR, exist_ok=True)
        st.session_state.current_db_path = None
        _CURRENT_DB_PATH = None
        st.session_state.chat_histories = {}
        for key in list(st.session_state.keys()):
            if key.startswith("uploaded_"):
                del st.session_state[key]
        st.success("✅ 已清空所有文档。")
        st.rerun()

    st.caption("💡 上传新文档后会自动切换到新文档。")

# ---------- 显示当前文档的对话历史 ----------
doc_id = get_current_doc_id()
ensure_current_history()
history = st.session_state.chat_histories.get(doc_id, [])

for msg in history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- 用户输入（多轮对话记忆） ----------
if prompt := st.chat_input("输入你的问题..."):
    if not doc_id:
        st.warning("请先上传或选择一个文档。")
    else:
        # 将用户问题加入历史
        history.append({"role": "user", "content": prompt})
        st.session_state.chat_histories[doc_id] = history

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.info("⏳ 正在检索知识库...")
            try:
                # 构建完整的消息列表（包含所有历史）
                messages = []
                for msg in history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
                # 调用 Agent，传入完整消息列表
                response = agent.invoke({"messages": messages})

                # ---------- 提取回答 ----------
                answer = None
                if isinstance(response, dict):
                    if "output" in response:
                        answer = response["output"]
                    elif "result" in response:
                        answer = response["result"]
                    elif "messages" in response and isinstance(response["messages"], list):
                        if response["messages"]:
                            last = response["messages"][-1]
                            if hasattr(last, "content"):
                                answer = last.content
                            elif isinstance(last, dict) and "content" in last:
                                answer = last["content"]
                    else:
                        answer = str(response) if response else "⚠️ 未能获取到有效回答。"
                elif hasattr(response, "content"):
                    answer = response.content
                else:
                    answer = str(response) if response else "⚠️ 未能获取到有效回答。"
                
                if not answer or answer.strip() == "":
                    answer = "⚠️ 未能获取到有效回答，请重试。"
                
                placeholder.empty()
                st.write(answer)
                # 将助手回答加入历史
                history.append({"role": "assistant", "content": answer})
                st.session_state.chat_histories[doc_id] = history
            except Exception as e:
                placeholder.error(f"❌ 出错了：{e}")