import os
from dotenv import load_dotenv

# --- 1. 加载环境变量 ---
load_dotenv()

# --- 2. 网络代理配置 (根据你的实际情况修改端口) ---
PROXY_URL = os.getenv("HTTP_PROXY_URL", "http://127.0.0.1:57751")  
os.environ["http_proxy"] = PROXY_URL
os.environ["https_proxy"] = PROXY_URL
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

# --- 3. 全局常量定义 ---
# 统一管理模型名称，防止在代码里散落各处
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"