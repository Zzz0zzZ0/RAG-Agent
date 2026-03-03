# Hybrid RAG & ReAct Agent System

本项目是一个基于 LangChain 和 Streamlit 构建的**混合检索增强生成与智能体决策系统**。系统融合了经典的检索算法（如 BM25、LinUCB、MMR、PRF）与现代大模型智能体（ReAct Agent）架构，支持自主规划任务并在本地知识库与外部网络搜索之间进行路由调度。

## ⚙️ 核心技术特性

*   **多步推理智能体 (ReAct Agent)**
    *   基于 `AgentExecutor` 实现，支持大模型通过 Thought-Action-Observation 闭环拆解复杂任务。
    *   动态工具调用：集成自定义的本地混合检索工具 (`LocalKnowledgeTool`) 与广域网搜索工具 (`DuckDuckGoSearchRun`)。
*   **自适应混合检索 (Hybrid RAG + LinUCB)**
    *   **双路召回**: 结合 ChromaDB 稠密向量检索与 BM25 稀疏词频检索。
    *   **动态权重调节**: 引入基于上下文的 LinUCB 强化学习算法，根据查询特征（长度、特殊字符密度等）在线自适应调整双路召回的 Alpha 权重。
*   **高阶检索增强策略 (PRF & MMR)**
    *   **伪相关反馈 (PRF)**: 结合 TF-IDF 算法提取初次召回文档的关键词，实现查询自动发散与扩展 (Query Expansion)。
    *   **最大边际相关性 (MMR)**: 对召回结果进行多样性重排，降低信息冗余度，提升上下文利用率。
*   **NLP 辅助处理流水线**
    *   集成本地 Hugging Face Pipeline，对用户输入进行情感分析 (Sentiment Analysis)，并对大模型输出进行命名实体识别 (NER) 抽提。
*   **可视化交互面板**
    *   基于 Streamlit 实现的参数控制台。
    *   集成 PCA 降维算法，提供本地向量库语义空间 (Semantic Space) 的 2D 散点图可视化。

## 🚀 安装与环境配置

### 1. 基础环境
建议使用 Python 3.10+ 环境。克隆项目后安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：`langchain`、`streamlit`、`chromadb`、`scikit-learn`、`transformers`、`duckduckgo-search` 等，详见 `requirements.txt`。

### 2. ⚠️ 网络代理配置 (访问 Gemini 必读)
如果您选择使用 Google Gemini 作为云端大模型，在国内或特定网络环境下，必须配置本地代理 (VPN) 才能正常访问 API。
请打开 `backend.py`，在文件顶部找到以下代码，并将端口号修改为您本地代理软件的实际端口（例如 Clash 常用的 7890 或 V2ray 的 10808 等）：

```python
PROXY_URL = "http://127.0.0.1:您的代理端口"
os.environ["http_proxy"] = PROXY_URL
os.environ["https_proxy"] = PROXY_URL
```

### 3. API 密钥配置
在项目根目录创建一个 `.env` 文件，写入您的 Google Gemini API 密钥：

```env
GOOGLE_API_KEY=your_api_key_here
```

### 4. 本地大模型配置 (可选 / 推荐)
系统完全支持本地化离线运行，彻底解决 API 限流与网络问题。如果您拥有 8GB 以上显存的设备，推荐安装 Ollama 并拉取模型：

```bash
ollama run qwen3  # 推荐使用最新的 Qwen 3 系列
```

## 🖥️ 运行与使用指南

启动 Streamlit 服务：

```bash
streamlit run app.py
```

### 功能面板说明
- **知识库构建 (文件上传)**：通过左侧边栏上传 PDF 或 TXT 文件。系统会自动执行文本切分 (Chunking) 并构建本地 Chroma 向量库。
- **Agent 模式开关**：
  - 开启：激活多步推理能力，模型将根据您的问题，自主决定是否调用网络搜索，并在聊天界面打印完整的推理链 (Chain of Thought)。
  - 关闭：回退至极速的单步混合 RAG 问答模式。
- **算法参数微调**：支持在 UI 界面实时开启/关闭 LinUCB 自适应学习、PRF 伪相关反馈、MMR 重排序，并可手动调节 Top-K 召回数量及模型生成温度 (Temperature)。

## 📁 项目结构简述
- `app.py`：Streamlit 前端交互层，负责 UI 渲染与状态管理。
- `backend.py`：核心算法引擎层，包含 RAG 构建流水线、机器学习算法实现 (LinUCB/MMR)、NLP 模型加载及 Agent 工具组装逻辑。