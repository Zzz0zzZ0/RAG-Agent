from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_ollama import ChatOllama

# 导入刚才封装好的内部工具
from agent.tools import LocalKnowledgeTool

def build_agent_executor(vectorstore, bm25_retriever, **kwargs):
    """构建完整的 Agent 大脑，并挂载所有可用工具"""
    
    # 1. 挂载内部知识库工具
    local_tool_instance = LocalKnowledgeTool(vectorstore, bm25_retriever, **kwargs)
    local_rag_tool = local_tool_instance.get_tool()

    # 2. 挂载外部网络搜索工具
    web_search_tool = DuckDuckGoSearchRun(
        name="web_search",
        description="当你需要查询外部世界的实时信息（如股票、天气、新闻、名人百科等）时使用。"
    )

    tools = [local_rag_tool, web_search_tool]

    # 3. 动态接收前端传递的模型参数
    model_type = kwargs.get("model_type", "cloud")
    temp_val = kwargs.get("temp_param", 0.1)

    if model_type == "local":
        llm = ChatOllama(model="qwen3:8b", temperature=temp_val)
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=temp_val,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            }
        )

    # 4. 编写 Agent 的全局系统指令
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "你是一个高级全能AI助手（Agent）。请遵循以下原则：\n"
            "1. 查阅内部信息，使用 local_knowledge_base 工具。\n"
            "2. 查阅外部实时事实，使用 web_search 工具。\n"
            "3. 【重点】为了获得准确的搜索结果，请在你的 query 中加入权威网站名称！\n"
            "   - 如果查股票，请搜：'上证指数 今日收盘 东方财富'\n"
            "   - 如果查人物百科，请搜：'李雷 百度百科'\n"
            "4. 如果任务复杂，请分步骤多次调用工具。"
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), 
    ])

    # 5. 创建并组装 Agent 执行器
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)