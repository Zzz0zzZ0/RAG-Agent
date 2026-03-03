from typing import Annotated, TypedDict
from langchain_core.messages import SystemMessage, AnyMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_ollama import ChatOllama

# 引入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent.tools import LocalKnowledgeTool

# ==========================================
# 1. 定义状态 (State) 
# 这是图网络中的“血液”，在各个节点之间流转。包含对话的全部历史。
# ==========================================
class AgentState(TypedDict):
    # add_messages 会自动将新产生的消息追加到历史记录中
    messages: Annotated[list[AnyMessage], add_messages]

def build_graph_agent(vectorstore, bm25_retriever, **kwargs):
    """基于 LangGraph 状态机重构的 Agent 大脑"""
    
    # 2. 准备工具箱 (和以前一样)
    local_tool_instance = LocalKnowledgeTool(vectorstore, bm25_retriever, **kwargs)
    tools =[
        local_tool_instance.get_tool(),
        DuckDuckGoSearchRun(name="web_search", description="当你需要查询外部世界的实时信息（股票、天气、新闻等）时使用。")
    ]

    # 3. 唤醒大模型并挂载工具 (bind_tools 是 LangGraph 的精髓)
    model_type = kwargs.get("model_type", "cloud")
    temp_val = kwargs.get("temp_param", 0.1)

    if model_type == "local":
        llm = ChatOllama(model="qwen3:8b", temperature=temp_val)
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=temp_val,
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        )
    
    # 给大脑装上手脚
    llm_with_tools = llm.bind_tools(tools)

    # 4. 定义系统人格 (System Prompt)
    sys_msg = SystemMessage(content=(
        "你是一个高级全能AI助手（Agent）。请遵循以下原则：\n"
        "1. 查阅内部信息，使用 local_knowledge_base 工具。\n"
        "2. 查阅外部实时事实，使用 web_search 工具。\n"
        "3. 务必提取核心关键词进行搜索！如'上证指数 今日收盘 东方财富'。\n"
        "4. 如果任务复杂，请分步骤多次调用工具。"
    ))

    # ==========================================
    # 5. 定义图网络节点 (Nodes)
    # ==========================================
    # 节点 A: 大脑思考节点
    def call_model(state: AgentState):
        # 每次思考前，把系统指令放在最前面，加上之前的历史对话
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": [response]} # 返回的新消息会自动追加到 State 中

    # 节点 B: 工具执行节点 (使用 LangGraph 内置的 ToolNode)
    tool_node = ToolNode(tools)

    # ==========================================
    # 6. 编织图网络 (Edges & Routing)
    # ==========================================
    workflow = StateGraph(AgentState)

    # 注册节点
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # 画线: 起点 -> agent (开始思考)
    workflow.add_edge(START, "agent")

    # 画条件线: agent 思考完后，有两个岔路口：
    # 1. 如果它决定调用工具 -> 走向 "tools" 节点
    # 2. 如果它给出了最终回答 -> 走向 END (结束网络)
    workflow.add_conditional_edges("agent", tools_condition)

    # 画线: tools -> agent (工具执行完毕后，强制把结果传回给大脑继续思考)
    workflow.add_edge("tools", "agent")

    # 编译成最终的引擎
    return workflow.compile()