import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# 导入你自己写的各个组件
from memory.rag_engine import embeddings
from perception.nlp_pipeline import analyze_emotion, extract_entities
from algorithms.linucb import linucb_agent
from algorithms.prf import algo_pseudo_relevance_feedback
from algorithms.mmr import algo_mmr_rerank

def get_answer_complex(vectorstore, bm25_retriever, question, k_param=3, temp_param=0.1, alpha=0.5, 
                       model_type="cloud", use_multiquery=False, use_rerank=False, 
                       use_auto_alpha=False, use_emotion=False, use_ner=False):
    """最核心的单步认知检索回路"""
    
    # 0. NLP 情感前处理
    emotion_label = "neutral"
    if use_emotion:
        emotion_label, _ = analyze_emotion(question)

    # 1. 模型初始化 
    if model_type == "local":
        llm = ChatOllama(model="qwen3", temperature=temp_param) # 默认使用最新的 Qwen 3
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temp_param)

    # 2. LinUCB 自适应 Alpha
    final_alpha = alpha
    arm_idx = -1
    context_vec = None
    if use_auto_alpha:
        arm_idx, final_alpha, context_vec = linucb_agent.select_arm(question)

    # 3. 初始混合检索 (广域召回)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[1-final_alpha, final_alpha]
    )
    initial_docs = ensemble.invoke(question)

    # 4. PRF 发散思维
    search_queries = [question]
    if use_multiquery:
        search_queries = algo_pseudo_relevance_feedback(question, initial_docs)
        if len(search_queries) > 1:
            more_docs = ensemble.invoke(search_queries[1])
            initial_docs.extend(more_docs)
    
    unique_docs = list({doc.page_content: doc for doc in initial_docs}.values())

    # 5. LinUCB 奖励更新
    if use_auto_alpha and unique_docs:
        q_vec = embeddings.embed_query(question)
        d_vec = embeddings.embed_query(unique_docs[0].page_content)
        reward = cosine_similarity([q_vec], [d_vec])[0][0]
        linucb_agent.update(arm_idx, context_vec, reward)

    # 6. 重排序与截断 (MMR)
    if use_rerank:
        final_docs = algo_mmr_rerank(question, unique_docs, embeddings, k_param=k_param)
    else:
        final_docs = unique_docs[:k_param]

    # 7. 生成阶段 (注入情感指令)
    tone_instruction = ""
    if emotion_label == "negative":
        tone_instruction = "检测到用户情绪焦虑。请使用安抚性、有同理心的语气回答。"
    elif emotion_label == "positive":
        tone_instruction = "用户情绪积极。请保持热情。"

    system_prompt = f"你是一个认知智能体。{tone_instruction}\n根据记忆片段回答。如果不知道说不知道。\n\n【记忆片段】:\n{{context}}"
    prompt = ChatPromptTemplate.from_template(system_prompt + "\n\n问题: {input}")
    context_text = "\n\n".join([d.page_content for d in final_docs])
    
    res = (prompt | llm).invoke({"input": question, "context": context_text})
    answer = res.content if hasattr(res, 'content') else str(res)
    
    # 8. 实体提取 (韦尼克区)
    extracted_entities =[]
    if use_ner:
        extracted_entities = extract_entities(answer)

    return {
        "answer": answer, "context": final_docs, 
        "generated_queries": search_queries,
        "used_alpha": final_alpha, "emotion": emotion_label,
        "entities": extracted_entities
    }

# --- Agent 专属封装 ---
class LocalSearchInput(BaseModel):
    query: str = Field(description="需要搜索的具体问题字符串")

class LocalKnowledgeTool:
    def __init__(self, vectorstore, bm25_retriever, **kwargs):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.config = {
            "k_param": kwargs.get("k_param", 3),
            "temp_param": kwargs.get("temp_param", 0.1),
            "alpha": kwargs.get("alpha", 0.5),
            "model_type": kwargs.get("model_type", "cloud"),
            "use_multiquery": kwargs.get("use_multiquery", True),
            "use_rerank": kwargs.get("use_rerank", True),
            "use_auto_alpha": kwargs.get("use_auto_alpha", True),
            "use_emotion": kwargs.get("use_emotion", True),
            "use_ner": kwargs.get("use_ner", False),
        }

    def _run_search(self, query: str) -> str:
        result = get_answer_complex(
            vectorstore=self.vectorstore,
            bm25_retriever=self.bm25_retriever,
            question=query,
            **self.config
        )
        docs_text = "\n".join([f"- {d.page_content}" for d in result["context"]])
        output = (
            f"【初步结论】: {result['answer']}\n"
            f"【参考片段】:\n{docs_text}\n"
            f"【情感状态】: {result['emotion']}\n"
        )
        return output

    def get_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self._run_search,
            name="local_knowledge_base",
            description="当用户询问关于内部文档、专属知识库、特定概念等内容时，必须调用此工具。",
            args_schema=LocalSearchInput
        )