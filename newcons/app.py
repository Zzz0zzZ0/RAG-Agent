import streamlit as st
import os
import tempfile
import plotly.express as px

# --- 模块化导入 ---
import core.config  # 触发环境变量和全局配置加载
from memory.rag_engine import build_hybrid_knowledge_base, visualize_semantic_space
from agent.tools import get_answer_complex
from agent.graph_brain import build_graph_agent

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title="Neuromorphic Brain", page_icon="🧠", layout="wide")
st.title("🧠 认知科学与类脑计算：全栈认知架构")
st.markdown("### Integrated Neuromorphic Architecture: LinUCB, MMR, PRF & NLP Modules")

# Init
if "messages" not in st.session_state: st.session_state.messages =[]
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "bm25" not in st.session_state: st.session_state.bm25 = None
if "viz_data" not in st.session_state: st.session_state.viz_data = None

# --- Sidebar ---
with st.sidebar:
    st.header("🎛️ 脑控中心")
    
    st.subheader("1. 认知核心 (Hardware)")
    brain_mode = st.radio("推理模型", ("☁️ 云端 Gemini", "💻 本地 Qwen"))
    selected_model = "cloud" if "云端" in brain_mode else "local"

    # Agent 模式开关
    use_agent = st.toggle("🌐 启用 Agent 模式 (自主规划+联网)", value=False, help="开启后，前额叶皮层激活，AI将自主拆解任务、调用网络搜索和本地记忆。")

    st.divider()

    st.subheader("2. 认知策略 (Software)")
    # LinUCB
    use_auto_alpha = st.toggle("🤖 LinUCB 自适应学习", value=True)
    if not use_auto_alpha:
        alpha = st.slider("Alpha 权重", 0.0, 1.0, 0.5)
    else:
        alpha = 0.5
        st.caption(">> 权重由 LinUCB 接管")

    # PRF & MMR
    use_multiquery = st.toggle("💡 PRF 伪相关反馈 (发散)", value=True)
    use_rerank = st.toggle("👁️ MMR 多样性重排 (聚焦)", value=True)

    st.divider()
    
    st.subheader("3. 神经调节参数")
    k_val = st.slider("工作记忆容量 (Top-K)", 1, 6, 3, help="决定最终喂给模型的记忆片段数量")
    temp_val = st.slider("突触噪声 (Temperature)", 0.0, 1.0, 0.1, help="控制生成的随机性：0=严谨，1=发散")

    st.divider()

    st.subheader("4. NLP 专用回路")
    use_emotion = st.toggle("🧠 杏仁核 (情感分析)", value=True)
    use_ner = st.toggle("🏷️ 韦尼克区 (实体识别)", value=True)

    st.divider()
    
    # Upload
    uploaded_file = st.file_uploader("📂 记忆注入 (上传知识)", type=["pdf", "txt"])
    if uploaded_file and st.session_state.vectorstore is None:
        with st.spinner("ETL 流水线运行中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            vs, bm25, count = build_hybrid_knowledge_base(tmp_path)
            st.session_state.vectorstore = vs
            st.session_state.bm25 = bm25
            st.session_state.viz_data = visualize_semantic_space(vs)
            st.success(f"记忆固化: {count} 片段")
            os.remove(tmp_path)

# --- Main ---
col_chat, col_viz = st.columns([1, 1])

with col_viz:
    st.subheader("🌌 语义流形可视化 (海马体投影)")
    if st.session_state.viz_data is not None:
        fig = px.scatter(st.session_state.viz_data, x='x', y='y', hover_data=['text'], color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无数据投影，请先注入记忆。")

with col_chat:
    st.subheader("💬 认知交互界面")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("输入指令或问题..."):
        with st.chat_message("user"): st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.vectorstore:
            with st.chat_message("assistant"):
                
                # ==========================================
                # 模式 A: Agent 自主规划模式 (新增的宏观架构)
                # ==========================================
                if use_agent:
                    st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                    try:
                        # 唤醒基于图网络的新 Agent
                        agent_app = build_graph_agent(
                            st.session_state.vectorstore, st.session_state.bm25,
                            k_param=k_val, temp_param=temp_val, alpha=alpha,
                            model_type=selected_model, use_multiquery=use_multiquery,
                            use_rerank=use_rerank, use_auto_alpha=use_auto_alpha,
                            use_emotion=use_emotion, use_ner=use_ner
                        )
                        
                        # 🚀 激活图网络流水线！(输入数据格式变为 messages 列表)
                        response = agent_app.invoke(
                            {"messages":[("user", prompt)]}, 
                            config={"callbacks": [st_callback]}
                        )
                        
                        # 提取图网络跑完后流转出的最后一条信息 (即最终答案)
                        answer = response["messages"][-1].content
                        
                        st.markdown(f"**🤖 最终结论:**\n{answer}")
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        st.error(f"Agent 前额叶皮层发生错误: {str(e)}")
                # ==========================================
                # 模式 B: 原有单步检索 RAG 模式 (原有的微观架构)
                # ==========================================
                else:
                    with st.status("🧠 神经网络单步检索激活中...", expanded=True) as status:
                        try:
                            result = get_answer_complex(
                                st.session_state.vectorstore, st.session_state.bm25, prompt,
                                k_param=k_val, temp_param=temp_val, alpha=alpha,
                                model_type=selected_model,
                                use_multiquery=use_multiquery, use_rerank=use_rerank,
                                use_auto_alpha=use_auto_alpha,
                                use_emotion=use_emotion, use_ner=use_ner
                            )
                            
                            if use_auto_alpha:
                                st.write(f"⚖️ **[LinUCB]** 策略: Alpha={result['used_alpha']:.2f}")
                            if use_multiquery and len(result['generated_queries']) > 1:
                                st.write(f"💡 **[PRF]** 扩展查询: '{result['generated_queries'][1]}'")
                            if use_emotion:
                                emo = result['emotion']
                                if emo == 'negative': st.write(f"🧠 **[杏仁核]** ⚠️ 检测到焦虑 -> 启动共情安抚模式")
                                else: st.write(f"🧠 **[杏仁核]** 情绪平稳")

                            status.update(label="✅ 推理完成", state="complete", expanded=False)
                            
                            answer = result["answer"]
                            st.markdown(answer)

                            if use_ner and result['entities']:
                                tags_html = ""
                                for word, tag in result['entities']:
                                    color = "#fce8e6" if tag == 'LOC' else "#e8f0fe"
                                    tags_html += f"<span style='background:{color};padding:2px 6px;border-radius:4px;margin-right:4px;font-size:0.8em'>{word}<small>_{tag}</small></span>"
                                st.markdown(f"🏷️ **韦尼克区 (实体抽提)**: {tags_html}", unsafe_allow_html=True)
                            
                            with st.expander("🔍 原始记忆片段"):
                                for doc in result["context"]:
                                    st.info(doc.page_content)
                                    
                            st.session_state.messages.append({"role": "assistant", "content": answer})

                        except Exception as e:
                            st.error(str(e))
        else:
            st.warning("⚠️ 海马体为空。请先在左侧上传文档注入记忆。")