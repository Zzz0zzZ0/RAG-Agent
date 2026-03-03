import os
import shutil
import pandas as pd
from sklearn.decomposition import PCA

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

# 导入核心配置
from core.config import EMBED_MODEL_NAME

print("🗂️ [Memory] 正在激活海马体 (Embedding Models)...")

# 全局初始化 Embedding 模型
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

def build_hybrid_knowledge_base(file_path):
    """构建 Chroma + BM25 混合索引"""
    persist_dir = "./chroma_db_data"
    if os.path.exists(persist_dir):
        try: shutil.rmtree(persist_dir)
        except: pass

    if file_path.endswith('.pdf'): 
        loader = PyPDFLoader(file_path)
    else: 
        loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 10 
    
    return vectorstore, bm25_retriever, len(splits)

def visualize_semantic_space(vectorstore):
    """降维可视化语义空间"""
    data = vectorstore.get(include=['embeddings', 'documents'])
    if len(data['embeddings']) < 3: return None
    reducer = PCA(n_components=2)
    vecs_2d = reducer.fit_transform(data['embeddings'])
    df = pd.DataFrame(vecs_2d, columns=['x', 'y'])
    df['text'] = [t[:50] + "..." for t in data['documents']]
    return df