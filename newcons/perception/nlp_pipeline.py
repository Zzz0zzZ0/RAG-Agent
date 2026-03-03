from transformers import pipeline

print("🧠 [Perception] 正在激活 NLP 边缘系统 (杏仁核 & 韦尼克区)...")

# --- 1. 初始化本地 NLP 模型 ---
# 杏仁核：情感分析
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="uer/roberta-base-finetuned-dianping-chinese"
)

# 韦尼克区：命名实体识别 (NER)
ner_tagger = pipeline(
    "ner", 
    model="uer/roberta-base-finetuned-cluener2020-chinese", 
    aggregation_strategy="simple"
)

# --- 2. 封装感知接口 ---
def analyze_emotion(text: str):
    """杏仁核接口：返回情感标签和置信度得分"""
    try:
        res = sentiment_analyzer(text[:512])[0]
        return res['label'], res['score']
    except Exception as e:
        print(f"情感分析异常: {e}")
        return "neutral", 0.5

def extract_entities(text: str):
    """韦尼克区接口：返回提取到的实体列表[(实体词, 实体类型), ...]"""
    try:
        res = ner_tagger(text[:512])
        return [(e['word'], e['entity_group']) for e in res]
    except Exception as e:
        print(f"实体识别异常: {e}")
        return