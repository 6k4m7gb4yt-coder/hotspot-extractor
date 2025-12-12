import math
import re
import sys
from collections import Counter
import requests  # 用于调用阿里云百炼 API

# 确保安装了所需的库
try:
    import jieba
    import pandas as pd
    import streamlit as st
except ImportError:
    # 引导用户安装依赖
    print("错误：请先安装所需的库。运行命令：pip3 install jieba pandas streamlit")
    sys.exit(1)


# ===== 1. 配置：要过滤掉的类目 & 噪音词 =====

STOP_CATEGORIES = {
    "影视娱乐", "体育赛事", "家居家装", "旅游出行", "美妆时尚",
    "社会热点", "游戏竞技", "其他", "热门榜单",
}

PLATFORM_NUM_PATTERNS = [
    re.compile(r"小红书\s*[\d,\.]+"),
    re.compile(r"抖音\s*[\d,\.]+"),
    re.compile(r"快手\s*[\d,\.]+"),
]

GENERIC_KEYS = [
    "热梗", "热点", "流行", "趋势", "话题", "文化", "营销", "策略", "运营", "内容",
    "品牌", "用户", "社交", "社媒", "新媒体", "整合营销", "传播", "短视频",
    "系列", "我们", "他们", "大家", "很多人", "年轻人"
]

GENERIC_EXACT = set([
    "热梗流行", "网络热梗", "流行词", "热门趋势", "热点趋势", "社交趋势",
    "整合营销策略", "社媒整合营销", "网络热点",
])

# ===== 新增：人名 & 国家名 过滤配置 =====

# 常见中文姓氏（覆盖绝大部分 2~3 字人名）
COMMON_SURNAMES = set(list(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张"
    "孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎"
    "鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤"
    "滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄"
    "和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞"
    "熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭"
    "梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫"
    "房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程"
    "嵇邢滑裴陆荣翁荀羊於惠甄奚桑桂濮牛寿通边扈燕冀郏"
    "浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈"
    "廖庚终暨居衡步都耿满弘匡国文寇广禄阙东殴殳沃利蔚"
    "越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养"
    "鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公万俟司马上"
    "欧阳夏侯诸葛闻人东方赫连皇甫尉迟公羊澹台公冶宗政"
    "濮阳淳于单于太叔申屠公孙仲孙轩辕令狐钟离宇文长孙"
    "慕容鲜于闾丘司徒司空"
))

# 常见国家 / 地缘词（你可以以后自己往里加）
COUNTRY_WORDS = [
    "中国", "大陆", "内地", "港澳台",
    "美国", "英", "英国", "法国", "德", "德国",
    "日本", "日韩", "韩国", "朝鲜",
    "俄罗斯", "俄国", "苏联", "乌克兰",
    "印度", "越南", "泰国", "新加坡", "马来西亚", "印尼", "菲律宾",
    "澳大利亚", "加拿大", "墨西哥", "巴西", "阿根廷",
    "中东", "以色列", "巴勒斯坦", "加沙",
    "非洲", "欧洲", "拉美", "亚太"
]


def looks_like_noise(term: str) -> bool:
    """判断一个候选短语是不是噪音"""
    t = term.strip()
    if not t:
        return True

    t_compressed = re.sub(r"\s+", "", t)

    if t_compressed in GENERIC_EXACT:
        return True
    if t_compressed in STOP_CATEGORIES:
        return True

    # 包含典型泛概念词且长度较短的，视为噪音
    for key in GENERIC_KEYS:
        if key in t_compressed and len(t_compressed) < len(key) + 3:
            return True

    if re.fullmatch(r"[0-9,\.]+", t_compressed):
        return True

    if len(t_compressed) <= 2 and not re.search(r"[0-9A-Za-z]", t_compressed):
        return True

    return False


# ===== 新增：人名 / 国家词 过滤函数 =====

def looks_like_person_name(text: str) -> bool:
    """
    粗略判断是否像中文人名：
    - 去掉空白后长度为 2 或 3
    - 全是中文
    - 第一个字是常见姓氏
    （会有少量误杀，但对你“不要人名梗”的目标是 ok 的）
    """
    if not text:
        return False
    t = re.sub(r"\s+", "", text)
    # 只考虑较短的 2~3 字串
    if len(t) not in (2, 3):
        return False
    # 必须全是中文
    if not all('\u4e00' <= ch <= '\u9fff' for ch in t):
        return False
    # 第一个字是常见姓氏
    return t[0] in COMMON_SURNAMES


def contains_country_or_name(text: str) -> bool:
    """是否包含国家名或看起来像人名"""
    if not text:
        return False
    t = re.sub(r"\s+", "", text)

    # 国家 / 地缘关键词
    for w in COUNTRY_WORDS:
        if w and w in t:
            return True

    # 像人名
    if looks_like_person_name(t):
        return True

    return False


# ===== 1.5 规则分类（天气 / 家人 / 打工人 / 宠物 / 生活方式） =====

CATEGORY_RULES = [
    {
        "name": "季节/天气/温度梗",
        "short": "天气温度",
        "keywords": [
            "冷空气", "降温", "入冬", "一夜换季", "换季", "暖冬", "回暖",
            "早晚冷", "中午热", "下雪", "雨夹雪", "回南天", "湿冷", "干冷",
            "风大", "暴雨", "高温", "热浪", "空调", "暖气", "地暖",
            "温度", "体感", "羽绒服", "秋裤", "棉袄", "短袖", "穿衣"
        ]
    },
    {
        "name": "家里人/家务日常梗",
        "short": "家人家务",
        "keywords": [
            "孩子", "宝宝", "娃", "熊孩子", "写作业", "寒假", "放假在家",
            "妈妈", "妈", "老妈", "爸爸", "爸", "老爹", "父母", "公婆",
            "在家", "宅家", "回家", "下班回家", "家里", "全家",
            "洗澡", "洗头", "洗衣服", "晾衣服", "家务", "做饭",
            "嫌冷", "嫌热", "嫌潮", "嫌味"
        ]
    },
    {
        "name": "打工人/城市生活梗",
        "short": "打工人",
        "keywords": [
            "打工人", "上班", "下班", "通勤", "早八", "晚八", "加班",
            "工位", "办公室", "工牌", "打卡", "牛马", "社畜",
            "月薪", "工资", "社保", "公积金", "地铁", "城巴", "城巴佬",
            "我的工作流程", "流程belike", "流程 belike"
        ]
    },
    {
        "name": "宠物梗（猫猫狗狗）",
        "short": "宠物",
        "keywords": [
            "猫", "猫猫", "狗", "狗狗", "小狗", "小猫", "主子",
            "铲屎官", "宠物", "猫毛", "狗毛", "掉毛", "猫窝", "猫砂",
            "汪", "喵"
        ]
    },
    {
        "name": "生活方式/审美趋势梗",
        "short": "生活方式",
        "keywords": [
            "国风", "国潮", "新中式", "唐风", "新中式家", "家装",
            "改造", "装修", "变废为宝", "旧物改造",
            "运动打卡", "打卡", "100天", "一百天", "挑战",
            "健身", "跑步", "骑行", "健康", "减脂", "养生",
            "极简", "断舍离", "美拉德", "多巴胺穿搭"
        ]
    }
]


def classify_example_text(text: str) -> str:
    """
    规则粗分：根据示例文案打标签。
    - 命中多个时，用 '、' 拼接
    - 都没命中则标记为 '未分类/其他'
    """
    if not text:
        return "未分类/其他"

    text_norm = re.sub(r"\s+", "", text)
    hits = []

    for rule in CATEGORY_RULES:
        for kw in rule["keywords"]:
            if kw in text_norm:
                hits.append(rule["short"])
                break

    if not hits:
        return "未分类/其他"

    return "、".join(hits)


# ===== 2. 预处理：解决混合语言分词 & 清洗元数据 =====

def smart_tokenize_for_jieba(text: str) -> str:
    """
    【修正混合语言的关键步骤】: 确保连续的英文/数字/符号被 Jieba 识别为一个 Token。
    """
    tokens = re.findall(r'[\u4e00-\u9fa5]+|[^\u4e00-\u9fa5]+', text)
    return " ".join(tokens).strip()


def clean_meta_fields(line: str) -> str:
    """去掉类目字段和热度数字"""
    if not line:
        return ""

    text = line.strip()

    if text in STOP_CATEGORIES:
        return ""
    for cat in STOP_CATEGORIES:
        text = text.replace(cat, "")

    for pat in PLATFORM_NUM_PATTERNS:
        text = pat.sub("", text)

    return text.strip()


def preprocess_docs(raw_text: str):
    """把输入的大文本，拆成“文案列表 docs”，并预清洗"""
    docs = []
    for line in raw_text.splitlines():
        line = clean_meta_fields(line)
        if not line:
            continue
        if len(line) < 3:
            continue
        docs.append(line)
    return docs


# ===== 3. PMI 计算 & 短语提取 (集成 A 级权重) =====

def build_pmi_and_doc_phrases(docs,
                              min_freq: int,
                              min_len: int,
                              max_len: int,
                              weight: int):
    """计算 PMI，并进行短语提取"""

    tokenized_docs = []
    for doc in docs:
        preprocessed_doc = smart_tokenize_for_jieba(doc)
        tokens = [t.strip() for t in jieba.lcut(preprocessed_doc) if t.strip()]
        tokenized_docs.append(tokens)

    # 统计词频并应用权重
    unigram = Counter()
    bigram = Counter()

    for tokens in tokenized_docs:
        unigram.update({t: weight for t in tokens})
        for i in range(len(tokens) - 1):
            bg = (tokens[i], tokens[i + 1])
            bigram.update({bg: weight})

    total_unigrams = sum(unigram.values()) or 1
    total_bigrams = sum(bigram.values()) or 1

    # 计算所有 bi-gram 的 PMI
    phrase_pmi = {}
    for (w1, w2), c12 in bigram.items():
        phrase = w1 + w2

        # 频率过滤 (注意 c12 是加权计数)
        if c12 < min_freq * weight:
            continue

        if looks_like_noise(phrase):
            continue
        if not (min_len <= len(phrase) <= max_len):
            continue

        c1 = unigram[w1]
        c2 = unigram[w2]
        if c1 <= 0 or c2 <= 0:
            continue

        # PMI 计算
        p12 = c12 / total_bigrams
        p1 = c1 / total_unigrams
        p2 = c2 / total_unigrams

        pmi = math.log((p12 / (p1 * p2 + 1e-9)) + 1e-9, 2)

        if pmi > 3.0:
            phrase_pmi[phrase] = pmi

    # 每条文案只选 1 个 PMI 最高的短语
    phrase_doc_count = Counter()
    phrase_example = {}

    for idx, tokens in enumerate(tokenized_docs):
        best_phrase = None
        best_pmi = -1e9

        for i in range(len(tokens) - 1):
            phrase = tokens[i] + tokens[i + 1]
            pmi = phrase_pmi.get(phrase, -1)

            if pmi > best_pmi:
                best_pmi = pmi
                best_phrase = phrase

        if best_phrase:
            phrase_doc_count[best_phrase] += 1
            phrase_example.setdefault(best_phrase, docs[idx])

    return phrase_pmi, phrase_doc_count, phrase_example


def build_result_df(phrase_pmi,
                    phrase_doc_count,
                    phrase_example,
                    top_k: int):
    """
    构建结果表：
    - 在这里做“最终结果过滤”：剔除含人名 / 国家名的短语
    """
    rows = []
    for phrase, freq in phrase_doc_count.most_common(top_k):
        example = phrase_example.get(phrase, "")

        # === 关键新增：过滤人名 / 国家名 ===
        if contains_country_or_name(phrase) or contains_country_or_name(example):
            # 直接跳过，不进入最终结果
            continue

        category = classify_example_text(example)
        rows.append({
            "短语": phrase,
            "文案频次": freq,
            "PMI凝固度": round(phrase_pmi.get(phrase, 0.0), 2),
            "字符长度": len(phrase),
            "示例文本": example,
            "主题分类": category
        })
    df = pd.DataFrame(
        rows,
        columns=["短语", "文案频次", "PMI凝固度", "字符长度", "示例文本", "主题分类"]
    )
    return df


# ===== 3.5 阿里云百炼 LLM 调用：可选深度分析 =====

def llm_analyze_phrase(api_key: str, phrase: str, example_text: str) -> str:
    """
    调用阿里云百炼的通义模型，对单个热梗做深度分析。
    """
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    prompt = f"""
你是一个短视频&内容营销的热梗分析专家，请帮助我从运营视角读懂一个梗。

【短语】：
{phrase}

【示例文案】：
{example_text}

请按以下结构输出（中文）：

1. 梗场景类型（用你自己的话命名，例如：冷空气突袭、打工人下班崩溃、妈味家务、宠物当小孩、自我奖励消费等）
2. 梗的真实含义（1-2 句）
3. 典型触发场景（什么人、在什么时刻/情绪下会说这句话）
4. 对家电品牌的适配建议（适合哪些品类？如：空调/热水器/洗衣机/冰箱/烘干机/取暖器等，并给出理由）
5. 3 条可直接参考的创意玩法文案（适合做标题/口播/脚本的短句）

补充说明：
- 你可以在结尾简单说明：这个梗大致接近哪一类（冷空气/天气、家里人/家务、打工人/城市、宠物、生活方式/审美），或者说明“不在这些里”，看你的判断。
- 不要解释你在做什么，直接给出条目。
"""

    payload = {
        "model": "qwen-plus",  # 先用性价比高的 plus
        "input": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    data = resp.json()

    # 尝试从标准字段取内容
    try:
        return data["output"]["text"]
    except Exception:
        # 出错时原样返回，方便排查
        return str(data)


# ===== 4. Streamlit 页面 (UI) =====

def main():
    st.set_page_config(page_title="本地热梗提取工具 (PMI 趋势版 + LLM 深度分析)",
                       layout="wide")

    st.title("📌 本地热门趋势 / 热梗提取工具（PMI 增强版 + 通义 LLM 深度分析）")

    with st.sidebar:
        st.header("⚙️ 分析参数设置")

        weight = st.slider("数据权重 W (A级数据模拟)",
                           min_value=1,
                           max_value=10,
                           value=5,
                           help="提高权重模拟高价值/权威数据源。权重越高，低频优质梗越容易入选。")

        min_freq = st.slider("最小文案出现次数",
                             min_value=1,
                             max_value=20,
                             value=2)

        st.markdown(f"**最低有效频次 (加权后):** **{min_freq * weight}** (实际计算次数)")
        st.markdown("---")

        top_k = st.slider("展示 TopK 数量", min_value=10, max_value=200, value=50)
        min_len = st.slider("最小字符长度", min_value=2, max_value=4, value=2)
        max_len = st.slider("最大字符长度", min_value=3, max_value=8, value=5)

    st.markdown("### 📥 粘贴新的原始文案数据")
    st.info("💡 **操作说明:** 将多条文案直接粘贴到下方，**每条文案独占一行**。程序将自动过滤类目、数字和噪音，并修正混合语言分词。")

    default_text = """
老式过冬是一种趋势
今天冷空气有点抽象
下班回家第一件事就是开空调
打工人的冬天流程belike：下班-洗个热水澡-钻被窝
我妈嫌冷让我别洗头
猫猫冬天也要有自己的小被窝
100天冬季运动打卡今天算第3天
"""

    raw_text = st.text_area(
        "",
        value=default_text,
        height=300,
        placeholder="每行一条文案，可以直接从 Excel 复制多列粘过来，程序会自动清洗和分析。"
    )

    df = None

    if st.button("🚀 第一步：提取热门趋势 / 热梗（本地 PMI，不耗 token）", use_container_width=True):
        if not raw_text.strip():
            st.warning("请先粘贴一些文案再点击按钮。")
            return

        with st.spinner('正在进行智能分词与 PMI 加权计算...'):
            docs = preprocess_docs(raw_text)
            if not docs:
                st.warning("有效文案为空，可能都被当成类目/噪音过滤掉了。")
                return

            phrase_pmi, phrase_doc_count, phrase_example = build_pmi_and_doc_phrases(
                docs,
                min_freq=min_freq,
                min_len=min_len,
                max_len=max_len,
                weight=weight  # 传入权重
            )

            df = build_result_df(
                phrase_pmi,
                phrase_doc_count,
                phrase_example,
                top_k=top_k,
            )

        st.success("提取完成：已应用加权分析、修正混合语言分词，并完成规则场景分类。"
                   "（已自动剔除含人名 / 国家名的短语）")

        st.subheader("✅ 候选热门趋势 / 热梗结果（本地分析）")
        st.dataframe(df, use_container_width=True)

        # 简单看一下主题分布
        if not df.empty:
            st.markdown("#### 📊 规则分类下的主题分布（参考用）")
            category_counts = df["主题分类"].value_counts().reset_index()
            category_counts.columns = ["主题分类", "短语数量"]
            st.table(category_counts)

        # 把 df 存到 session_state，方便后面 LLM 用
        st.session_state["last_df"] = df

    # ===== 第二步：可选 LLM 深度分析 =====
    st.markdown("---")
    st.subheader("✨ 第二步（可选）：使用通义大模型进行深度语义分析（按需调用，节约 token）")

    enable_llm = st.checkbox("开启通义大模型分析（仅对前 N 个短语调用）")

    if enable_llm:
        api_key = st.text_input(
            "请输入你的阿里云百炼 API-Key（不会被保存）：",
            type="password",
            help="在百炼控制台的「密钥管理」里可以找到以 sk- 开头的 API Key。"
        )

        last_df = st.session_state.get("last_df", None)
        if last_df is None or last_df.empty:
            st.info("请先完成上面的『第一步：本地 PMI 提取』，再进行大模型分析。")
            return

        max_rows = st.slider(
            "选择要用 LLM 深度分析的梗数量（按当前排序的前 N 条）：",
            min_value=5,
            max_value=min(50, len(last_df)),
            value=min(20, len(last_df))
        )

        if st.button("🚀 开始大模型分析（按需消耗 tokens）", use_container_width=True):
            if not api_key:
                st.warning("请先输入 API-Key。")
                return

            target_df = last_df.head(max_rows).copy()
            llm_results = []

            for idx, row in target_df.iterrows():
                phrase = row["短语"]
                example = row["示例文本"]

                with st.spinner(f"正在分析：{phrase} ..."):
                    analysis = llm_analyze_phrase(api_key, phrase, example)

                llm_results.append(analysis)

            target_df["LLM深度洞察"] = llm_results

            st.success("大模型分析完成 ✅（仅对前 N 条进行了处理）")

            st.subheader("📊 含大模型洞察的结果（Top N）")
            st.dataframe(
                target_df[
                    ["短语", "文案频次", "PMI凝固度", "字符长度", "示例文本", "主题分类", "LLM深度洞察"]
                ],
                use_container_width=True
            )


if __name__ == "__main__":
    main()
