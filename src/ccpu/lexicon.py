"""Keyword lexicons.

The paper's classifier is MacBERT, not a dictionary. Dictionaries are used here
for two auxiliary jobs only:

1. ``CLIMATE``: a high-recall candidate filter. Articles with no climate term
   are assigned CPU = 0 without running the transformer, which makes full-corpus
   inference feasible on a CPU. Filter recall is checked on a random sample of
   non-candidate articles (see docs/ANNOTATION_GUIDE.md).
2. ``POLICY`` and ``UNCERTAINTY``: a Baker-Bloom-Davis style rule baseline
   (climate AND policy AND uncertainty), similar in spirit to the dictionary
   indices the paper compares against (Xu et al., 2023; Lin and Zhao, 2023).
"""
import re

CLIMATE = (
    "气候变化", "气候变暖", "全球变暖", "全球暖化", "应对气候", "气候谈判", "气候大会", "气候峰会",
    "气候协定", "气候协议", "气候公约", "气候政策", "气候治理", "气候行动", "气候目标", "气候融资",
    "气候资金", "气候危机", "气候问题", "气候雄心", "气候外交", "气候合作", "气候适应", "气候风险",
    "气候立法", "气候法", "联合国气候", "政府间气候变化专门委员会", "IPCC",
    "温室气体", "温室效应", "碳排放", "二氧化碳排放", "排放二氧化碳", "碳排量", "排碳",
    "减排", "节能减排", "低碳", "去碳", "脱碳", "零碳", "负碳", "碳达峰", "碳中和", "双碳",
    "碳交易", "碳市场", "碳排放权", "排放权交易", "碳配额", "碳税", "碳关税", "碳边境",
    "碳汇", "碳足迹", "碳金融", "碳价", "碳强度", "碳捕集", "碳封存", "CCUS", "CDM",
    "清洁发展机制", "京都议定书", "巴黎协定", "哥本哈根", "坎昆会议", "德班会议", "多哈会议",
    "华沙会议", "利马会议", "格拉斯哥", "COP26", "COP27", "COP15", "缔约方大会",
)

POLICY = (
    "政策", "政府", "国务院", "发改委", "发展改革委", "生态环境部", "环保部", "环境保护部",
    "部委", "监管", "法规", "法律", "条例", "规定", "规划", "立法", "措施", "制度", "目标",
    "协议", "协定", "谈判", "议定书", "公约", "配额", "税", "补贴", "试点", "行动方案",
)

UNCERTAINTY = (
    "不确定", "不明确", "不明朗", "不稳定", "未知", "未明", "难以预料", "难以预测", "难以估计",
    "不可预测", "不可预料", "变数", "风险", "波动", "动荡", "前景不明", "悬而未决", "分歧",
    "争议", "僵局", "可能", "或将", "也许", "存疑", "担忧", "挑战", "难题", "困境", "压力",
)


def _pattern(terms):
    return re.compile("|".join(sorted(map(re.escape, terms), key=len, reverse=True)), re.IGNORECASE)


CLIMATE_RE = _pattern(CLIMATE)
POLICY_RE = _pattern(POLICY)
UNCERTAINTY_RE = _pattern(UNCERTAINTY)


def climate_hits(text):
    return [m.group(0) for m in CLIMATE_RE.finditer(text)]


def rule_label(text, min_climate=1):
    """Dictionary baseline: climate AND policy AND uncertainty term present."""
    return int(len(CLIMATE_RE.findall(text)) >= min_climate
               and POLICY_RE.search(text) is not None
               and UNCERTAINTY_RE.search(text) is not None)


def focus_window(title, text, lead=100, window=60, budget=600):
    """Model/annotation input: title, the lead, then passages around climate terms.

    Chinese BERT tokenizes roughly one character per token, so a 256-token model
    sees about 250 characters. Long articles often mention climate policy far
    below the lead; the windows bring those passages forward. Annotators read
    the full ``budget`` characters; the classifier reads the first 256 tokens of
    the same string (as in the paper, humans see more context than the model).
    """
    head = (title + "。" if title else "") + text[:lead]
    parts, used, last_end = [head], len(head), lead
    for m in CLIMATE_RE.finditer(text, lead):
        if used >= budget:
            break
        start, end = max(last_end, m.start() - window), min(len(text), m.end() + window)
        if end <= last_end:
            continue
        piece = text[start:end]
        parts.append("……" + piece)
        used += len(piece) + 2
        last_end = end
    return "".join(parts)[:budget]
