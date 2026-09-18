"""
关键句提取器 —— 从对话文本中提取代表性关键句

支持两种模式:
  1. 基于位置/统计的启发式提取（轻量，无需 LLM）
  2. 预留 LLM 接口（可对接本地模型进行语义提取）
"""

import re
from typing import Optional

from ..config import HConfig
from .text_processor import TextProcessor


class KeySentenceExtractor:
    """
    从文本块中提取关键句。

    启发式策略:
      - 包含"总结/关键/重点/重要的是/核心/问题/方案/结论/决定/最终"等词的句子优先
      - 较长句子（含有较多信息量）优先
      - 首尾句（话题引入/总结）优先
    """

    # 信号词权重
    SIGNAL_WORDS = {
        "总结": 3, "综上所述": 3, "关键": 3, "重点": 3,
        "核心": 3, "重要的是": 3, "关键在于": 3,
        "问题": 2, "方案": 2, "结论": 2, "决定": 2,
        "最终": 2, "建议": 2, "需要": 2, "必须": 2,
        "第一": 1, "第二": 1, "首先": 1, "其次": 1, "最后": 1,
        "因此": 1, "所以": 1, "但是": 1, "然而": 1,
        "summary": 3, "key": 3, "important": 3, "critical": 3,
        "conclusion": 2, "result": 2, "proposal": 2,
        "first": 1, "second": 1, "finally": 1,
    }

    def __init__(self, config: Optional[HConfig] = None):
        self.config = config or HConfig()

    def extract(self, text: str, max_sentences: Optional[int] = None) -> list[str]:
        """
        从文本中提取关键句。

        Args:
            text: 源文本
            max_sentences: 最多提取句数，默认使用 config.max_key_sentences_per_chunk

        Returns:
            关键句列表（按重要性降序）
        """
        max_s = max_sentences or self.config.max_key_sentences_per_chunk
        sentences = TextProcessor.split_sentences(text)
        if len(sentences) <= max_s:
            return sentences

        scored = []
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, i, len(sentences))
            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:max_s]]

    def _score_sentence(self, sentence: str, idx: int, total: int) -> float:
        """为单句打分"""
        score = 0.0
        s_lower = sentence.lower()

        # 信号词加分
        for word, weight in self.SIGNAL_WORDS.items():
            if word in sentence or word in s_lower:
                score += weight

        # 长度分（过短或过长都不是好关键句）
        length = len(sentence)
        if 20 <= length <= self.config.key_sentence_max_length:
            score += 1.0
        elif length < 10:
            score -= 0.5

        # 首尾句加分
        if idx == 0:
            score += 1.5
        elif idx == total - 1:
            score += 1.0

        # 疑问句/反问句扣分
        if sentence.strip().endswith(("?", "？")):
            score -= 0.5

        return score