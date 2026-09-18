"""
文本处理工具
"""

import re


class TextProcessor:
    """文本分割、清洗工具"""

    SENTENCE_DELIMITERS = re.compile(r"(?<=[。！？.!?])")

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """按句末标点分割句子"""
        raw = TextProcessor.SENTENCE_DELIMITERS.split(text)
        sentences = [s.strip() for s in raw if s.strip()]
        return sentences

    @staticmethod
    def split_chunks(text: str, max_chars: int = 512, overlap: int = 32) -> list[str]:
        """
        将文本切分为重叠块，保证语义连续性。
        """
        sentences = TextProcessor.split_sentences(text)
        chunks: list[str] = []
        buffer: list[str] = []
        buf_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if buf_len + sent_len > max_chars and buffer:
                chunks.append("".join(buffer))
                # 保留 overlap 个字符用于重叠
                overlap_text = "".join(buffer)
                # 找到重叠起点
                overlap_start = max(0, len(overlap_text) - overlap)
                overlap_sents = TextProcessor.split_sentences(overlap_text[overlap_start:])
                buffer = overlap_sents if overlap_sents else []
                buf_len = sum(len(s) for s in buffer)
            buffer.append(sent)
            buf_len += sent_len

        if buffer:
            chunks.append("".join(buffer))
        return chunks

    @staticmethod
    def clean_text(text: str) -> str:
        """清理多余空白"""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()