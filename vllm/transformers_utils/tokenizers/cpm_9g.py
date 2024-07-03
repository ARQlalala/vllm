import io
import json
import os
from shutil import copyfile
from typing import Any, Dict, IO, List, Optional, Tuple

import pkg_resources
import sentencepiece as spm
from pytrie import StringTrie
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {},
    "tokenizer_file": {},
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {}


class CPM9GTokenizer(PreTrainedTokenizer):
    """
    CPM9G 分词器类。用于基于字节对编码的分词。

    参数:
        path (str, 可选): 词汇表文件的路径。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: Optional[str] = None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        self.sp_model_kwargs = sp_model_kwargs or {}
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.byte_list: List[str] = (
            [f"<0x0{hex(i).upper()[2:]}>" for i in range(0x10)] +
            [f"<0x{hex(i).upper()[2:]}>" for i in range(0x10, 0x100)]
        )

        self._special_token_set = set([self.unk_token, self.bos_token, self.eos_token] + self.byte_list)

        if vocab_file:
            all_tokens = self.load_vocab(io.FileIO(vocab_file + VOCAB_FILES_NAMES['vocab_file'], "rb"))
        else:
            all_tokens = self.load_vocab(io.FileIO(vocab_files_names['vocab_file'], "rb"))

        self.encoder: Dict[str, int] = {}
        self._special_encoder: Dict[str, int] = {}
        for token, token_id in all_tokens.items():
            if token in self._special_token_set:
                self._special_encoder[token] = token_id
            else:
                self.encoder[token] = token_id

        self.decoder = {v: k for k, v in self.encoder.items()}
        self._byte_decoder = {self._special_encoder[token]: i for i, token in enumerate(self.byte_list)}

        self._max_word_len = max([len(x) for x in self.encoder.keys()])

        self._len_word_first = {}
        for x in self.encoder.keys():
            if not x[0] in self._len_word_first:
                self._len_word_first[x[0]] = 1
            if len(x) > self._len_word_first[x[0]]:
                self._len_word_first[x[0]] = len(x)
        self.tencoder = StringTrie(self.encoder)

        super().__init__(
            bos_token=AddedToken(bos_token, lstrip=False, rstrip=False),
            eos_token=AddedToken(eos_token, lstrip=False, rstrip=False),
            unk_token=AddedToken(unk_token, lstrip=False, rstrip=False),
            pad_token=AddedToken(pad_token, lstrip=False, rstrip=False) if pad_token else None,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d: Dict[str, Any]) -> None:
        self.__dict__ = d

    def load_vocab(self, fp: IO[bytes]) -> Dict[str, int]:
        """
        加载词汇表文件到字典中。

        参数:
            fp (IO[bytes]): 词汇表文件指针。

        返回:
            Dict[str, int]: 词汇表字典。
        """
        vocab: Dict[str, int] = {}
        reader = io.TextIOWrapper(fp, encoding="utf-8")
        for token in reader.readlines():
            token = token.strip()
            if len(token) == 0:
                continue
            token = json.loads(token)
            vocab[token] = len(vocab)
        return vocab

    @property
    def vocab_size(self) -> int:
        """返回词汇表大小"""
        return len(self.encoder) + len(self._special_encoder)

    @property
    def eos_id(self):
        return self._special_encoder[self.eos_token]

    @property
    def bos_id(self):
        return self._special_encoder[self.bos_token]

    @property
    def unk_id(self):
        return self._special_encoder[self.unk_token]

    def get_vocab(self) -> Dict[str, int]:
        """返回词汇表作为字典"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """返回分词后的字符串"""
        output_tokens: List[str] = []
        st = 0
        while st < len(text):
            piece = self.get_piece(text[st:])
            output_tokens.append(piece)
            st += len(piece)
        return output_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """使用词汇表将标记（字符串）转换为 id"""
        return self.encoder.get(token, self.unk_id)

    def _convert_id_to_token(self, index: int) -> str:
        """使用词汇表将索引（整数）转换为标记（字符串）"""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """将标记序列（字符串）转换为单个字符串"""
        current_sub_tokens: List[str] = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            if token in self._special_token_set:
                if not prev_is_special and i != 0:
                    out_string += " "
                out_string += self.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        保存词汇表和特殊标记文件到目录。

        参数:
            save_directory (str): 要保存词汇表的目录。

        返回:
            Tuple[str]: 保存的文件路径。
        """
        if not os.path.isdir(save_directory):
            raise ValueError(f"Vocabulary path ({save_directory}) should be a directory")

        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                fi.write(self.sp_model.serialized_model_proto())

        return (out_vocab_file, )

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None, 
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        获取从未添加特殊标记的标记列表中检索到的序列 id。
        在使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        参数:
            token_ids_0 (List[int]): id 列表。
            token_ids_1 (List[int], 可选): 序列对的可选第二 id 列表。
            already_has_special_tokens (bool, 可选, 默认值为 False): 
                标记列表是否已使用模型的特殊标记进行格式化。

        返回:
            List[int]: 一个包含整数（0 或 1）的列表。1 表示特殊标记，0 表示序列标记。
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id + bos_token_id + ([0] * len(token_ids_1)) + eos_token_id

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列创建掩码，用于序列对分类任务。

        参数:
            token_ids_0 (List[int]): id 列表。
            token_ids_1 (List[int], 可选): 序列对的可选第二 id 列表。

        返回:
            List[int]: 根据给定序列的标记类型 id 列表。
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output

    def get_piece(self, text: str) -> str:
        """
        获取文本中的分词片段。

        参数:
            text (str): 输入文本。

        返回:
            str: 分词片段。
        """
        if text[0] in self._len_word_first:
            text = text[: self._len_word_first[text[0]]]
            len_text = len(text)
            for i in range(len(text)):
                sub = text[: len_text - i]
                if sub in self.encoder:
                    return sub
        return text[0]


    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 ID 列表。

        参数:
            text (str): 输入文本。

        返回:
            List[int]: 编码后的 ID 列表。
        """
        #if len(text) > 20480:
        #    return [0 for _ in range(20480)]
        ret = []
        for x in self._tokenize(text):
            if x in self.encoder:
                ret.append(self.encoder[x])
            else:
                ret.extend(self._encode_unicode(x))
        return ret

    def decode(self, tokens: List[int]) -> str:
        """
        将 ID 列表解码为字符串。

        参数:
            tokens (List[int]): ID 列表。

        返回:
            str: 解码后的字符串。
        """
        ret = []
        st = 0

        while st < len(tokens):
            if tokens[st] in self._byte_decoder:
                if (
                    st + 3 < len(tokens)
                    and tokens[st + 1] in self._byte_decoder
                    and tokens[st + 2] in self._byte_decoder
                    and tokens[st + 3] in self._byte_decoder
                ):
                    first_id = self._byte_decoder[tokens[st]]
                    plane_id = self._byte_decoder[tokens[st + 1]]
                    row_id = self._byte_decoder[tokens[st + 2]]
                    cell_id = self._byte_decoder[tokens[st + 3]]
                    ret.append(
                        int.to_bytes(first_id << 24 | plane_id << 16 | row_id << 8 | cell_id, 4, "big").decode("utf-8")
                    )
                    st += 4
                elif (
                    st + 2 < len(tokens)
                    and tokens[st + 1] in self._byte_decoder
                    and tokens[st + 2] in self._byte_decoder
                ):
                    plane_id = self._byte_decoder[tokens[st]]
                    row_id = self._byte_decoder[tokens[st + 1]]
                    cell_id = self._byte_decoder[tokens[st + 2]]
                    ret.append(int.to_bytes(plane_id << 16 | row_id << 8 | cell_id, 3, "big").decode("utf-8"))
                    st += 3
                elif st + 1 < len(tokens) and tokens[st + 1] in self._byte_decoder:
                    row_id = self._byte_decoder[tokens[st]]
                    cell_id = self._byte_decoder[tokens[st + 1]]
                    ret.append(int.to_bytes(row_id << 8 | cell_id, 2, "big").decode("utf-8"))
                    st += 2
                else:
                    cell_id = self._byte_decoder[tokens[st]]
                    ret.append(int.to_bytes(cell_id, 1, "big").decode("utf-8"))
                    st += 1
            elif tokens[st] == self.eos_id:
                ret.append(self.eos_token)
                st += 1
            elif tokens[st] == self.bos_id:
                ret.append(self.bos_token)
                st += 1
            else:
                ret.append(tokens[st])
                st += 1
            #else:
            #    ret.append(self.unk_token)
            #    st += 1
        return ''.join(ret)

    def _encode_unicode(self, token: str) -> List[int]:
        """
        将 Unicode 编码包装到一个辅助函数中。

        参数:
            token (str): 要编码的标记。

        返回:
            List[int]: 编码后的 ID 列表。
        """
        ids = []
        utf8_id = token.encode("utf-8")
        for _id in utf8_id:
            ids.append(self._special_encoder[self.byte_list[_id]])
        return ids

    def next_token(self, text: str) -> Tuple[str, List[int]]:
        """
        快速获取下一个匹配的标记。

        参数:
            text (str): 输入文本。

        返回:
            Tuple[str, List[int]]: 匹配的标记及其 ID 列表。
        """
        token, token_id = self.tencoder.longest_prefix_item(text, (None, None))
        if token is None:
            token = text[0]
            token_ids = self._encode_unicode(token)
        else:
            token_ids = [token_id]
        return token, token_ids

