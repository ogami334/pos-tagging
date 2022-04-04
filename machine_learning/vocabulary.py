from collections import Counter
from pathlib import Path
from typing import List, Union

PADDING_TOKEN = "[PADDING]"
UNKNOWN_TOKEN = "[UNKNOWN]"


class Vocabulary:
    """
    This class handles token ←→ ID mappings.
    """

    def __init__(self, token_set: List[str], use_padding: bool = False, use_unknown: bool = False):

        if use_unknown:
            token_set = [UNKNOWN_TOKEN] + token_set

        if use_padding:
            token_set = [PADDING_TOKEN] + token_set

        self.use_padding = use_padding
        self.use_unknown = use_unknown

        self._index2token = token_set
        self._token2index = {t: i for i, t in enumerate(token_set)}

    def __len__(self):
        return len(self._index2token)

    def get_index_from_token(self, token: str) -> int:
        if token not in self._token2index:
            if self.use_unknown:
                token = UNKNOWN_TOKEN
            else:
                raise KeyError()
        return self._token2index[token]

    def get_token_from_id(self, index: int) -> str:
        return self._index2token[index]

    def get_indices_from_tokens(self, tokens: List[str]) -> List[int]:
        return [self.get_index_from_token(t) for t in tokens]

    def get_tokens_from_indices(self, indices: List[int]) -> List[str]:
        return [self.get_token_from_id(i) for i in indices]

    @classmethod
    def build_from_counter(
        cls,
        token_counter: Counter,
        min_num_tokens: int = 0,
        max_vocab_size: int = None,
        use_padding: bool = False,
        use_unknown: bool = False,
    ) -> "Vocabulary":
        tokens_in_vocab = [
            token for token, count in token_counter.most_common(n=max_vocab_size) if count > min_num_tokens
        ]
        return Vocabulary(tokens_in_vocab, use_padding=use_padding, use_unknown=use_unknown)

    def save(self, save_path: Union[str, Path]):
        with open(save_path, "w") as f:
            f.write("\n".join(self._index2token))

    @classmethod
    def load(cls, save_path: Union[str, Path]) -> "Vocabulary":
        with open(save_path, "r") as f:
            token_set = f.read().strip().split("\n")

        return Vocabulary(token_set, use_padding=PADDING_TOKEN in token_set, use_unknown=UNKNOWN_TOKEN in token_set)
