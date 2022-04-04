from typing import Dict, Iterator


def parse_wsj_file(file_path: str, separator: str = "/") -> Iterator[Dict[str, str]]:
    with open(file_path, "r") as f:
        for tagged_sentence in f:
            tokens_with_tag = tagged_sentence.strip().split()
            tokens_with_tag = [token_pos.rsplit(separator, maxsplit=1) for token_pos in tokens_with_tag]
            words, tags = map(list, zip(*tokens_with_tag))
            assert len(words) == len(tags)
            yield {"words": words, "tags": tags}
