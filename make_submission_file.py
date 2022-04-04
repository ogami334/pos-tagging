import logging
from pathlib import Path

import click

from machine_learning.ml_pos_tagger import MLPoSTagger
from machine_learning.utils.import_util import import_submodules
from pos_tagging.read_wsj_data import SEPARATOR, parse_word_only_file

logger = logging.getLogger(__name__)
fmt = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=fmt)


@click.command()
@click.argument("result-save-directory", type=click.Path(exists=False))
@click.argument("input-file-path", type=click.Path(exists=True), default="data/wsj/wsj22-24.txt")
def make_submission_file(result_save_directory: str, input_file_path: str):
    import_submodules("machine_learning")

    logger.info(f"Load MLPoSTagger from {result_save_directory}")
    pos_tagger = MLPoSTagger.load(result_save_directory)

    output_file_name = Path(input_file_path).stem + ".prediction.pos"
    output_file_path = Path(result_save_directory) / output_file_name
    logger.info(f"Make prediction with {input_file_path}")
    with open(output_file_path, "w") as f:
        for words in parse_word_only_file(input_file_path):
            predicted_tags = pos_tagger.tag_words(words)
            word_tag_list = [SEPARATOR.join([w, t]) for w, t in zip(words, predicted_tags)]
            f.write(" ".join(word_tag_list))
            f.write("\n")
    logger.info(f"Saved the prediction in {output_file_path}")


if __name__ == "__main__":
    make_submission_file()
