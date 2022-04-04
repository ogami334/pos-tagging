import sys
from pathlib import Path

import click

sys.path.append("./")
from pos_tagging.read_wsj_data import parse_wsj_file


@click.command()
@click.argument("input-file-path", type=click.Path(exists=True))
def make_word_only_file(input_file_path: str):
    output_directory = Path(input_file_path).parent
    output_filename = Path(input_file_path).stem + ".txt"

    with open(output_directory / output_filename, "w") as f:
        for data in parse_wsj_file(input_file_path):
            f.write(" ".join(data["words"]))
            f.write("\n")


if __name__ == "__main__":
    make_word_only_file()
