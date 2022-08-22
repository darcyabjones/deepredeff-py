#!/usr/bin/env python3

from typing import List, Literal, Dict, Callable, Sequence

import sys
import argparse

import pandas as pd
import numpy as np
from tensorflow.keras import Model
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from . import __version__
from .exitcodes import (
    EXIT_VALID, EXIT_KEYBOARD, EXIT_UNKNOWN, EXIT_CLI, EXIT_INPUT_FORMAT,
    EXIT_INPUT_NOT_FOUND, EXIT_SYSERR, EXIT_CANT_OUTPUT
)


class MyArgumentParser(argparse.ArgumentParser):

    def error(self, message: str):
        """ Override default to have more informative exit codes. """
        self.print_usage(sys.stderr)
        raise MyArgumentError("{}: error: {}".format(self.prog, message))


class MyArgumentError(Exception):

    def __init__(self, message: str):
        self.message = message
        self.errno = EXIT_CLI

        # This is a bit hacky, but I can't figure out another way to do it.
        if "No such file or directory" in message:
            if "infile" in message:
                self.errno = EXIT_INPUT_NOT_FOUND
            elif "outfile" in message:
                self.errno = EXIT_CANT_OUTPUT
        return


def cli(prog: str, args: List[str]) -> argparse.Namespace:

    parser = MyArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Examples:\n\n"
            "```bash\n"
            "$ %(prog)s -o predictions.tsv my.fasta\n"
            "```\n"
        ),
        epilog=(
            "Exit codes:\n\n"
            f"{EXIT_VALID} - Everything's fine\n"
            f"{EXIT_KEYBOARD} - Keyboard interrupt\n"
            f"{EXIT_CLI} - Invalid command line usage\n"
            f"{EXIT_INPUT_FORMAT} - Input format error\n"
            f"{EXIT_INPUT_NOT_FOUND} - Cannot open the input\n"
            f"{EXIT_SYSERR} - System error\n"
            f"{EXIT_CANT_OUTPUT} - Can't create output file\n"
            f"{EXIT_UNKNOWN} - Unhandled exception, please file a bug!\n"
        )
    )

    parser.add_argument(
        "infile",
        type=argparse.FileType('r'),
        help=(
            "Path to the input FASTA file."
        )
    )

    parser.add_argument(
        "-o", "--outfile",
        default=sys.stdout,
        type=argparse.FileType('w'),
        help=(
            "File path to write tab delimited output to. Default is STDOUT."
        )
    )

    parser.add_argument(
        "-t", "--taxon",
        default="fungi",
        type=str,
        choices=["bacteria", "oomycete", "fungi"],
        help=(
            "The deepredeff model to use."
        )
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s {}'.format(__version__),
        help="Print the version of %(prog)s and exit"
    )

    return parser.parse_args()


def ensemble_weighted(
    preds: pd.DataFrame,
    weights: Dict[str, float]
) -> pd.Series:
    """ Gets the weighted average of an ensemble predictor """
    order = list(weights.keys())
    weights_ordered = np.array([weights[k] for k in order])
    preds = preds.loc[:, order]
    weighted = weights_ordered * preds / weights_ordered.sum()
    return weighted.sum(axis=1)


def get_ensemble_method(
    taxon: Literal["bacteria", "fungi", "oomycete"]
) -> Callable[[pd.DataFrame], float]:
    from .data import BACTERIA_WEIGHTS
    if taxon == "bacteria":
        return lambda x: ensemble_weighted(x, BACTERIA_WEIGHTS)
    else:
        return lambda x: x.iloc[:, 0]


def encode_one_hot(sequence: SeqRecord, max_length: int = 4034) -> np.ndarray:
    """ The implementation in deepredeff relies on extra characters mapping
    to NULL. Then when assigning the value of the matrix nothing gets filled.
    So AAs with non-standard AAs get all zeros.
    """
    keys = dict(zip("RKDEQNHSTYCWAILMFVPG", range(1, 21)))
    seq_ints = np.array([
        keys.get(aa.upper(), 0)
        for aa
        in sequence.seq[:max_length]
    ])

    encoded_sequence = np.zeros((max_length, 21), dtype=float)
    encoded_sequence[np.arange(len(seq_ints)), seq_ints] = 1
    return encoded_sequence[:, 1:]


def integer_encoder(sequence: SeqRecord, max_length: int = 4034) -> np.ndarray:
    import re

    keys = dict(zip(" ACDEFGHIKLMNPQRSTVWYXBUJZO", range(26)))
    seq = re.sub(r'[^A-Z]', '', str(sequence.seq).upper()[:max_length])
    seq_ints = np.array([keys.get(aa.upper(), 0) for aa in seq])

    encoded_sequence = np.zeros((1, max_length), dtype=int)
    encoded_sequence[0, :len(seq)] = seq_ints
    return encoded_sequence


def prediction_mapper_one(
    sequences: Sequence[SeqRecord],
    model: Model
) -> np.ndarray:
    input_shape = model.layers[0].input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    max_length = input_shape[1]

    if "embedding" in model.layers[0].name:
        sequence_array = np.concatenate(
            [
                integer_encoder(seq, max_length)
                for seq
                in sequences
            ],
            axis=0
        )
    else:
        sequence_array = np.concatenate(
            [
                np.expand_dims(encode_one_hot(seq, max_length), 0)
                for seq
                in sequences
            ],
            axis=0
        )

    return (
        model
        .predict(sequence_array, batch_size=32, verbose=False)
        .reshape(-1)
    )


def prediction_mapper(
    sequences: Sequence[SeqRecord],
    models: Dict[str, Model]
) -> pd.DataFrame:

    preds = {}
    for k, v in models.items():
        preds_ = prediction_mapper_one(sequences, v)
        preds[k] = preds_

    seq_names = np.array([seq.id for seq in sequences])
    preds["name"] = seq_names
    return pd.DataFrame.from_dict(preds).set_index("name")


def predict_effector(
    sequences: Sequence[SeqRecord],
    taxon: Literal['bacteria', 'fungi', 'oomycete'],
) -> pd.DataFrame:
    from .data import load_model

    ensemble_method = get_ensemble_method(taxon=taxon)
    models = load_model(taxon=taxon)

    preds = prediction_mapper(sequences, models)
    preds = ensemble_method(preds)
    preds.name = "s_score"
    preds = preds.reset_index(drop=False)
    preds["prediction"] = "non-effector"
    preds.loc[preds["s_score"] >= 0.5, "prediction"] = "effector"
    return preds


def main():
    try:
        args = cli(prog=sys.argv[0], args=sys.argv[1:])
    except MyArgumentError as e:
        print(e.message, file=sys.stderr)
        sys.exit(e.errno)

    seqs = list(SeqIO.parse(args.infile, "fasta"))

    preds = predict_effector(seqs, args.taxon)
    preds.to_csv(args.outfile, sep="\t", index=False, na_rep=".")
    return


if __name__ == "__main__":
    main()
