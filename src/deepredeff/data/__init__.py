#!/usr/bin/env python3

from typing import (
    Literal,
    Dict,
)

from tensorflow.keras import Model


BACTERIA_WEIGHTS = {
    'gru_emb': 0.8947368421052632,
    'cnn_lstm': 0.8815789505055076,
    'lstm_emb': 0.934210529452876,
    'cnn_gru': 0.9605263157894736
}

BAC_TRAIN_FILES = [
    ('gru_emb', 'bacteria_gru_emb.csv'),
    ('bacteria_cnn_lstm', 'bacteria_cnn_lstm.csv'),
    ('lstm_emb', 'bacteria_lstm_emb.csv'),
    ('cnn_gru', 'bacteria_cnn_gru.csv')
]

MODELS = {
    "fungi": [
        "fungi_cnn_lstm.hdf5",
    ],
    "all": [
        "all_cnn_gru.hdf5",
        "all_cnn_lstm.hdf5",
        "all_gru_emb.hdf5",
        "all_lstm_emb.hdf5",
    ],
    "bacteria": [
        "bacteria_cnn_gru.hdf5",
        "bacteria_cnn_lstm.hdf5",
        "bacteria_gru_emb.hdf5",
        "bacteria_lstm_emb.hdf5",
    ],
    "oomycete": [
        "oomycete_cnn_lstm.hdf5",
    ],
}


def resource_filename(module, resource):
    """ Emulates the behaviour of the old setuptools resource_filename command.
    Basically it just gets rid of the context manager thing, because it's not
    needed. None of the files are zip files or create any temporary files
    that need to be cleaned up.
    This function would be unsafe to use with anything that will be extracted.
    """

    from importlib.resources import path
    with path(module, resource) as handler:
        filename = str(handler)

    return filename


def get_bacteria_weights(taxon="bacteria"):
    """
    This one is used to find BACTERIA_WEIGHTS
    """
    import pandas as pd

    files = {
        n: resource_filename(__name__, f)
        for n, f
        in BAC_TRAIN_FILES
    }

    return {
        n: pd.read_csv(f).tail(3)["acc"].max()
        for n, f
        in files
    }


def load_model(
    taxon: Literal['bacteria', 'fungi', 'oomycete'],
) -> Dict[str, Model]:
    """ Loads the model from hdf5 file. """
    import tensorflow as tf
    # Loads the keras models
    model_paths = MODELS[taxon]
    model_names = [m[(len(taxon) + 1):-len(".hdf5")] for m in model_paths]
    model_paths = [resource_filename(__name__, f) for f in model_paths]

    return {
        n: tf.keras.models.load_model(f)
        for n, f
        in zip(model_names, model_paths)
    }
