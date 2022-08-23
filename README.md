# deepredeff-py

A python port of the [deepredeff R-package](https://github.com/ruthkr/deepredeff).

We were finding it difficult to get deepredeff to work in complex environments, and because dependencies were hard-coded it is increasingly difficult to update other software.
All of these issues stem from the fact that Deepredeff internally creates abd yses a conda-environment to run a keras model in python.
This is fine if you're using the package interactively, but it makes it a pain if you want to run in an existing conda environment. 
Soooo... why not just run it in python?

That's all we're doing.


**THIS IS NOT OUR WORK!!!**
If you use this code, please cite the original [deepredeff](https://github.com/ruthkr/deepredeff) paper (https://doi.org/10.1186/s12859-021-04293-3).


## Quick usage

This is primarily intended to be used as a command line tool.

```
deepredeff --outfile predictions.tsv --taxon fungi in.fasta
```

The default taxon is fungi. If `--outfile` is not provided we'll write the table to stdout.
If you want to pipe the fastas to the input, use `-` for the positional argument to indicate stdin.
e.g. `cat in.fasta | deepredeff --taxon bacteria - > out.tsv`


## Install

You can install this from the predector conda channel.

```
conda install -c predector deepredeff-py
```

If you'd like to use pip, you can run:

```
# use of Virtual environments (e.g. venv, conda) is strongly recommended.
# Never install python packages with pip as root user.
pip install git+https://github.com/darcyabjones/deepredeff-py.git
```
