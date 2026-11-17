#!/bin/bash
# Installs the packages that the scripts and the notebooks use. TransformerLens is pinned to a version that provides
# HookedTransformer, from which each model takes its initial weights.
pip install torch numpy matplotlib seaborn jupyter "transformer_lens==2.18.0" "transformers==4.57.6"
