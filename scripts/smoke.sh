#!/bin/bash
# Smoke test of Table 5 (25 simultaneous edits, zsRE, BART-base); caps on the command line only:
# 2001 iterations = 200 optimizer steps (accumulate_bs=10), loss logged every 10 iterations,
# validation every 1000 iterations (500 val batches), checkpoint save on each new best,
# final evaluation on 200 dev batches after reloading the best checkpoint (debug=True save=True).
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
BART=${BART:-$HOME/mend/hf/bart-base}
python -m run +alg=mend +experiment=qa +model=bart-base data.zsre_nq=False \
  data.n_edits=25 batch_size=26 val_batch_size=26 \
  max_iters=2001 log_interval=10 val_interval=1000 debug=True save=True \
  model.name=$BART model.tokenizer_name=$BART "$@"
