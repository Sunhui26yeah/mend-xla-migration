#!/bin/bash
# Full run of Table 5 (25 simultaneous edits, zsRE, BART-base), repository defaults:
# seed 0, early stopping (patience 20000 on loss/total_edit_val, 500 val batches every 5000 iterations),
# max_iters 1e6, final evaluation on the full dev set after reloading the best checkpoint.
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
BART=${BART:-$HOME/mend/hf/bart-base}
python -m run +alg=mend +experiment=qa +model=bart-base data.zsre_nq=False \
  data.n_edits=25 batch_size=26 val_batch_size=26 \
  model.name=$BART model.tokenizer_name=$BART "$@"
