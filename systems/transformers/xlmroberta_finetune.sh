#!/usr/bin/env bash

export MAX_LENGTH=
export TRAIN_BATCH_SIZE=
export LEARNING_RATE=
export SAVE_STEPS=
export SEED=
export EPOCHS=
export TRANSFORMER_BASE_DIR=xlm-roberta-base 
export CORPUS_DIR=
export RESULTS_DIR=

for CORPUS_MOTA in 1

do
  mkdir $RESULTS_DIR #mkdir $RESULTS_DIR-$i
  python ./run_classification_LM.py --data_dir=$CORPUS_DIR  \
    --labels ./labels.txt \
    --model_type xlm-roberta \
    --model_name_or_path $TRANSFORMER_BASE_DIR \
    --output_dir=$RESULTS_DIR \
    --max_seq_length $MAX_LENGTH \
    --num_train_epochs=$EPOCHS \
    --per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --overwrite_cache \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --get_all_preds \

  #rm cache*
done
