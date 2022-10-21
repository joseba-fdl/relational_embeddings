#!/usr/bin/env bash

export MAX_LENGTH=
export TRAIN_BATCH_SIZE=
export LEARNING_RATE=
export SAVE_STEPS=
export SEED=
export EPOCHS=
export TRANSFORMER_BASE_DIR=bert-base-multilingual-cased
export CORPUS_DIR=
export RESULTS_DIR=
export REL_EMB_DIR= # DIRECTORY OF THE RELATIONAL EMBEDDINGS

for CORPUS_MOTA in 1

do
  mkdir $RESULTS_DIR$CORPUS_MOTA #mkdir $RESULTS_DIR-$i
  python run_classification_LM_ensemble.py --data_dir=$CORPUS_DIR  \
    --labels ./labels.txt \
    --model_type bert-base-multilingual-cased \
    --model_name_or_path $TRANSFORMER_BASE_DIR \
    --output_dir=$RESULTS_DIR$CORPUS_MOTA \
    --relational_embedding_dir=$REL_EMB_DIR \
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
