export SQUAD_DIR=data/

CUDA_VISIBLE_DEVICES=1 python3 run_squad.py \
  --model_type bert \
  --model_name_or_path models/squad-v2 \
  --do_eval \
  --do_lower_case \
  --version_2_with_negative \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir models/squad-v2/ \
  --overwrite_output_dir \
  --save_steps 2000 \
  --eval_all_checkpoints \
  --ensemble 1000 \
  --ensemble_top_n 5 \
  --test_subset test_subset_3.pkl \
  --save_result Results_1000_set3.pkl \
#  --load_saved_results result_saving_test.pkl
#--do_train \
#--model_name_or_path bert-base-uncased \
