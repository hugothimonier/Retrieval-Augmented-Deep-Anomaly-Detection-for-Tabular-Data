#!/bin/bash

conda activate ret_ad
mkdir -p ./logs/backdoor
mkdir -p ./results/backdoor
mkdir -p ./tblogs/backdoor

python -u run.py --exp_n_runs 1 \
    --exp_cat_as_num True \
    --data_path ./data/backdoor \
    --data_set backdoor \
    --res_dir ./results/ \
    --full_dataset_cuda False \
    --np_seed 10 \
    --torch_seed 10 \
    --exp_batch_size 512 \
    --exp_val_batchsize 64 \
    --exp_force_all_masked False \
    --exp_deterministic_masks_val True \
    --exp_retrieval_type attention_bsim \
    --exp_retrieval_knn_type brute \
    --exp_retrieval_metric euclidean \
    --exp_retrieval_location post-encoder \
    --exp_retrieval_agg_location post-encoder \
    --exp_retrieval_agg_lambda 0.5 \
    --exp_retrieval_num_helpers -1 \
    --exp_retrieval_num_candidate_helpers 512 \
    --exp_retrieval_normalization_sim False \
    --exp_num_masks_val 20 \
    --exp_n_hidden_features 0 \
    --exp_n_hidden_features 0 \
    --exp_n_hidden_features 0 \
    --exp_n_hidden_features 0 \
    --exp_n_hidden_features 0 \
    --exp_aggregation sum \
    --exp_normalize_ad_loss False \
    --exp_anomalies_in_inference False \
    --exp_patience 800 \
    --exp_cadence_type improvement \
    --exp_cache_cadence 5 \
    --exp_eval_every_n 601 \
    --exp_train_total_epochs 600 \
    --exp_optimizer lookahead_lamb \
    --exp_lookahead_update_cadence 6 \
    --exp_optimizer_warmup_proportion 0.7 \
    --exp_optimizer_warmup_fixed_n_steps -1 \
    --exp_lr 0.001 \
    --exp_scheduler flat_and_anneal \
    --exp_gradient_clipping 1.0 \
    --exp_weight_decay 1e-05 \
    --mp_distributed True \
    --mp_nodes 4 \
    --mp_gpus 16 \
    --mp_nr 0 \
    --model_dtype float32 \
    --data_loader_nprocs 0 \
    --load_from_checkpoint False \
    --model_amp True \
    --model_feature_type_embedding True \
    --model_feature_index_embedding True \
    --model_augmentation_bert_mask_prob '{'\''train'\'':0.2, '\''val'\'':0}' \
    --model_hidden_dropout_prob 0.1 \
    --model_dim_hidden 32 \
    --model_num_heads_e 4 \
    --model_num_layers_e 4 > ./logs/backdoor/backdoor_transformer_att_bs.log