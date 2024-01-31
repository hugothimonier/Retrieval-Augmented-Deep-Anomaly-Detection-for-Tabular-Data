#!/bin/bash

conda activate ret_ad
mkdir -p ./logs/wine
mkdir -p ./results/wine
mkdir -p ./tblogs/wine

python -u run.py --exp_n_runs 20 \
    --data_path ./data/wine \
    --data_set wine \
    --res_dir ./results/ \
    --full_dataset_cuda True \
    --np_seed 1 \
    --torch_seed 1 \
    --exp_batch_size -1 \
    --exp_val_batchsize -1 \
    --exp_force_all_masked False \
    --exp_deterministic_masks_val True \
    --exp_retrieval_type v-attention \
    --exp_retrieval_knn_type brute \
    --exp_retrieval_metric euclidean \
    --exp_retrieval_location post-encoder \
    --exp_retrieval_agg_location post-encoder \
    --exp_retrieval_agg_lambda 0.5 \
    --exp_retrieval_num_helpers -1 \
    --exp_retrieval_num_candidate_helpers -1 \
    --exp_retrieval_normalization_sim False \
    --exp_num_masks_val 20 \
    --exp_n_hidden_features 2 \
    --exp_n_hidden_features 3 \
    --exp_n_hidden_features 0 \
    --exp_n_hidden_features 0 \
    --exp_n_hidden_features 0 \
    --exp_aggregation sum \
    --exp_normalize_ad_loss False \
    --exp_anomalies_in_inference False \
    --exp_patience 50 \
    --exp_cadence_type improvement \
    --exp_cache_cadence 5 \
    --exp_eval_every_n 500 \
    --exp_train_total_epochs 1000 \
    --exp_optimizer lookahead_lamb \
    --exp_lookahead_update_cadence 6 \
    --exp_optimizer_warmup_proportion 0.7 \
    --exp_optimizer_warmup_fixed_n_steps -1 \
    --exp_lr 0.001 \
    --exp_scheduler flat_and_anneal \
    --exp_gradient_clipping 10.0 \
    --exp_weight_decay 1e-05 \
    --mp_distributed False \
    --mp_nodes 1 \
    --mp_gpus 1 \
    --mp_nr 0 \
    --model_dtype float32 \
    --data_loader_nprocs 0 \
    --load_from_checkpoint False \
    --model_amp True \
    --model_mutual_update False \
    --model_feature_type_embedding True \
    --model_feature_index_embedding True \
    --model_augmentation_bert_mask_prob '{'\''train'\'':0.15, '\''val'\'':0}' \
    --model_hidden_dropout_prob 0.1 \
    --model_dim_hidden 8 \
    --model_num_heads_e 4 \
    --model_num_layers_e 4 > ./logs/wine/wine_transformer_v_att.log