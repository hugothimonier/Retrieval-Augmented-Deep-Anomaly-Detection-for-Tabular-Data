"""Define argument parser."""
import argparse

DEFAULT_AUGMENTATION_BERT_MASK_PROB = {
    'train': 0.15,
    'val': 0.15,
}

def str2bool(v):
    """https://stackoverflow.com/questions/15008758/
    parsing-boolean-values-with-argparse/36031646"""
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")

def build_parser():
    '''Build parser'''

    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    ###########################################################################
    # #### Data Config ########################################################
    ###########################################################################
    parser.add_argument(
        '--res_dir', type=str, default='./',
        help='Revelevant for distributed training. Name of directory where '
             ' the dictionnaries containing the anomaly scores.')
    parser.add_argument(
        '--data_path', type=str, default='data',
        help='Path of data')
    parser.add_argument(
        '--data_set', type=str, default='abalone',
        help='accepted values are currently: '
            'abalone, forestcoverad, mammography, satellite, vertebral, '
            'annthyroid, glass, mnistad, satimage, vowels, '
            'arrhythmia, ionosphere, mulcross, seismic, wbc, '
            'kdd, musk, separable, wine, '
            'breastw, kddrev, optdigits, shuttle, '
            'cardio, letter, pendigits, speech, '
            'ecoli, lympho, pima, thyroid.')
    parser.add_argument(
        '--full_dataset_cuda', type='bool', default=False,
        help='Put the whole dataset on cuda to accelerate training.'
        'Induces a higher memory cost, only to be used for small'
        ' medium sized datasets.')
    parser.add_argument(
         '--exp_cat_as_num', type='bool', default=False,
    )
    
    ###########################################################################
    # #### Experiment Config ##################################################
    ###########################################################################
    parser.add_argument(
        '--exp_n_runs', type=int, default=1,)
    parser.add_argument(
        '--exp_device', default=None, type=str,
        help='If provided, use this (CUDA) device for the run.')
    parser.add_argument(
        '--np_seed', type=int, default=42,
        help='Random seed for numpy. Set to -1 to choose at random.')
    parser.add_argument(
        '--torch_seed', type=int, default=42,
        help='Random seed for torch. Set to -1 to choose at random.')
    parser.add_argument(
        '--exp_batch_size', type=int, default=-1,
        help='Number of instances (rows) in each batch '
             'taken as input by the model. -1 corresponds to no '
             'minibatching.')
    parser.add_argument(
        '--exp_force_all_masked', type='bool', default=False,
        help='Force all train samples to be masked in training.'
             'For inference, we believe the model shoudl also learn'
             'To see unmasked samples.')
    
    ###########################################################################
    # #### Retrieval Config ###################################################
    ###########################################################################
    parser.add_argument(
        '--exp_val_batchsize', type=int, default=1,
        help='Number of validation samples to feed for the reconstruction '
             'in the matrix for AD.')
    parser.add_argument(
        '--exp_retrieval_type', type=str, default='None',
        choices=['None','knn','v-attention', "attention_bsim_nval",
                 'attention_bsim',"attention_bsim_bval"],
        help='Retrieval module type.')
    parser.add_argument(
        '--exp_retrieval_knn_type', type=str, default='brute',
        choices=["auto", "ball_tree", "kd_tree", "brute"],
        help='Retrieval module knn type. Relevant if knn.')
    parser.add_argument(
        '--exp_retrieval_metric', type=str, default='euclidean',
        help='Retrieval module metric. Relevant if knn.')
    parser.add_argument(
        '--exp_retrieval_location', type=str, 
        choices=['pre-embedding', 'post-embedding', 'post-encoder'],
        help='Where to locate the retrieval module (per the figures).')
    parser.add_argument(
        '--exp_retrieval_agg_location', type=str, 
        choices=['pre-embedding', 'post-embedding', 'post-encoder'],
        help='Where to aggregate representation of the target sample,'
        'and the sample helping.')
    parser.add_argument(
        '--exp_retrieval_agg_lambda', type=float,
        help='How much to value the help of the helping samples. Currently'
             'Value is aggregated as such (1-\lambda) x + \lambda. help_samples.'
             'The higher the lambda, the more weight on the helping samples.')
    parser.add_argument(
        '--exp_retrieval_num_helpers', type=int, default=5,
        help='Number of sample to be chosen amongst the candidate helpers.')
    parser.add_argument(
        '--exp_retrieval_num_candidate_helpers', type=int, default=-1,
        help='Number of sample among which to choose helpers.')
    parser.add_argument(
        '--exp_retrieval_normalization_sim',type='bool',
        default=False,
        help='Whether to normalize the similarity module by the dimension.'
             'This is usually done in self-attention mechanisms.')
    parser.add_argument(
        '--exp_deterministic_masks_val', type='bool', 
        default=False,
        help='Whether to use ret_ad-like masking in inference.')
    parser.add_argument(
        '--exp_num_masks_val', type=int, 
        default=0,
        help='Relevant only if exp_deterministic_masks_val is False.'
             ' Number of random masks to compute in inference.')
    parser.add_argument(
        '--exp_n_hidden_features', action='append', type=int,
        default=[1],
        help='Number of features to mask at a time.'
             'Determines the number of reconstruction.')
    parser.add_argument(
        '--exp_aggregation', type=str, default='sum', choices={'max', 'sum'},
        help='Type of reconstruction aggregation. Must be none or sum.'
             ' Valid only for AD.')
    parser.add_argument(
        '--exp_normalize_ad_loss', type='bool', default=False,
        help='Whether to normalize each reconstruction loss in the process'
        'of computing the anomaly score.')
    parser.add_argument(
        '--exp_contamination', type=float, default=0.,
        help='Share of anomalies in training set. Max 10%=0.1')
    parser.add_argument(
        '--exp_anomalies_in_inference', type='bool', default=0.,
        help='Share of anomalies in training set. Max 10%=0.1')
    
    ###########################################################################
    ######## Caching Config ###################################################
    ###########################################################################
    parser.add_argument(
        '--exp_patience', type=int, default=-1,
        help='Early stopping -- number of epochs that '
             'training may not improve before training stops. '
             'Turned off by default.')
    parser.add_argument(
        '--exp_cadence_type', type=str, default='recurrent',
        choices=['improvement','recurrent','None'],
        help='What type of caching to consider. If \'improvement\''
        ',caching every exp_cache_cadence times that train loss'
        'improved. If \'recurrent\' caching every exp_cache_cadence'
        'epochs.')
    parser.add_argument(
        '--exp_cache_cadence', type=int, default=1,
        help='Checkpointing -- we cache the model every `exp_cache_cadence` '
             'Set this value to -1 to disable caching.')
    parser.add_argument(
        '--exp_eval_every_n', type=int, default=5,
        help='Evaluate the model every n steps/epochs. (See below).')
    

    # Optimization
    # -------------
    parser.add_argument(
        '--exp_train_total_epochs', type=int, default=100,
        help='Number of epochs.')
    parser.add_argument(
        '--exp_optimizer', type=str, default='lookahead_lamb',
        help='Model optimizer: see npt/optim.py for options.')
    parser.add_argument(
        '--exp_lookahead_update_cadence', type=int, default=6,
        help='The number of steps after which Lookahead will update its '
             'slow moving weights with a linear interpolation between the '
             'slow and fast moving weights.')
    parser.add_argument(
        '--exp_optimizer_warmup_proportion', type=float, default=0.7,
        help='The proportion of total steps over which we warmup.'
             'If this value is set to -1, we warmup for a fixed number of '
             'steps. Literature such as Evolved Transformer (So et al. 2019) '
             'warms up for 10K fixed steps, and decays for the rest. Can '
             'also be used in certain situations to determine tradeoff '
             'annealing, see exp_tradeoff_annealing_proportion below.')
    parser.add_argument(
        '--exp_optimizer_warmup_fixed_n_steps', type=int, default=10000,
        help='The number of steps over which we warm up. This is only used '
             'when exp_optimizer_warmup_proportion is set to -1. See above '
             'description.')
    parser.add_argument(
        '--exp_lr', type=float, default=1e-3,
        help='Learning rate')
    parser.add_argument(
        '--exp_scheduler', type=str, default='flat_and_anneal',
        help='Learning rate scheduler: see npt/optim.py for options.')
    parser.add_argument(
        '--exp_gradient_clipping', type=float, default=1.,
        help='If > 0, clip gradients.')
    parser.add_argument(
        '--exp_weight_decay', type=float, default=0,
        help='Weight decay / L2 regularization penalty. Included in this '
             'section because it is set in the optimizer. '
             'HuggingFace default: 1e-5')

    ###########################################################################
    # #### Multiprocess Config ################################################
    ###########################################################################

    parser.add_argument(
        '--mp_distributed', dest='mp_distributed', default=False, type='bool',
        help='If True, run data-parallel distributed training with Torch DDP.')
    parser.add_argument(
        '--mp_nodes', dest='mp_nodes', default=1, type=int,
        help='number of data loading workers')
    parser.add_argument(
        '--mp_gpus', dest='mp_gpus', default=1, type=int,
        help='number of gpus per node')
    parser.add_argument(
        '--mp_nr', dest='mp_nr', default=0, type=int,
        help='ranking within the nodes')
    
    ###########################################################################
    # #### General Model Config ###############################################
    ###########################################################################

    parser.add_argument(
        '--model_dtype',
        default='float32',
        type=str, help='Data type (supported for float32, float64) '
                       'used for model.')
    parser.add_argument(
        '--data_loader_nprocs', type=int, default=0,
        help='Number of processes to use in data loading. Specify -1 to use '
             'all CPUs for data loading. 0 (default) means only the main  '
             'process is used in data loading. Applies for serial and '
             'distributed training.')
    parser.add_argument(
        '--load_from_checkpoint', type='bool', default=False,
        help='Whether to load from last saved checkpoints.')
    parser.add_argument(
        '--model_amp',
        default=False,
        type='bool', 
        help='If True, use automatic mixed precision (AMP), '
            'which can provide significant speedups on V100/'
            'A100 GPUs.')
    parser.add_argument(
        '--model_feature_type_embedding', type='bool', default=True,
        help='When True, learn an embedding on whether each feature is '
             'numerical or categorical. Similar to the "token type" '
             'embeddings canonical in NLP. See https://github.com/huggingface/'
             'transformers/blob/master/src/transformers/models/bert/'
             'modeling_bert.py')
    parser.add_argument(
        '--model_feature_index_embedding', type='bool', default=True,
        help='When True, learn an embedding on the index of each feature '
             '(column). Similar to positional embeddings.')
    parser.add_argument(
        '--model_augmentation_bert_mask_prob',
        type=str, default=DEFAULT_AUGMENTATION_BERT_MASK_PROB,
        help='Use bert style augmentation with the specified mask probs'
             'in training/validation/testing.')
    parser.add_argument(
        '--model_hidden_dropout_prob', type=float, default=0.1,
        help='The dropout probability for all fully connected layers in the '
             '(in, but not out) embeddings, attention blocks.')
    parser.add_argument(
        '--model_act_func', type=str, default='relu',
        help='Activation functions in model.', 
        choices=['relu','gelu'])    
    parser.add_argument(
        '--model_dim_hidden', type=int, default=64,
        help='Intermediate feature dimension.')
    parser.add_argument(
        '--model_num_heads_e', type=int, default=8,
        help='Number of attention heads. Must evenly divide model_dim_hidden.')
    parser.add_argument(
        '--model_num_layers_e', type=int, default=4,
        help='')
    parser.add_argument(
        '--model_mutual_update', type='bool', default=True,
        help='Whether to update the encoder using the gradient of both'
             'the sample of interest and the helpers.')
    
    return parser