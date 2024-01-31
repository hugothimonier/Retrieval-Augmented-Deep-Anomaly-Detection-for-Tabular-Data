import os
from datetime import datetime


def make_job_name(kwargs, seed=None, iteration=None):
    
    now = datetime.now()
    date = now.strftime("%m_%d__%H_%M_%S")
    
    str_train = str(kwargs.model_augmentation_bert_mask_prob['train'])
    retrieval = str((not kwargs.exp_retrieval_num_candidate_helpers == 0))[0]
    str_type_embed = str(kwargs.model_feature_type_embedding)[0]
    str_index_embed = str(kwargs.model_feature_index_embedding)[0]
    
    if seed is not None:
        current_seed = seed + iteration
        job_name = ('{date}_{dataset}__seed_{seed}__nruns_{n_runs}__bs_{batchsize}__lr_{lr}'
                '__nepochs_{nsteps}__hdim_{hdim}__'
                'maskprob_{trainmaskprob}__'
                'ftypeembd_{type_embed}__'
                'indexembd_{idx_embed}__'
                'n_hds_e_{num_heads_e}__'
                'n_lyrs_e_{num_lyrs_e}__'
                'ret_{retrieval}')
        job_name= job_name.format(
                date=date,
                dataset=kwargs.data_set.upper(),
                seed=current_seed,
                n_runs=kwargs.exp_n_runs,
                batchsize=str(kwargs.exp_batch_size),
                lr=kwargs.exp_lr,
                nsteps=int(kwargs.exp_train_total_epochs),
                hdim=kwargs.model_dim_hidden,
                trainmaskprob=str_train,
                type_embed=str_type_embed,
                idx_embed=str_index_embed,
                num_heads_e=kwargs.model_num_heads_e,
                num_lyrs_e=kwargs.model_num_layers_e,
                retrieval=retrieval
                )
    else:
        job_name = ('{dataset}__nruns_{n_runs}__ret_{retrieval}'
                    '__bs_{batchsize}__lr_{lr}'
                    '__nepochs_{nsteps}__hdim_{hdim}__'
                    'maskprob_{trainmaskprob}__'
                    'ftypeembd_{type_embed}__'
                    'indexembd_{idx_embed}__'
                    'n_hds_e_{num_heads_e}__'
                    'n_lyrs_e_{num_lyrs_e}__'
                    )
        job_name= job_name.format(
                dataset=kwargs.data_set.upper(),
                n_runs=kwargs.exp_n_runs,
                retrieval=retrieval,
                batchsize=str(kwargs.exp_batch_size),
                lr=kwargs.exp_lr,
                nsteps=int(kwargs.exp_train_total_epochs),
                hdim=kwargs.model_dim_hidden,
                trainmaskprob=str_train,
                type_embed=str_type_embed,
                idx_embed=str_index_embed,
                num_heads_e=kwargs.model_num_heads_e,
                num_lyrs_e=kwargs.model_num_layers_e,
                )
        
    

    if kwargs.exp_retrieval_num_candidate_helpers != 0:
        job_name_addition = ('__ret_type_{retrieval_type}'
                            '__ret_loc_{retrieval_location}'
                            '_agg_loc_{retrieval_agg_location}'
                            '__n_help_{retrieval_num_helpers}'
                            '__ret_n_cand_helpers_{num_train_inference}')
        if kwargs.exp_retrieval_type == 'attention_bsim_bval':
            log_name_ret_type = 'att_bs_bv'
        elif kwargs.exp_retrieval_type == 'attention_bsim':
            log_name_ret_type = "att_bs"
        elif kwargs.exp_retrieval_type == "v-attention":
            log_name_ret_type = "v_att"
        else:
            log_name_ret_type = kwargs.exp_retrieval_type
        job_name += job_name_addition.format(
            retrieval_type=log_name_ret_type,
            retrieval_location = kwargs.exp_retrieval_location[:7],
            retrieval_agg_location = kwargs.exp_retrieval_agg_location[:7],
            retrieval_num_helpers = kwargs.exp_retrieval_num_helpers,
            num_train_inference = kwargs.exp_retrieval_num_candidate_helpers
        )

    return job_name

def print_args(args):
    print(f'Running {args.exp_n_runs} experiments.')
    if args.exp_retrieval_type=='None':
        to_print = 'No retrieval'
    else:
        to_print = (f'Retrieval type: {args.exp_retrieval_type}\n'
                    f'Retrieval location: {args.exp_retrieval_location}\n'
                    f'Retrieval aggregation location: {args.exp_retrieval_agg_location}\n'
                    f'Retrieval lambda aggregation: {args.exp_retrieval_agg_lambda}\n'
                    f'Number of candidate helpers: {args.exp_retrieval_num_candidate_helpers}\n'
                    f'Number of chosen helpers (k): {args.exp_retrieval_num_helpers}\n'
                   )
    print(to_print)
    str_train = str(args.model_augmentation_bert_mask_prob['train'])
    str_type_embed = str(args.model_feature_type_embedding)
    str_index_embed = str(args.model_feature_index_embedding)
    to_print = (f'Batch size: {args.exp_batch_size}\n'
                f'Number of epochs: {args.exp_train_total_epochs}\n'
                f'Learning rate: {args.exp_lr}\n'
                f'Hidden dim: {args.model_dim_hidden}\n'
                f'Masking prob: {str_train}\n'
                f'Number of attention heads: {args.model_num_heads_e}\n'
                f'Number of layers: {args.model_num_layers_e}\n'
                f'Feature type embedding: {str_type_embed}\n'
                f'Index embedding: {str_index_embed}\n')
    print(to_print)
        