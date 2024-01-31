"""Load model, data and corresponding configs. Trigger training."""
import os, ast, json
import random
from datetime import datetime

import numpy as np
import torch
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformer import Model
from retrieval import Retrieval
from loss import Loss
from torch_dataset import TorchDataset
from train import Trainer

from configs import build_parser
from utils.encode_utils import get_torch_dtype
from utils.train_utils import init_optimizer, count_parameters
from utils.log_utils import print_args

from datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP

# Environment variables set by torch.distributed.launch
if torch.cuda.is_available():
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])

def main(args):

    if args.mp_distributed:
        dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=WORLD_SIZE,
                rank=WORLD_RANK)
    
    kwargs = vars(args)
    
    dataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](**kwargs)
    args.is_batchlearning = (args.exp_batch_size != -1)
    args.iteration = 0
    if ((not args.mp_distributed) 
        or (args.mp_distributed and LOCAL_RANK ==0)):
        print_args(args)
    
    metrics = []
    val_duration = []
    
    for iteration in range(args.exp_n_runs):
        args.iter_seed = args.np_seed + iteration
        
        #seed setting
        torch.manual_seed(args.torch_seed + iteration)
        np.random.seed(args.iter_seed)
        random.seed(args.iter_seed)

        if args.mp_distributed:
            torch.cuda.set_device(LOCAL_RANK)
            dist_args = {'world_size': WORLD_SIZE,
                        'rank': WORLD_RANK,
                        'gpu': LOCAL_RANK}
            
            args.iteration = iteration 
            dataset.load()
            dataset.split_train_val(seed=args.np_seed, 
                                    iteration=iteration,
                                    contamination_share=args.exp_contamination)
            
            torchdataset = TorchDataset(dataset=dataset,
                                        mode='train',
                                        kwargs=args,
                                        device=LOCAL_RANK)
            
            if args.exp_retrieval_type != 'None':
                if args.exp_retrieval_type.lower() == 'knn':
                    retrieval_dict = {
                        'knn_type':args.exp_retrieval_knn_type,
                        'k':args.exp_retrieval_num_helpers, 
                        'metric':args.exp_retrieval_metric}
                else:
                    err_msg = ('Retrieval location cannot be before embedding'
                              +' if the retrieval module is not knn.')
                    assert args.exp_retrieval_agg_location!='pre-embedding', err_msg

                    in_dim_ret = args.model_dim_hidden * dataset.D
                    out_dim_ret = args.model_dim_hidden * dataset.D

                    retrieval_dict = {
                    'retrieval_type':args.exp_retrieval_type,
                    'in_dim':in_dim_ret,
                    'out_dim':out_dim_ret,
                    'k': args.exp_retrieval_num_helpers,
                    'normalization_sim':args.exp_retrieval_normalization_sim,
                    'n_features':dataset.D}

                retrieval_module = Retrieval(
                    type_retrieval=args.exp_retrieval_type,
                    num_helpers=args.exp_retrieval_num_helpers,
                    retrieval_kwargs=retrieval_dict,
                    deterministic_selection=[],
                    device=LOCAL_RANK
                )
            else:
                retrieval_module = None
            
            model = Model(
                idx_num_features=dataset.num_features,
                cardinalities=dataset.cardinalities,
                hidden_dim=args.model_dim_hidden,
                num_layers_e=args.model_num_layers_e,
                num_layers_d=args.model_num_layers_d,
                num_heads_e=args.model_num_heads_e,
                num_heads_d=args.model_num_heads_d,
                layer_norm_eps=1e-5,
                p_dropout=args.model_hidden_dropout_prob,
                gradient_clipping=args.exp_gradient_clipping,
                feature_type_embedding=args.model_feature_type_embedding,
                feature_index_embedding=args.model_feature_index_embedding,
                retrieval=retrieval_module,
                device=LOCAL_RANK,
                args=args,
                          )
                    
            
            torch_dtype = get_torch_dtype(args.model_dtype)
            model = model.to(LOCAL_RANK).type(torch_dtype)
            print(f'Model has {count_parameters(model)} parameters,'
                  f'batch size {args.exp_batch_size}.')
            
            if args.exp_retrieval_type.lower() in ['none', 'knn']: 
                optimizer = init_optimizer(args=args,
                                          model_parameters=model.parameters())
            else:
                # explicitly tell the optimizer to include the retrieval_module params
                optimizer = init_optimizer(args=args,
                                          model_parameters=list(model.parameters())+
                                          list(model.retrieval_module.retrieval_module.parameters())
                                          )
            print(f'Initialized "{args.exp_optimizer}" optimizer.')
            scaler = GradScaler(enabled=args.model_amp)
            if args.model_amp:
                print(f'Initialized gradient scaler for Automatic Mixed Precision.')

            # Wrap model
            model = DDP(model, 
                        device_ids=[LOCAL_RANK],
                        find_unused_parameters=True)
            dist.barrier()

            loss = Loss(args, 
                        is_batchlearning=args.is_batchlearning,
                        device=LOCAL_RANK)

            trainer = Trainer(model=model,
                              optimizer=optimizer,
                              loss=loss,
                              scaler=scaler,
                              kwargs=args,
                              torch_dataset=torchdataset,
                              distributed_args=dist_args,
                              device=LOCAL_RANK)
            
            trainer.train()
            val_start_time = datetime.now()
            trainer.val()
            val_end_time = datetime.now()
            val_duration_sec = (val_end_time - val_start_time).total_seconds()
            val_duration.append(val_duration_sec)
            ratio_metrics = trainer.compute_metrics()
            metrics.append(ratio_metrics)
            if LOCAL_RANK == 0:
                print(ratio_metrics)
                print('Val duration:', val_duration_sec)

        else:
            if not torch.cuda.is_available():
                args.device = 'cpu'
            else:
                args.device = 'cpu' if args.mp_gpus==0 else 'cuda:0'
            if args.mp_gpus==0 or args.device == 'cpu':
                args.model_amp = False
                args.full_dataset_cuda = False
            
            if args.device != 'cpu':
                torch.cuda.set_device(args.device)
            
            args.iteration = iteration 
            dataset.load()
            dataset.split_train_val(seed=args.np_seed, 
                                    iteration=iteration,
                                    contamination_share=args.exp_contamination)
            torchdataset = TorchDataset(dataset=dataset,
                                        mode='train',
                                        kwargs=args,
                                        device=args.device)
            
            if args.exp_retrieval_type != 'None':
                if args.exp_retrieval_type.lower() == 'knn':
                    retrieval_dict = {
                        'knn_type':args.exp_retrieval_knn_type,
                        'k':args.exp_retrieval_num_helpers, 
                        'metric':args.exp_retrieval_metric}
                else:
                    err_msg = ('Retrieval location cannot be before embedding'
                              +' if the retrieval module is not knn.')
                    assert args.exp_retrieval_agg_location!='pre-embedding', err_msg

                    in_dim_ret = args.model_dim_hidden * dataset.D
                    out_dim_ret = args.model_dim_hidden * dataset.D
                        
                    retrieval_dict = {
                    'retrieval_type':args.exp_retrieval_type,
                    'in_dim':in_dim_ret,
                    'out_dim':out_dim_ret,
                    'k': args.exp_retrieval_num_helpers,
                    'normalization_sim':args.exp_retrieval_normalization_sim,
                    'n_features':dataset.D}

                retrieval_module = Retrieval(
                    type_retrieval=args.exp_retrieval_type,
                    num_helpers=args.exp_retrieval_num_helpers,
                    retrieval_kwargs=retrieval_dict,
                    deterministic_selection=[],
                    device=args.device)
            else:
                retrieval_module = None
            
            model = Model(
                idx_num_features=dataset.num_features,
                cardinalities=dataset.cardinalities,
                hidden_dim=args.model_dim_hidden,
                num_layers_e=args.model_num_layers_e,
                num_heads_e=args.model_num_heads_e,
                layer_norm_eps=1e-5,
                p_dropout=args.model_hidden_dropout_prob,
                gradient_clipping=args.exp_gradient_clipping,
                feature_type_embedding=args.model_feature_type_embedding,
                feature_index_embedding=args.model_feature_index_embedding,
                retrieval=retrieval_module,
                device=args.device,
                args=args,)
                    
            
            torch_dtype = get_torch_dtype(args.model_dtype)
            model = model.to(args.device).type(torch_dtype)
            print(f'Model has {count_parameters(model)} parameters,'
                  f'batch size {args.exp_batch_size}.')
            if args.exp_retrieval_type.lower() in ['none', 'knn']: 
                optimizer = init_optimizer(args=args,
                                          model_parameters=model.parameters())
            else:
                # explicitly tell the optimizer to include the retrieval_module params
                print('Optimizer look at both module params')
                optimizer = init_optimizer(args=args,
                                          model_parameters=list(model.parameters())+
                                          list(model.retrieval_module.retrieval_module.parameters())
                                          )
            print(f'Initialized "{args.exp_optimizer}" optimizer.')
            scaler = GradScaler(enabled=args.model_amp)
            if args.model_amp:
                print(f'Initialized gradient scaler for Automatic Mixed Precision.')

            loss = Loss(args, 
                        is_batchlearning=args.is_batchlearning,
                        device=args.device)

            trainer = Trainer(model=model,
                              optimizer=optimizer,
                              loss=loss,
                              scaler=scaler,
                              kwargs=args,
                              torch_dataset=torchdataset,
                              distributed_args=None,
                              device=args.device)
            
            trainer.train()
            print('\n\nTraining is over, final validation step.\n\n')
            val_start_time = datetime.now()
            trainer.val()
            val_end_time = datetime.now()
            val_duration_sec = (val_end_time - val_start_time).total_seconds()
            val_duration.append(val_duration_sec)
            ratio_metrics = trainer.compute_metrics()
            metrics.append(ratio_metrics)
            print(ratio_metrics)
            print('Val duration', val_duration_sec)
            
    if ((not args.mp_distributed) or (LOCAL_RANK == 0)): 
        if len(metrics)>1:
            f1 = []
            auc = []
            ap = []
            train_time = []
            for ele in metrics:
                f1.append(ele['F1'])
                auc.append(ele['auc'])
                ap.append(ele['ap'])
                train_time.append(ele['training_time'])
            to_print = (f'Mean F1 score and std over {args.exp_n_runs} runs: {np.mean(f1)},{np.std(f1)}\n'
                        f'Mean AUC score and std over {args.exp_n_runs} runs: {np.mean(auc)},{np.std(auc)}\n'
                        f'Mean AP score and std over {args.exp_n_runs} runs: {np.mean(ap)},{np.std(ap)}\n'
                        f'Average training time over {args.exp_n_runs} runs: {np.mean(train_time)},{np.std(train_time)}\n'
                        f'Average inference time over {args.exp_n_runs} runs: {np.mean(val_duration)},{np.std(val_duration)}\n'
                       )
            print(to_print)
            out_score = dict()
            out_score['F1'] = (np.mean(f1),np.std(f1))
            out_score['AUC'] = (np.mean(auc),np.std(auc))
            out_score['AP'] = (np.mean(ap),np.std(ap))
            out_score['train_time'] = (np.mean(train_time),np.std(train_time))
            out_score['val_time'] = (np.mean(val_duration),np.std(val_duration))
            out_filename = ('mean_results.json')
            out_path = os.path.join(trainer.res_dir, out_filename)

            with open(out_path, 'w') as fp:
                json.dump(out_score, fp)

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args.model_augmentation_bert_mask_prob =  ast.literal_eval(
        args.model_augmentation_bert_mask_prob)

    main(args)
