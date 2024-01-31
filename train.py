import os, glob, re, pickle, sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader

from optim import LRScheduler
from torch_dataset import TorchDataset
from utils.log_utils import make_job_name
from utils.checkpoint_utils import EarlyStopCounter, EarlyStopSignal
from loss import Loss
from mask import apply_mask, generate_mask_val
from sklearn.metrics import precision_recall_fscore_support as prf
from torchmetrics import (AveragePrecision, AUROC)

class Trainer():

    def __init__(
            self,
            model:nn.Module,
            optimizer:optim.Optimizer,
            loss:Loss,
            scaler,
            kwargs,
            torch_dataset:TorchDataset,
            distributed_args:dict,
            device:torch.device,
    ):
        self.is_distributed = False
        self.kwargs = kwargs
        self.seed = self.kwargs.np_seed
        self.iteration = self.kwargs.iteration
        self.device = device
        self.n_epochs = kwargs.exp_train_total_epochs
        self.p_mask_train = kwargs.model_augmentation_bert_mask_prob['train']
        self.p_mask_val = kwargs.model_augmentation_bert_mask_prob['val']
        self.total_train_time = 0
        self.total_val_time = 0
        self.data_loader_nprocs = self.kwargs.data_loader_nprocs
        self.iter_seed = kwargs.iter_seed

        if distributed_args is not None:
            self.is_distributed = True
            self.world_size = distributed_args['world_size']
            self.rank = distributed_args['rank']
            self.gpu = distributed_args['gpu']

        self.model = model
        if self.is_distributed:
            self.retrieval = (self.model.module.retrieval_module is not None)
        else:
            self.retrieval = (self.model.retrieval_module is not None)
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = LRScheduler(c=self.kwargs, 
                                     name=kwargs.exp_scheduler,
                                     optimizer=optimizer)
        
        # at this point, dataset should have been loaded (dataset.load())
        # and split between train and val should have been done as well
        # .split_train_val(seed=, iteration=)
        self.dataset = torch_dataset
        self.epoch = 0
        self.loss = loss

        self.is_batch_learning = (self.kwargs.exp_batch_size != -1)
        self.is_batch_eval = (self.kwargs.exp_val_batchsize != -1)
        self.num_epoch = self.kwargs.exp_train_total_epochs
        
        if not self.is_batch_learning:
            self.batch_size = len(self.dataset.train[0])
        else:
            self.batch_size = self.kwargs.exp_batch_size
        
        if not self.is_batch_eval:
            self.batch_size_val = len(self.dataset.val[0])
        else:
            self.batch_size_val = self.kwargs.exp_val_batchsize
        
        self.full_val_inference = not ((self.is_batch_learning) or 
                                        (self.kwargs.exp_val_batchsize != -1))
        
        self.val_masks = generate_mask_val((1,self.dataset.n_features),
                                    self.kwargs.exp_n_hidden_features,
                                    self.batch_size_val,
                                    self.kwargs.exp_deterministic_masks_val,
                                    self.kwargs.exp_num_masks_val,
                                    self.p_mask_val)
        
        self.val_ad_score = []

        self.is_main_process = ((self.is_distributed and self.rank==0) or 
                                (not self.is_distributed))
        
        self.job_name = make_job_name(self.kwargs, 
                                      self.seed,
                                      self.iteration)
        self.job_name_noseed = make_job_name(self.kwargs, 
                                      )
        
        self.early_stop_counter = EarlyStopCounter(self.kwargs,
                                        self.job_name,
                                        self.dataset.dataset_name,
                                        device=self.device,
                                        is_distributed=self.is_distributed)
        
        self.training_is_over = False
        
        if self.is_main_process:

            self.res_dir = os.path.join('./results', 
                    self.dataset.dataset_name, 
                    self.job_name_noseed)
            
            if not os.path.isdir(self.res_dir):
                os.makedirs(self.res_dir)
                
            self.avg_time = []

        self.checkpoint_dir = os.path.join('./checkpoints', 
                                    self.dataset.dataset_name, 
                                    self.job_name)
    
        if not os.path.isdir(self.checkpoint_dir):
            if self.is_main_process:
                success = False
                while not success:
                    try:
                        os.makedirs(self.checkpoint_dir)
                        success = True
                    except:
                        self.job_name = make_job_name(self.kwargs, 
                                      self.seed,
                                      self.iteration)
                        self.checkpoint_dir = os.path.join('./checkpoints', 
                                    self.dataset.dataset_name, 
                                    self.job_name)
                        pass
                    
        else:
            if self.kwargs.load_from_checkpoint:
                try:
                    name_list = glob.glob(os.path.join(self.checkpoint_dir, 
                                                      f"seed_{self.iter_seed}_model_epoch_*.pth"))
                    epoch_st = 0
                    if len(name_list) > 0:
                        epoch_list = []
                        for name in name_list:
                            s = re.findall(r'\d+', os.path.basename(name))[0]
                            epoch_list.append(int(s))

                        epoch_list.sort()
                        epoch_st = epoch_list[-1]
                    if epoch_st > 0:
                        print('==========================================================')
                        print('===> Resuming model from epoch %d' %epoch_st)
                        print('==========================================================')
                        (self.model, 
                        self.optimizer,
                        self.scaler,
                        self.scheduler) = self.early_stop_counter.load_model(epoch_st,
                                                                      self.model,
                                                                      self.optimizer,
                                                                      self.scaler,
                                                                      self.scheduler
                                                                      )
                        self.epoch = epoch_st
                except Exception as e:
                    print('No model to load, continuing as such.')

    def train(self,):
        
        self.dataset.set_mode('train')
        if not self.is_distributed:
            dataloader = DataLoader(self.dataset,
                                        batch_size=self.batch_size,
                                        num_workers=0,
                                        shuffle=True,
                                        drop_last=True)
        else:
            dataloader = self.get_distributed_dataloader(self.batch_size,
                                                             self.dataset)
            
        self.model.to(self.device)
        self.loss.set_mode('train')

        while self.epoch < self.num_epoch:
            start_time = datetime.now()
            self.epoch += 1
                
            to_print= f'Training epoch: {self.epoch}/{self.num_epoch}'
            if self.is_main_process:
                print(f'{to_print:#^80}')
            if self.is_distributed:
                dataloader.sampler.set_epoch(epoch=self.epoch)
            self.model.train()
            if not self.is_distributed:
                self.model.set_retrieval_module_mode('train')
            else:
                self.model.module.set_retrieval_module_mode('train')
            

            for iteration, batch in enumerate(dataloader, 1):
                data_dict = apply_mask(
                            batch, 
                            p_mask=self.p_mask_train,
                            force_mask=self.kwargs.exp_force_all_masked,
                            cardinalities=self.dataset.cardinalities,
                            device=self.device)

                ### clear gradient
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.kwargs.model_amp):
                    if self.retrieval:
                        output = self.model(data_dict['masked_tensor'],
                                            data_dict['masked_tensor'])
                    else:
                        output = self.model(data_dict['masked_tensor'])
                    self.loss.compute(
                        output=output,
                        ground_truth_data=data_dict['ground_truth'],
                        num_or_cat=self.dataset.num_or_cat,
                        mask_matrix=data_dict['mask_matrix_list'],)
                    loss_dict = self.loss.finalize_batch_loss()
                    total_loss = loss_dict['train']['total_loss']
                    self.loss.update_losses() # for logging
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.scheduler.step()

            # we set eval to true because loss_dict here is
            # only used for logging
            loss_dict = self.loss.finalize_epoch_losses(eval_model=True)
            loss_dict = loss_dict['train']
            if self.is_distributed:
                distributed_loss_dict = {}
                for key in loss_dict:
                    gathered_tensors = [torch.zeros_like(loss_dict[key]) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_tensors, loss_dict[key])
                    gathered_tensors = torch.mean(torch.cat(gathered_tensors, dim=0))
                    distributed_loss_dict[key] = gathered_tensors
                loss_dict = distributed_loss_dict

            self.loss.reset()
            stop_time = datetime.now()
            # epoch training time in seconds
            total_epoch_time = (stop_time - start_time).total_seconds()
            self.total_train_time += total_epoch_time

            kwargs_early_stop = {"train_loss":loss_dict['total_loss'].item(),
                                 "model":self.model,
                                 "optimizer":self.optimizer,
                                 "scaler":self.scaler,
                                 "scheduler":self.scheduler,
                                 "epoch":self.epoch,
                                 "end_experiment":(self.epoch == self.num_epoch)
                                 }
            (early_stop_signal, self.model, 
            self.optimizer, self.scaler, 
            self.scheduler)= self.early_stop_counter.update(
                                **kwargs_early_stop)
                
            if early_stop_signal==EarlyStopSignal.STOP:
                if not (self.epoch == self.num_epoch):
                    print(self.early_stop_counter.early_stop_signal_message)
                    break
                else:
                    print(self.early_stop_counter.training_over_stop_signal_message)
                    print(f'Total training time took: {self.total_train_time} seconds')
                    print('This amounts to an average epoch time of {avg_time}'.format(
                    avg_time = self.total_train_time/self.epoch
                    ))
                
            if self.epoch % self.kwargs.exp_eval_every_n == 0:
                print(f'Epoch {self.epoch}: evaluation.')
                self.val()
                self.compute_metrics()
                self.dataset.set_mode('train')
                self.loss.set_mode('train')
                self.loss.reset()
                self.model.train()
                if not self.is_distributed:
                    self.model.set_retrieval_module_mode('train')
                else:
                    self.model.module.set_retrieval_module_mode('train')

        print(f'Total training time took: {self.total_train_time} seconds')
        print('This amounts to an average epoch time of {avg_time}'.format(
            avg_time = self.total_train_time/self.n_epochs
        ))
        self.training_is_over = True
        if self.is_main_process:
            self.avg_time = self.total_train_time

    def val(self,):
        
        self.dataset.set_mode('val')
        self.loss.set_mode('val')
        self.loss.reset()
        self.model.eval()
        if not self.is_distributed:
            self.model.set_retrieval_module_mode('val')
        else:
            self.model.module.set_retrieval_module_mode('val')
        self.val_ad_score = []
 
        if not self.is_distributed:
            dataloader = DataLoader(self.dataset,
                                    batch_size=self.batch_size_val,
                                    num_workers=0,
                                    shuffle=False,
                                    drop_last=False)
        else:
            dataloader = self.get_distributed_dataloader(self.batch_size_val,
                                                             self.dataset)
            dataloader.sampler.set_epoch(epoch=0)
        self.model.to(self.device)
        
        for batch in dataloader:
            batch_ad_score = []
            val_batch, label = batch
            if self.retrieval:
                if self.kwargs.exp_retrieval_num_candidate_helpers == -1:
                    train_cand = self.dataset.train
                else:
                    random_indices = np.random.choice(len(self.dataset.train[0]),
                                                      self.kwargs.exp_retrieval_num_candidate_helpers,
                                                      replace=False)
                    train_cand = [x[random_indices]
                                 for x in self.dataset.train]
                datashape_train = len(train_cand[0]), len(train_cand)
                # we take a mask that does not mask anything for the helpers
                # at this point 1: unmasked, 0 masked.
                mask_iter_train = torch.ones(datashape_train).bool()
                data_dict_train = apply_mask(
                            train_cand,
                            p_mask=0,
                            cardinalities=self.dataset.cardinalities,
                            device=self.device,
                            eval_mode=True,
                            mask_matrix=mask_iter_train)
            for index, mask_iter in enumerate(self.val_masks):
                if self.is_main_process and not self.is_batch_eval:
                    to_print= f'recon: {index+1}/{len(self.val_masks)}'
                    print(f'\n{to_print:#^80}\n')
                data_dict = apply_mask(
                            val_batch, 
                            p_mask=0,
                            eval_mode=True,
                            cardinalities=self.dataset.cardinalities,
                            device=self.device, 
                            mask_matrix=~mask_iter,
                            is_distributed=self.is_distributed)
                
                if not self.retrieval:
                    output = self.model(data_dict['masked_tensor'])
                else:
                    output = self.model(data_dict['masked_tensor'],
                                        data_dict_train['masked_tensor'])
                self.loss.compute(
                    output=output,
                    ground_truth_data=data_dict['ground_truth'],
                    num_or_cat=self.dataset.num_or_cat,
                    mask_matrix=data_dict['mask_matrix_list'],)
                total_loss = self.loss.get_individual_val_loss()
                self.loss.reset_val_loss()


                batch_ad_score.append(total_loss)
            
            stacked_ad_score = torch.stack(batch_ad_score,dim=1)
            if self.kwargs.exp_normalize_ad_loss:
                max_col = stacked_ad_score.max(0)[0]
                min_col = stacked_ad_score.min(0)[0]
                stacked_ad_score = ((stacked_ad_score - min_col) / 
                                    (max_col - min_col))
            
            if self.kwargs.exp_aggregation == 'sum':
                total_score = torch.sum(stacked_ad_score, dim=1)
            elif self.kwargs.exp_aggregation == 'max':
                total_score = torch.max(stacked_ad_score, dim=1)[0]
            self.val_ad_score.append(torch.stack([total_score, label.squeeze(0)], dim=1))
            
        if len(self.val_ad_score) > 1:
            self.val_ad_score = torch.cat(self.val_ad_score, dim=0)
        else:
            self.val_ad_score = self.val_ad_score[0]
        
        if self.is_distributed:
            gathered_tensors = [torch.zeros_like(self.val_ad_score) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_tensors, self.val_ad_score)
            self.val_ad_score = torch.cat(gathered_tensors, dim=0)
            
        # Sum along the new dimension to get a final n x 1 tensor
        self.sum_score_ad = (self.val_ad_score[:,0].to("cpu"),
                            self.val_ad_score[:,1].to("cpu"))
        
        if torch.isnan(self.sum_score_ad[0]).any():
            print('Validation score is NaN, stopping run.')
            sys.exit(1)
    
    def compute_metrics(self,):
        avg_prec = AveragePrecision(task="binary")
        auroc = AUROC(task="binary")

        val_score, val_labels = self.sum_score_ad
        val_labels = val_labels.type(torch.int)
        val_score = ((val_score - torch.mean(val_score)) 
                       / torch.std(val_score))
        ap = avg_prec(val_score, val_labels)
        auc = auroc(val_score, val_labels)

        thresh = np.percentile(val_score.numpy(), self.dataset.ratio)
        target_pred = (val_score.numpy() >= thresh).astype(int)
        target_true = val_labels.numpy().astype(int)
        (_, _, f_score_ratio, _) = prf(target_true, 
                                       target_pred,
                                       average='binary')
        
        ratio_metric_dict = {'seed': self.kwargs.iter_seed,
                             'F1':f_score_ratio,
                             'ap':ap,
                             'auc': auc,}
        print('ratio F1 Score', ratio_metric_dict)
        out_filename = (f'results_ratiof1_'
                    f'epoch_{self.epoch}__'
                    f'seed_{self.kwargs.iter_seed}.pkl')
        if self.is_main_process:
            if self.training_is_over:
                ratio_metric_dict['training_time'] = self.avg_time
            out_path = os.path.join(self.res_dir, out_filename)
            with open(out_path, 'wb') as f:
                pickle.dump(ratio_metric_dict, f)

        return ratio_metric_dict
    
    def get_distributed_dataloader(self, batchsize, dataset,):
        if not self.is_distributed:
            raise Exception

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            shuffle=False,
            num_workers=self.data_loader_nprocs,
            pin_memory=False,
            sampler=sampler)

        if self.is_main_process:
            print('Successfully loaded distributed batch dataloader.')

        return dataloader


