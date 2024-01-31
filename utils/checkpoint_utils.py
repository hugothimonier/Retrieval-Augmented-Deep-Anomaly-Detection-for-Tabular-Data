import os, glob, re
from enum import Enum

import torch
from time import sleep

MODEL_CP_NAME =  "seed_{seed}_model_epoch_{epoch}.pth"

class EarlyStopSignal(Enum):
    CONTINUE = 0
    STOP = 1  # Early stopping has triggered
    END = 2  # We have reached the final epoch

class EarlyStopCounter:
    def __init__(self, kwargs, jobname, dataset_name,
                 device=None, is_distributed=False):
        '''
        Args:
        - kwargs: the arg parser
        - jobname: str for finding where the model has to be stored
                   or is already stored.
        - device: to put model back on the device is getting cached model
        '''
        self._jobname = jobname
        self.checkpoint_dir = os.path.join('./checkpoints', 
                            dataset_name, 
                            self._jobname)
        self.last_model_path = None
        self.is_distributed = is_distributed
        
        self.best_epoch = 0
        self.iter_seed = kwargs.iter_seed

        # The number of contiguous epochs for which train
        # loss has not improved (early stopping)
        self.num_inc_train_loss_epochs = 0

        # The number of times train loss has improved since last
        # caching the model -- used for (infrequent) model checkpointing
        self.num_train_improvements_since_cache = 0

        self.cadence_type = kwargs.exp_cadence_type
        err_mess = 'Chosen cadence type not allowed'
        assert self.cadence_type in ['improvement','recurrent','None'], err_mess

        self.iter_recurrent_caching = 0

        # The number of times/epochs train loss must improve prior to our
        # caching of the model if cadence_type == improvment
        # The number of epochs before between moments where the model is
        # cached
        if kwargs.exp_cache_cadence == -1:
            self.cache_cadence = float('inf')  # We will never cache
        else:
            self.cache_cadence = kwargs.exp_cache_cadence

        # Minimum train loss that the counter has observed
        self.min_train_loss = float('inf')

        # number of epochs we allow without train improvement
        # before ending training
        self.patience = kwargs.exp_patience
        self.kwargs = kwargs
        
        self.early_stop_signal_message = (
            f'Training loss has not improved '
            f'for {self.patience} contiguous epochs. '
            f'Stopping training now.')
        
        self.training_over_stop_signal_message = (
            f'Training completed.'
            f'Moving to final evaluation now.')
        
        # Only needed for distribution
        self.device = device
        if not self.is_distributed:
            self.main_process = True
        else:
            self.main_process = (self.device==0)
        if self.main_process:
            if not self.kwargs.load_from_checkpoint:
                self.clear_checkpoint_path()

    def update(self, train_loss, model, 
               optimizer, scaler, scheduler, 
               epoch, end_experiment,):
        
        if self.cadence_type == 'improvement':
            return self.update_improvement(train_loss, model,
                                            optimizer, scaler,
                                            scheduler, epoch,
                                            end_experiment,)
        elif self.cadence_type == 'recurrent':
            return self.update_recurrent(train_loss, model,
                                            optimizer, scaler,
                                            scheduler, epoch,
                                            end_experiment,)
        
        else:
            return
        
    def update_recurrent(self, train_loss, model, 
               optimizer, scaler, scheduler, 
               epoch, end_experiment):
        
        self.iter_recurrent_caching += 1
        if (self.iter_recurrent_caching > self.cache_cadence or 
            end_experiment):
            if self.iter_recurrent_caching > self.cache_cadence:
                print(f'{self.iter_recurrent_caching} iterations'
                  f'since last caching the model. Caching now.')
            else:
                print('Caching last epoch.')
            self.cache_model(train_loss,
                    model, optimizer,
                    scaler, scheduler,
                    epoch)
            self.best_epoch = epoch
            self.iter_recurrent_caching = 0
            
        # Disallow early stopping with patience == -1
        if end_experiment:
            return (EarlyStopSignal.END, model, 
                    optimizer, scaler,
                    scheduler,)
        else:
            return (EarlyStopSignal.CONTINUE, 
                    model, optimizer, 
                    scaler, scheduler,)

    def update_improvement(self, train_loss, model,
                            optimizer, scaler,
                            scheduler, epoch,
                            end_experiment):

        # if train loss improves
        if train_loss < self.min_train_loss:
            self.min_train_loss = train_loss
            self.num_inc_train_loss_epochs = 0
            self.num_train_improvements_since_cache += 1

            if (self.main_process and 
                (self.num_train_improvements_since_cache >=
                            self.cache_cadence)):
                self.clear_checkpoint_path()
                self.cache_model(train_loss,
                    model, optimizer,
                    scaler, scheduler,
                    epoch)
                print(
                    f'Training loss has improved '
                    f'{self.num_train_improvements_since_cache} times since '
                    f'last caching the model. Caching now.')
            self.num_train_improvements_since_cache = 0
            self.best_epoch = epoch
        else: #if train loss did not improve
            self.num_inc_train_loss_epochs += 1

        # Disallow early stopping with patience == -1
        if end_experiment:
            try:
                return EarlyStopSignal.END, *self.load_model(self.best_epoch, model, 
                                                        optimizer, scaler,
                                                        scheduler)
            except Exception as e:
                return EarlyStopSignal.END, model, optimizer, scaler, scheduler
            
        elif self.patience == -1:
            return (EarlyStopSignal.CONTINUE, model, 
                    optimizer, scaler, scheduler)
        elif self.num_inc_train_loss_epochs > self.patience:
            print(self.early_stop_signal_message)
            try:
                return EarlyStopSignal.STOP, *self.load_model(self.best_epoch, model, 
                                                        optimizer, scaler,
                                                        scheduler)
            except Exception:
                print('Best epoch model was not cached. Keeping last epoch.')
                return EarlyStopSignal.STOP, model, optimizer, scaler, scheduler
                
        return (EarlyStopSignal.CONTINUE, model, 
                    optimizer, scaler, scheduler)

    def clear_checkpoint_path(self):
        name_list =  glob.glob(os.path.join(self.checkpoint_dir, 
                                f"seed_{self.iter_seed}_model_epoch_*.pth"))
        if len(name_list) > 0:
            for f in name_list:
                os.remove(f)

    def cache_model(self, train_loss,
                    model, optimizer,
                    scaler, scheduler,
                    epoch,):
        if self.cadence_type == 'None':
            return
        
        retrieval_state_dict = None
        
        if not self.is_distributed:
            if model.retrieval_module is not None:
                retrieval_state_dict = model.retrieval_module.state_dict()
        else:
            if model.module.retrieval_module is not None:
                retrieval_state_dict = model.module.retrieval_module.state_dict()

        checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'retrieval_state_dict': retrieval_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss}

        self.last_model_path = os.path.join(self.checkpoint_dir,
                                            MODEL_CP_NAME.format(seed=self.iter_seed,
                                                                 epoch=epoch))
        
        # We encountered issues with the model being reliably checkpointed.
        # This is a clunky way of confirming it is / giving the script
        # "multiple tries", but, if it ain't broke...
        model_is_checkpointed = False
        counter = 0
        while model_is_checkpointed is False and counter < 10000:
            if counter % 10 == 0:
                print(f'Model checkpointing attempts: {counter}.')

            # Attempt to save
            torch.save(checkpoint_dict, self.last_model_path)

            # If we find the file there, continue on
            if os.path.isfile(self.last_model_path):
                model_is_checkpointed = True

            # If the file is not yet found, sleep to avoid bothering the server
            if model_is_checkpointed is False:
                sleep(0.5)

            counter += 1

        print(
            f'Stored epoch {epoch} model checkpoint to '
            f'{self.last_model_path}.')
        print(f'Train loss: {train_loss}.')

            
    def load_model(self, epoch, model, optimizer, scaler, scheduler):
        model_filename = os.path.join(self.checkpoint_dir,
                                MODEL_CP_NAME.format(seed=self.iter_seed,
                                                     epoch=epoch))
        print("Load %s" %model_filename)
        state_dict = torch.load(model_filename,
                                map_location=self.device)
        
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        scaler.load_state_dict(state_dict['scaler_state_dict'])
        scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        
        if not self.is_distributed:
            if model.retrieval_module is not None:
                model.retrieval_module.load_state_dict(state_dict['retrieval_state_dict'])
        else:
            if model.module.retrieval_module is not None:
                model.module.retrieval_module.load_state_dict(state_dict['retrieval_state_dict'])
        
        return model, optimizer, scaler, scheduler