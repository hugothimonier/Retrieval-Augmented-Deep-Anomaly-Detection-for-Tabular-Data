from typing import List

import torch
import torch.nn as nn

from retrieval import Retrieval

from utils.encode_utils import torch_cast_to_dtype

ORDER_RETRIEVAL = {'pre-embedding':0,
                   'post-embedding':1,
                   'post-encoder':2}
RET_ERR = ('Retrieval location cannot be placed'
           ' after aggregation location.')
AGG_ERR = ('agg location can be \'pre-embedding\' only'
           ' if the dataset does not contain categorical'
           ' features.')

class Model(nn.Module):
    def __init__(self,
                 ## embedding params
                 ### num
                 idx_num_features:list,
                 ### categorical embedding
                 cardinalities:list,
                 ## model parameters
                 hidden_dim:int,
                 num_layers_e:int,
                 num_heads_e:int, 
                 p_dropout:int,
                 layer_norm_eps:float,
                 gradient_clipping:float,
                 feature_type_embedding:bool,
                 feature_index_embedding:bool,
                 retrieval:Retrieval,
                 device:torch.device,
                 args:dict,
                 ):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.idx_num_features = idx_num_features
        self.idx_cat_features = [card[0] for card in cardinalities]
        self.cardinalities = cardinalities
        self.n_input_features = len(idx_num_features) + len(cardinalities)
        self.device = device
        self.mutual_update = args.model_mutual_update
        
        self.retrieval_loc = args.exp_retrieval_location
        self.retrieval_agg_loc = args.exp_retrieval_agg_location
        self.agg_lambda = args.exp_retrieval_agg_lambda
        
        assert (ORDER_RETRIEVAL[self.retrieval_loc] 
        <= ORDER_RETRIEVAL[self.retrieval_agg_loc]), RET_ERR
        assert not (self.retrieval_loc=='pre-embedding' 
                    and len(self.cardinalities)>0), AGG_ERR
        
        ## Embedding after encoding
        linear_in_embeddings = [
            nn.Linear(2, self.hidden_dim)
            for idx in idx_num_features
            ]
        self.in_embeddings = nn.ModuleList(linear_in_embeddings)
        for idx, card in cardinalities:
            self.in_embeddings.insert(idx, nn.Linear(card + 1, self.hidden_dim))
            # card + 1 because, one-hot dimension + mask_token
        del linear_in_embeddings

        ## Add feature type embedding
        ## Optionally, we construct "feature type" embeddings:
        ## a representation H is learned based on wether a feature
        ## is numerical of categorical.
        self.feature_type_embedding = feature_type_embedding
        if (self.feature_type_embedding 
            and self.cardinalities 
            and len(idx_num_features)>0):
            self.feature_types = torch_cast_to_dtype(torch.empty(
                    self.n_input_features), 'long').to(self.device)
            for feature_index in range(self.n_input_features):
                if feature_index in self.idx_num_features:
                    self.feature_types[feature_index] = 0
                elif feature_index in self.idx_cat_features:
                    self.feature_types[feature_index] = 1
                else:
                    raise Exception
            self.feature_type_embedding = nn.Embedding(
                    2, self.hidden_dim)
            print(
                f'Using feature type embedding (unique embedding for '
                f'categorical and numerical features).')
        else:
            self.feature_type_embedding = None
            
        # Feature Index Embedding
        # Optionally, learn a representation based on the index of the column.
        # Allows us to explicitly encode column identity, as opposed to
        # producing this indirectly through the per-column feature embeddings.
        self.feature_index_embedding = feature_index_embedding
        if self.feature_index_embedding:
            self.feature_indices = torch_cast_to_dtype(
                torch.arange(self.n_input_features), 'long').to(self.device)
            self.feature_index_embedding = nn.Embedding(
                self.n_input_features, self.hidden_dim)
            print(
                f'Using feature index embedding (unique embedding for '
                f'each column).')
        else:
            self.feature_index_embedding = None

        ## Embedding after decoding
        linear_out_embeddings = [
            nn.Linear(self.hidden_dim, 1)
            for _ in idx_num_features
            ]
        self.out_embedding = nn.ModuleList(linear_out_embeddings)
        for idx, card in cardinalities:
            self.out_embedding.insert(idx, nn.Linear(self.hidden_dim, card))
        del linear_out_embeddings

        self.encoder = TabularEncoder(hidden_dim=self.hidden_dim,
                                      num_layers=num_layers_e,
                                      num_heads=num_heads_e,
                                      p_dropout=p_dropout,
                                      layer_norm_eps=layer_norm_eps,
                                      activation=args.model_act_func,
                                      )
        
        #TODO
        self.retrieval_module = retrieval
        self.args = args
          
        # *** Gradient Clipping ***
        if gradient_clipping:
            clip_value = gradient_clipping
            print(f'Clipping gradients to value {clip_value}.')
            for p in self.parameters():
                p.register_hook(
                    lambda grad: torch.clamp(grad, -clip_value, clip_value))
                
    def set_retrieval_module_mode(self, mode):
        if self.retrieval_module is None:
            return None
        elif self.retrieval_module.__type__() == 'knn':
            return None
        if mode=='train':
            self.retrieval_module.retrieval_module.train()
        if mode=='val':
            print('Setting retrieval module mode to eval')
            self.retrieval_module.retrieval_module.eval()
            
                
    def in_embbed_sample(self, x):
        out = [emb(x[idx]) for idx, emb 
                in enumerate(self.in_embeddings)]
        out = torch.stack(out).squeeze(1).permute(1, 0, 2)

        if self.feature_type_embedding is not None:
            feature_type_embeddings = self.feature_type_embedding(
                self.feature_types)
            # Add a batch dimension (the rows)
            feature_type_embeddings = torch.unsqueeze(
                feature_type_embeddings, 0)
            # Tile over the rows
            feature_type_embeddings = feature_type_embeddings.repeat(
                out.size(0), 1, 1)
            # Add to X
            out = out + feature_type_embeddings

        # Compute feature index embeddings, and add them
        if self.feature_index_embedding is not None:
            feature_index_embeddings = self.feature_index_embedding(
                self.feature_indices)

            # Add a batch dimension (the rows)
            feature_index_embeddings = torch.unsqueeze(
                feature_index_embeddings, 0)

            # Tile over the rows
            feature_index_embeddings = feature_index_embeddings.repeat(
                out.size(0), 1, 1)

            # Add to X
            out = out + feature_index_embeddings

        return out
    
    def print_retrieval_module_params(self, grad=False):
        ''' used for debug only'''
        if not grad:
            print('Printing the retrieval module parameters')
            for name, param in self.retrieval_module.retrieval_module.named_parameters():
                if param.requires_grad:
                    print(name, param.data)
        else:
            print('Printing the retrieval module gradients')
            for name, param in self.retrieval_module.retrieval_module.named_parameters():
                if param.requires_grad:
                    print(name, param.grad)
    
    def out_embbed_sample(self, x):
        out = [emb(x[:,i,:]) for i, emb 
               in enumerate(self.out_embedding)]
        return out
    
    def forward(self, x, cand_samples=None):
        if self.retrieval_module is not None:
            assert cand_samples is not None
            return self.forward_retrieval(x, cand_samples)
        else:
            return self.forward_no_retrieval(x)
    
    def forward_retrieval(self, x:List[torch.tensor], 
                          cand_samples:List[torch.tensor]):
        '''
        Takes x and the candidate samples as input.
        Forward using both.
        Args:
        - x: a d-dimensional list of torch.tensor, each of dimension n_features x H_j. H_j here corresponds
             to the encoding dimension: 1 for numerical, cardinality for categorical. 
             Each torch.tensor contains the value of the feature for each sample in the batch.
        - cand_samples: a d-dimensional list of torch.tensor, each of dimension n_features x H_j. H_j here corresponds
             to the encoding dimension: 1 for numerical, cardinality for categorical. 
             Each torch.tensor contains the value of the feature for each sample in the candidate samples.
        Returns:
        A d-dimensional list of predicted feature values. Dimension of the tensors should be the same as 
        the input.
        '''
        if self.retrieval_loc =='pre-embedding':
            return self.forward_retrieval_pre_emb(x, cand_samples)
        
        if self.retrieval_loc =='post-embedding':
            return self.forward_retrieval_post_emb(x, cand_samples)
        
        if self.retrieval_loc =='post-encoder':
            return self.forward_retrieval_post_enc(x, cand_samples)
            
        
    def forward_no_retrieval(self, x:List[torch.tensor],)->List[torch.tensor]:
        '''
        Args:
        - x: a d-dimensional list of torch.tensor, each of dimension n_features x H_j. H_j here corresponds
             to the encoding dimension: 1 for numerical, cardinality for categorical. 
             Each torch.tensor contains the value of the feature for each sample in the batch.
        Returns:
        A d-dimensional list of predicted feature values. Dimension of the tensors should be the same as 
        the input.
        '''
        x = self.in_embbed_sample(x)
        x = self.encoder(x)
        out = self.out_embbed_sample(x)
        return out
    
    def forward_retrieval_pre_emb(self, x, cand_samples):
        topkvalues = None
        mask_cols = [ele[:,1] for ele in x]
        x = [ele[:,0] for ele in x]
        x = torch.stack(x, dim=1)
        cand_samples = [ele[:,:-1] for ele in cand_samples]
        cand_samples = torch.cat(cand_samples, dim=1)
        select_cand_indices, distance = self.retrieval_module(x, cand_samples)
        
        if self.retrieval_agg_loc == 'pre-embedding':
            selected_cand = [torch.stack([cand_samples[idx]],dim=0) for idx in select_cand_indices]
            selected_cand = [torch.mean(ele, dim=1) for ele in selected_cand]
            selected_cand = torch.stack(selected_cand, dim=0).squeeze(1)
            x = self.aggregate(x, selected_cand, self.agg_lambda)
            # transform back to original shape
            x = [torch.stack([x[:,idx], mask_cols[idx]], dim=1) 
                     for idx in range(x.size(1))]
            return self.forward_no_retrieval(x)
        
        elif self.retrieval_agg_loc in ['post-embedding', 'post-encoder']:
            x = self.in_embbed_sample(x)
            cand_samples = self.in_embbed_sample(cand_samples)
            
            if self.retrieval_agg_loc=='post-encoder':
                # if aggregation is post-encoder, samples have to be passed
                # through the encoder before aggregation
                x = self.encoder(x)
                cand_samples = self.encoder(cand_samples)
            
            selected_cand = [torch.stack([cand_samples[idx]],dim=0) for 
                                  idx in select_cand_indices]
            selected_cand = [torch.mean(ele, dim=1) for ele in selected_cand]
            selected_cand = torch.stack(selected_cand, dim=0).squeeze(1)
            x = self.aggregate(x, selected_cand,self.agg_lambda)
            
            if self.retrieval_agg_loc=='post-embedding':
                # if aggregation is post-embedding, at this stage,
                # sample has not been passed through the encoder
                x = self.encoder(x)
                
            return self.out_embbed_sample(x)
        
    def forward_retrieval_post_emb(self, x, cand_samples):
        '''
        Takes x and the candidate samples as input.
        Choose, in the embedded data space the helpers using knn or attention.
        Aggregate the samples in the chosen location.
        '''
        x = self.in_embbed_sample(x)
        cand_samples = self.in_embbed_sample(cand_samples)
        x_flatten = x.reshape(x.size(0), -1)
        cand_flatten = cand_samples.reshape(cand_samples.size(0), -1)
            
        if self.retrieval_module.__type__() == 'knn':
            select_cand_indices, _ = self.retrieval_module(x_flatten, 
                                                           cand_flatten)
            if self.retrieval_agg_loc == 'post-encoder':
                # if aggregation is post-encoder, samples have to be passed
                # through the encoder before aggregation
                x = self.encoder(x)
                cand_samples = self.encoder(cand_samples)
                
            selected_cand = [torch.stack([cand_samples[idx]],dim=0) for 
                                  idx in select_cand_indices]
            selected_cand = [torch.mean(ele, dim=1) for ele in selected_cand]
            selected_cand = torch.stack(selected_cand, dim=0).squeeze(1)
            x = self.aggregate(x, selected_cand, self.agg_lambda)
            
            if self.retrieval_agg_loc == 'post-embedding':
                # if aggregation is post-embedding, at this stage,
                # sample has not been passed through the encoder
                x = self.encoder(x)
        
        else:
            selected_cand, topkvalues = self.retrieval_module(x_flatten, 
                                                              cand_flatten)
            if self.retrieval_agg_loc == 'post-encoder':
                # if aggregation is post-encoder, samples have to be passed
                # through the encoder before aggregation
                x = self.encoder(x)
                cand_samples = self.encoder(cand_samples)
                
            topkvalues = topkvalues.unsqueeze(2).unsqueeze(3)
            selected_cand = selected_cand * topkvalues
            # for weighted mean
            selected_cand = torch.sum(selected_cand, dim=1)
            topkvalues_sum = torch.sum(topkvalues, dim=1)
            selected_cand = (selected_cand 
                             / topkvalues_sum)
            x = self.aggregate(x, selected_cand, self.agg_lambda)
            if self.retrieval_agg_loc == 'post-embedding':
                # if aggregation is post-embedding, at this stage,
                # sample has not been passed through the encoder
                x = self.encoder(x)
                
        return self.out_embbed_sample(x)
    
    def forward_retrieval_post_enc(self, x, cand_samples):
        '''
        Takes x and the candidate samples as input.
        Choose, in the encoded data space the helpers using knn or attention.
        Aggregate the samples in the chosen location.
        '''
        
        x = self.in_embbed_sample(x)
        x = self.encoder(x)
        
        if not self.mutual_update:
            cand_samples = [x.detach() for x in cand_samples]
            cand_samples = self.in_embbed_sample(cand_samples)
            cand_samples = cand_samples.detach()
            cand_samples = self.encoder(cand_samples)
        else:
            cand_samples = self.in_embbed_sample(cand_samples)
            cand_samples = self.encoder(cand_samples)

        if self.retrieval_module.__type__() == 'knn':
            x_flatten = x.reshape(x.size(0), -1)
            cand_flatten = cand_samples.reshape(cand_samples.size(0), -1)
            select_cand_indices, _ = self.retrieval_module(x_flatten, cand_flatten)
            
            selected_cand = [torch.stack([cand_samples[idx]],dim=0) for 
                             idx in select_cand_indices]
            selected_cand = [torch.mean(ele, dim=1) for ele in selected_cand]
            selected_cand = torch.stack(selected_cand, dim=0).squeeze(1)
            x = self.aggregate(x, selected_cand, self.agg_lambda)
            
        else:
            x_flatten = x.reshape(x.size(0), -1)
            cand_flatten = cand_samples.reshape(cand_samples.size(0), -1)
            selected_cand, topkvalues = self.retrieval_module(x_flatten, cand_flatten)
            # for broadcasting
            topkvalues = topkvalues.unsqueeze(2).unsqueeze(3)
            selected_cand = selected_cand * topkvalues
            # for weighted mean
            selected_cand = torch.sum(selected_cand, dim=1)
            topkvalues_sum = torch.sum(topkvalues, dim=1)
            selected_cand = (selected_cand 
                             / topkvalues_sum)
            x = self.aggregate(x, selected_cand, self.agg_lambda)

        return self.out_embbed_sample(x)
    
    def aggregate(self, x, cand_samples, lambda_):
        return (1 - lambda_) * x + lambda_ * cand_samples

class TabularEncoder(nn.Module):
    def __init__(self, hidden_dim:int, num_layers:int, 
                 num_heads:int, p_dropout:float,
                 layer_norm_eps:float,
                 activation:str):
        super(TabularEncoder, self).__init__()
        self.transformer_layers = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                             nhead=num_heads,
                                                             batch_first=True,
                                                             activation=activation)
        self.transformer = nn.TransformerEncoder(self.transformer_layers, num_layers,)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.layernorm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(p=p_dropout)
        self.dropout2 = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = self.transformer(x)
        x = self.dropout1(x)
        x = self.layernorm1(x)
        x = self.fc(x)
        x = self.dropout2(x)
        x = self.layernorm2(x)
        return x