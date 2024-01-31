import torch
import torch.nn as nn

from sklearn.neighbors import NearestNeighbors, BallTree, KDTree
import numpy as np

VALID_METRICS = {'brute': ['euclidean', 'manhattan', 
                         'chebyshev', 'minkowski', 
                         'wminkowski', 'seuclidean',
                         'euclidean',],
                 'kd_tree': KDTree.valid_metrics,
                 'ball_tree': BallTree.valid_metrics} 

DEEP_RETRIEVAL_TYPES = ['v-attention',
                        'attention_bsim',
                        'attention_bsim_bval',
                        'attention_bsim_nval']

class Retrieval():
    def __init__(self, type_retrieval:str, 
                 num_helpers:int, 
                 retrieval_kwargs:dict,
                 device:torch.device,
                 deterministic_selection:list=[],
                 ):
        self.type_retrieval = type_retrieval
        self.num_helpers = num_helpers
        self.retrieval_kwargs = retrieval_kwargs
        self.device = device
        self.deterministic_selection = deterministic_selection

        if self.type_retrieval.lower() == 'knn':
            self.retrieval_module = KNNRetrievalModule(
                **self.retrieval_kwargs, 
                deterministic_selection=self.deterministic_selection)
        else:
            self.retrieval_module = DeepRetrievalModule(
                **self.retrieval_kwargs,
                deterministic_selection=self.deterministic_selection,
                )
            self.retrieval_module = self.retrieval_module.to(self.device)

    def __call__(self, query_sample, candidate_samples):
        return self.retrieval_module(query_sample, candidate_samples)
    
    def __type__(self,):
        return self.type_retrieval
    
    def state_dict(self,):
        if self.type_retrieval=='knn':
            return None
        return self.retrieval_module.state_dict()
    
    def load_state_dict(self, state_dict):
        if self.type_retrieval=='knn':
            return None
        else:
            self.retrieval_module.load_state_dict(state_dict)



class DeepRetrievalModule(nn.Module):
    def __init__(self, 
                 retrieval_type:str,
                 in_dim:int,
                 out_dim:int,
                 k:int,
                 normalization_sim:bool,
                 deterministic_selection:list=[],
                 n_features:int=None):
        
        super(DeepRetrievalModule, self).__init__()
        assert retrieval_type in DEEP_RETRIEVAL_TYPES, 'Chosen retrieval type does not exist.'
        self.retrieval_type = retrieval_type
        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalization_sim = normalization_sim
        self.deterministic_selection = deterministic_selection
        self.num_features = n_features

        self.softmax = nn.Softmax(dim=1)
        self.K = nn.Linear(in_dim, out_dim)
        self.Q = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        
        self.in_linear = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.linear_nobias = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, query_sample, candidate_samples):
        if self.retrieval_type=='v-attention':
            return self.forward_attention(query_sample, candidate_samples)
        elif self.retrieval_type=='attention_bsim':
            return self.forward_attention_bsim(query_sample, candidate_samples)
        elif self.retrieval_type=='attention_bsim_bval':
            return self.forward_attention_bsim_bval(query_sample, candidate_samples)
        elif self.retrieval_type=='attention_bsim_nval':
            return self.forward_attention_bsim_nval(query_sample, candidate_samples)

    def forward_attention(self, query_sample, candidate_samples):
        q_querysample = self.Q(query_sample)
        k_candidatessamples = self.K(candidate_samples)
        similarity = torch.matmul(k_candidatessamples, 
                     q_querysample.t())
        if self.normalization_sim:
            similarity /= torch.rsqrt(torch.tensor(self.out_dim))
        similarity = similarity.t()
        v_candidatessamples = self.V(candidate_samples)
        if self.k!=-1:
            topk_values, topk_indices = torch.topk(self.softmax(similarity),
                                                self.k)
        else:
            topk_values, topk_indices = torch.topk(self.softmax(similarity),
                                                 len(k_candidatessamples))
            
        out_v_cand = v_candidatessamples[topk_indices,:]
        #reshape to original shape
        out_v_cand = out_v_cand.reshape(out_v_cand.size(0),
                                        out_v_cand.size(1),
                                        self.num_features,
                                        int(out_v_cand.size(2)/self.num_features))
        return out_v_cand, topk_values
    
    def forward_attention_bsim(self, query_sample, candidate_samples):
        k_querysample = self.K(query_sample)
        k_candidatessamples = self.K(candidate_samples)
        # for broadcasting
        k_querysample = k_querysample.unsqueeze(1)
        k_candidatessamples = k_candidatessamples.unsqueeze(0)
        similarity = -(torch.norm(k_querysample - k_candidatessamples,
                                 dim=2, p=2) ** 2)
        if self.normalization_sim:
            similarity /= torch.rsqrt(torch.tensor(self.out_dim))
        v_candidatessamples = self.V(candidate_samples)
        if self.k!=-1:
            topk_values, topk_indices = torch.topk(self.softmax(similarity),
                                                self.k)
        else:
            topk_values, topk_indices = torch.topk(self.softmax(similarity),
                                                 len(k_candidatessamples))
        out_v_cand = v_candidatessamples[topk_indices,:]
        out_v_cand = out_v_cand.reshape(out_v_cand.size(0),
                                        out_v_cand.size(1),
                                        self.num_features,
                                        int(out_v_cand.size(2)/self.num_features))
        return out_v_cand, topk_values
    
    def forward_attention_bsim_nval(self, query_sample, candidate_samples):
        '''
        In this retrieval type, we use the original encoded representation
        to be aggregated, e_x.
        '''
        k_querysample = self.K(query_sample)
        k_candidatessamples = self.K(candidate_samples)
        # for broadcasting
        k_querysample = k_querysample.unsqueeze(1)
        k_candidatessamples = k_candidatessamples.unsqueeze(0)
        similarity = -(torch.norm(k_querysample - k_candidatessamples,
                                 dim=2, p=2) ** 2)
        if self.normalization_sim:
            similarity /= torch.rsqrt(torch.tensor(self.out_dim))

        if self.k!=-1:
            topk_values, topk_indices = torch.topk(self.softmax(similarity),
                                                   self.k)
        else:
            topk_values, topk_indices = torch.topk(self.softmax(similarity),
                                                  len(k_candidatessamples))
            
        out_cand = candidate_samples[topk_indices,:]
        out_cand = out_cand.reshape(out_cand.size(0),
                                        out_cand.size(1),
                                        self.num_features,
                                        int(out_cand.size(2)/self.num_features))

        return out_cand, topk_values
    
    def forward_attention_bsim_bval(self, query_sample, candidate_samples):
        k_querysample = self.K(query_sample)
        k_candidatessamples = self.K(candidate_samples)
        # for broadcasting
        k_querysample = k_querysample.unsqueeze(1)
        k_candidatessamples = k_candidatessamples.unsqueeze(0)
        similarity = -(torch.norm(k_querysample - k_candidatessamples,
                                 dim=2, p=2) ** 2)
        if self.normalization_sim:
            similarity *= torch.rsqrt(torch.tensor(self.out_dim))
        
        v_candidatessamples = self.T(k_candidatessamples - k_querysample)
        if self.k!=-1:
            topk_values, topk_indices = torch.topk(self.softmax(similarity),
                                                self.k)
        else:
            topk_values, topk_indices = torch.topk(self.softmax(similarity),
                                                 len(k_candidatessamples))
        out_v_cand = v_candidatessamples[torch.arange(v_candidatessamples.size(0)).unsqueeze(1),
                                         topk_indices, :]
        out_v_cand = out_v_cand.reshape(out_v_cand.size(0),
                                        out_v_cand.size(1),
                                        self.num_features,
                                        int(out_v_cand.size(2)/self.num_features))
        return out_v_cand, topk_values

    def T(self, x):
        x = self.in_linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_nobias(x)
        return x

class KNNRetrievalModule():
    def __init__(self, 
                 knn_type:str,
                 k:int, 
                 metric:str='euclidean',
                 deterministic_selection:list=[]):
        if knn_type!='ann':
            assert metric in VALID_METRICS[knn_type], 'Chosen metric is not allowed.'
        assert k!=-1, 'Retrieval type cannot be knn and number of nearest neighbours -1.'
        self.knn_type = knn_type
        self.k = k
        self.metric = metric
        self.deterministic_selection = deterministic_selection

        if self.knn_type != 'ann':
            self.nn = NearestNeighbors(n_neighbors=k+1,
                                       algorithm=self.knn_type,
                                       metric=self.metric)
        else:
            self.nn = None

    def __call__(self, 
                 query_sample,
                 candidate_samples):
        query_sample_np, candidate_samples_np, _type = self.to_numpy(query_sample,
                                                                     candidate_samples)
        if self.knn_type:
            self.nn.fit(candidate_samples_np)
            distance, indices = self.nn.kneighbors(query_sample_np)
            distance = [x[1:] for x in distance] #remove self element
            indices = [x[1:] for x in indices] #remove self element
            return indices, distance

        else:
            raise ValueError('Not Implemented yet. Problem with faiss.')


    def to_numpy(self, query_sample, candidates_samples):

        if (isinstance(query_sample, np.ndarray) and 
            isinstance(candidates_samples, np.ndarray)):
            return query_sample, candidates_samples, np.ndarray
        
        if (isinstance(query_sample, torch.Tensor) and 
            isinstance(candidates_samples, torch.Tensor)):
            return query_sample.detach().cpu().numpy(), candidates_samples.detach().cpu().numpy(), torch.Tensor
        
        if (isinstance(query_sample, list) and 
            isinstance(candidates_samples, list)):
                if all(isinstance(item, np.ndarray) for item in query_sample):
                    return (np.concatenate(query_sample, axis=1),
                            np.concatenate(candidates_samples, axis=1),
                            (list, np.ndarray))
                if all(isinstance(item, torch.Tensor) for item in query_sample):
                    return (torch.cat(query_sample, axis=1).cpu().numpy(),
                            torch.cat(candidates_samples, axis=1).cpu().numpy(),
                            (list, torch.Tensor))


            
