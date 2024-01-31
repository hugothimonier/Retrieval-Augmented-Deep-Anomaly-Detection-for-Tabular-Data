import os
import numpy as np

from datasets.base import BaseDataset

CAT_FEATURES = [1,2,3,4,5,6,7,8,9,14]

class CampaignDataset(BaseDataset):

    '''
    https://archive.ics.uci.edu/dataset/222/bank+marketing
    
    Additional Information

    Input variables:
       # bank client data:
       1 - age (numeric)
       2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
       3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
       4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
       5 - default: has credit in default? (categorical: 'no','yes','unknown')
       6 - housing: has housing loan? (categorical: 'no','yes','unknown')
       7 - loan: has personal loan? (categorical: 'no','yes','unknown')
       # related with the last contact of the current campaign:
       8 - contact: contact communication type (categorical: 'cellular','telephone') 
       9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
      10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
      11 - duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
       # other attributes:
      12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
      13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
      14 - previous: number of contacts performed before this campaign and for this client (numeric)
      15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
       # social and economic context attributes
      16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
      17 - cons.price.idx: consumer price index - monthly indicator (numeric)     
      18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)     
      19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
      20 - nr.employed: number of employees - quarterly indicator (numeric)
    '''
    
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.is_data_loaded = False
        self.tmp_file_names = ['5_campaign.npz']
        self.name = 'campaign'
        
    def load(self,):
        
        filename = os.path.join(self.data_path, self.tmp_file_names[0])
        data = np.load(filename, allow_pickle=True)
        self.data_table  = data['X']
        self.target = ((data['y']).astype(np.int32)).reshape(-1)
        
        self.norm_samples = self.data_table [self.target == 0]
        self.anom_samples = self.data_table [self.target == 1]

        self.norm_samples = np.c_[self.norm_samples, 
                            np.zeros(self.norm_samples.shape[0])]
        self.anom_samples = np.c_[self.anom_samples, 
                                  np.ones(self.anom_samples.shape[0])]

        self.ratio = (100.0 * (0.5*len(self.norm_samples)) / ((0.5*len(self.norm_samples)) +
                                                             len(self.anom_samples)))
        self.data_table = np.concatenate((self.norm_samples, self.anom_samples),
                                         axis=0)
        self.N, self.D = self.data_table.shape
        self.D -= 1
        self.cat_features = CAT_FEATURES
        self.num_features = [ele for ele in range(self.D) if ele not in self.cat_features]
        self.cardinalities = [(ele,2) for ele in CAT_FEATURES]

        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}
        self.is_data_loaded = True

    def __repr__(self):
        repr = f'CampaignDataset(BaseDataset): {self.N} samples, {self.D} features\n'\
               f'{len(self.cat_features)} categorical features\n'\
               f'{len(self.num_features)} numerical features'
        return repr