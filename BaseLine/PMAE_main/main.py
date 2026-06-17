from utils import *
from configs import *
from copy import deepcopy
from evaluate import *

from fit_pMAE import *
from fit_ReMasker import *

import random

set_all_seeds(42)
args = get_args_parser().parse_args([])

# CONFIGS
basedir = './'
dset = 'diabetes'
# key 1: pattern (full / quasi)
missing_pattern = 'full' # quasi
# key 2: seed (0~9)
seed = 2

# print( res_new.keys() )
# print( res_new['quasi'].keys() )
# print( res_new['full'].keys() )

# LOAD
proc_new = torch.load(f'{basedir}/amputation/{dset}/new_proc.pkl', weights_only=False)

X_num = proc_new['X_num']
X_cat = proc_new['X_cat']
d_numerical = proc_new['d_numerical']

# full, quasi
res_new = torch.load(f'{basedir}/amputation/{dset}/amputed.pkl', weights_only=False)
res_new_ = res_new[missing_pattern][seed]

X_init_new = res_new_['X_init'].float()
mask_new = res_new_['mask'] == 1

X_incomp_new = res_new_['X_incomp'].float()

n, d = X_init_new.shape
d_numerical = proc_new['d_numerical']
miss_cols_new = [i for i in range(d) if i not in res_new_['full_cols']]

if (n < 1000):
    batch_size = 128
elif (n < 2500):
    batch_size = 256
elif (n <5000):
    batch_size = 512
elif (n <10000):
    batch_size = 1024
elif (n <20000):
    batch_size = 2048
else:
    batch_size = 4096

print(args)

model_pmae = ProportionalMasker(args)
model_pmae.batch_size = batch_size
model_pmae.device = 'cuda:0'
model_pmae.max_epochs = 300


model_remasker = ReMasker(args)
model_remasker.batch_size = batch_size
model_remasker.device = 'cuda:0'
model_remasker.max_epochs = 300

# pMAE

model_pmae.old_loss = False
model_pmae.block_mlp = 0 # Mixer -0 / Transformer - None
model_pmae.new_imp = True

X_imputed1 = model_pmae.fit(pd.DataFrame(X_incomp_new))
                                                                                         

                                                                                         # ReMasker

model_remasker.new_imp = False
model_remasker.device = 'cuda:0'
X_imputed2 = model_remasker.fit(pd.DataFrame(X_incomp_new))
                                                                                         

                                                                                         # pMAE

X_imputed = X_imputed1

score_dict = evaluator(X_init_new, X_imputed, mask_new, d, d_numerical, 
                       miss_cols = miss_cols_new, 
                       cat_exists = (d - d_numerical > 0) )

print(score_dict)

# # ReMasker

# X_imputed = X_imputed2

# score_dict = evaluator(X_init_new, X_imputed, mask_new, d, d_numerical, 
#                        miss_cols = miss_cols_new, 
#                        cat_exists = (d - d_numerical > 0) )
# print(score_dict)