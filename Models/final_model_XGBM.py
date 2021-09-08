# -*- coding: utf-8 -*-
"""
@author: Cihan Yatbaz

Initial modeling with ecfc encoder
"""


print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%    IMPORTING FILES and PREPROCESSING   %%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(" ")



# Seed values for reproducibility

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 1

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']= '0'

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np 
np.random.seed(seed_value)



#%%


# Call get_ecfc encoder

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# import get_ecfc encoder from fingerprints  
from fingerprints import get_ecfc



#%%

#Get and Arrange data

print("Get and Arrange data")
print(" ")

df_data= pd.read_csv('../Data/all_data.csv')

train_data = df_data[df_data['data_type'] == 0]
test1_data = df_data[df_data['data_type'] == 1]
test2_data = df_data[df_data['data_type'] == 2]






#%%     MODELING

def modeling(train_encoded, test1_encoded, test2_encoded, model, model_name):
    
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import time
    
    model_name = model_name 
    start_time = time.time()
    
    # Training
    X = train_encoded
    y = train_data['reaction_energy']
    
    model.fit(X.values, y)
    
    # Predicting
    pred_train = model.predict(train_encoded.values)
    pred_test1 = model.predict(test1_encoded.values)
    pred_test2 = model.predict(test2_encoded.values)
    
    
    # Scores of Train Data 
    tr_mae = mean_absolute_error(y, pred_train)
    tr_rmse = mean_squared_error(y ,pred_train , squared=False)
    tr_r2 = r2_score(y, pred_train)
    print('##########################  Scores of Train Data  ##########################')
    print('Train set MAE of {}: {:.3f}'.format(model_name, tr_mae))
    print('Train set RMSE of {}: {:.3f}'.format(model_name, tr_rmse))
    print('Train set R2 Score of {}: {:.3f}'.format(model_name, tr_r2))
    
    print("----------------------------------------------------------------------------")
    
    # Test1 Data
    test1_mae = mean_absolute_error(test1_data['reaction_energy'], pred_test1)
    test1_rmse = mean_squared_error(test1_data['reaction_energy'], pred_test1, squared=False)
    test1_r2 = r2_score(test1_data['reaction_energy'], pred_test1)
    print('##########################  Scores of Test1 Data  ##########################')
    print('Test1 set MAE of {}: {:.3f}'.format(model_name, test1_mae))
    print('Test1 set RMSE of {}: {:.3f}'.format(model_name, test1_rmse))
    print('Test1 set R2 Score of {}: {:.3f}'.format(model_name, test1_r2))
    
    print("----------------------------------------------------------------------------")
    
    # Test2 Data
    test2_mae = mean_absolute_error(test2_data['reaction_energy'], pred_test2)
    test2_rmse = mean_squared_error(test2_data['reaction_energy'], pred_test2, squared=False)
    test2_r2 = r2_score(test2_data['reaction_energy'], pred_test2)
    print('##########################  Scores of Test2 Data  ##########################')
    print('Test2 set MAE of {}: {:.3f}'.format(model_name, test2_mae))
    print('Test2 set RMSE of {}: {:.3f}'.format(model_name, test2_rmse))
    print('Test2 set R2 Score of {}: {:.3f}'.format(model_name, test2_r2))
    
    print("----------------------------------------------------------------------------")

    elapsed_time = time.time() - start_time
    print('##########################  Details  ##########################')
    print(f'{elapsed_time:.2f}s elapsed during modeling')



#%%   ENCODING and MODELING with reactant_smiles
print(" ")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%    ENCODING and MODELING with reactant_smiles   %%%%%%%%%%%%%%%%%%%%%%%%%%%%")

# ENCODING
train_encoded = get_ecfc(train_data["reactant_smiles"])
test1_encoded = get_ecfc(test1_data["reactant_smiles"])
test2_encoded = get_ecfc(test2_data["reactant_smiles"])



# MODELING


print(" ")
print(" ")
print(" ")
print("-------------    XGBM with reactant_smiles & ecfc fingerprint   -------------")
### XGBM with reactant_smiles & ecfc fingerprint
import xgboost as xgb

# Model
model = xgb.XGBRegressor(random_state=1, learning_rate=0.1, max_depth=7, reg_lambda=1.2)
model_name = "XGBM"

# Training
modeling(train_encoded=train_encoded, test1_encoded=test1_encoded, test2_encoded=test2_encoded, model=model, model_name=model_name)



#%%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%    SAVING THE MODEL and PREDICTIONS   %%%%%%%%%%%%%%%%%%%%%%%%%%%%")


# ------------------------    Saving TEST 1 Predictions   ------------------------
pred_test1 = model.predict(test1_encoded.values)
xgbm_result_test1 = test1_data[["reaction_id", "reactant_smiles", "reaction_energy"]]

xgbm_result_test1["pred_test1"] = pred_test1

xgbm_result_test1.to_csv(r'.\final_models\xgbm_result_test1.csv', index=False)



# ------------------------    Saving TEST 2 Predictions   ------------------------

pred_test2 = model.predict(test2_encoded.values)
xgbm_result_test2 = test2_data[["reaction_id", "reactant_smiles", "reaction_energy"]]
xgbm_result_test2["pred_test2"] = pred_test2

# saving the dataframe
xgbm_result_test2.to_csv(r'.\final_models\xgbm_result_test2.csv', index=False)



# ------------------------    Saving the Model   ------------------------

import pickle
# save
pickle.dump(model, open(r'.\final_models\xgbm_final_model.txt', "wb"))



# load
#xgbm_final_model = pickle.load(open(r'.\final_models\xgbm_final_model.txt', "rb"))


#s_pred_test1 = xgbm_final_model.predict(test1_encoded.values)
#s_pred_test1

