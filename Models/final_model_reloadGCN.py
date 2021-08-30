#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 17:58:04 2021

@author: elham
"""

# -*- coding: utf-8 -*-
"""
@author: Cihan Yatbaz

Initial modeling with ecfc encoder
"""


print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%    IMPORTING FILES and PREPROCESSING   %%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(" ")

import numpy as np
import pandas as pd
import tensorflow as tf
import deepchem as dc
from deepchem.models import GraphConvModel
from deepchem.utils.save import load_from_disk
from keras.callbacks import ModelCheckpoint
#from deepchem.utils.save import save_to_disk
#from deepchem.utils.evaluate import Evaluator
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Run before every test for reproducibility
def seed_all():
    np.random.seed(123)
    tf.random.set_seed(123)





#%%

#Get and Arrange data

print("Get and Arrange data")
print(" ")

####Read data from csv files , train, test1 and test2

input_data_train ="../Data/train_s.csv"
input_data_test1= '../Data/test1.csv'
input_data_test2= '../Data/test2.csv'



#%%   ENCODING and MODELING with reactant_smiles
print(" ")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%    ENCODING and MODELING with reactant_smiles   %%%%%%%%%%%%%%%%%%%%%%%%%%%%")

# ENCODING
# chose our targer and features
####Deepchem feature extraxtion and data loaded

tasks=['reaction_energy']
featurizer = dc.feat.ConvMolFeaturizer()
loader = dc.data.CSVLoader(tasks=tasks, feature_field="reactant_smiles",featurizer=featurizer)
dataset_train=loader.featurize(input_data_train)
dataset_test1=loader.featurize(input_data_test1)
dataset_test2=loader.featurize(input_data_test2)

##Normalized datasets
transformers_train = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset_train, move_mean=True)


#There is a "split" field in the dataset file where I  defined the training/valid/test set
seed_all()
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset_train,frac_train = 0.8,frac_valid = 0.2, frac_test = 0.0,seed=0)

#Normalizes data set to zero mean
train_dataset = transformers_train.transform(train_dataset)
valid_dataset= transformers_train.transform(valid_dataset)
test1_dataset = transformers_train.transform(dataset_test1)
test2_dataset = transformers_train.transform(dataset_test2)



# MODELING


print(" ")
print(" ")
print(" ")
print("-------------    Deep Chem GCN with reactant_smiles   -------------")
## here you have to select the directory where you put the saved model 

model_dir = 'final_models/GCN_chp_final'
gcne_model = dc.models.GraphConvModel(n_tasks=1, batch_size=100, mode='regression', dropout=0.25,model_dir= model_dir,random_seed=0)
gcne_model.restore('final_models/GCN_chp_final/ckpt-94/ckpt-197')

metric = dc.metrics.Metric(dc.metrics.r2_score, mode='regression')


###Real value and predicted value for target
train_y = train_dataset.y
train_pred = gcne_model.predict(train_dataset)

test1_y = test1_dataset.y
test1_pred = gcne_model.predict(test1_dataset)

test2_y = test2_dataset.y
test2_pred = gcne_model.predict(test2_dataset)


model = "DeepChem"
  # Scores of Train Data 
train_mae = mean_absolute_error(dc.trans.undo_transforms(train_y,[transformers_train]), 
                                dc.trans.undo_transforms(train_pred,[transformers_train]))
print(train_mae)
train_rmse = mean_squared_error(dc.trans.undo_transforms(train_y,[transformers_train]), 
                                dc.trans.undo_transforms(train_pred,[transformers_train]) , squared=False)
train_r2 = r2_score(dc.trans.undo_transforms(train_y,[transformers_train]), 
                                dc.trans.undo_transforms(train_pred,[transformers_train]))
print('##########################  Scores of Train Data  ##########################')
print('Train set MAE of {}: {:.3f}'.format(model, train_mae))
print('Train set RMSE of {}: {:.3f}'.format(model, train_rmse))
print('Train set R2 Score of {}: {:.3f}'.format(model, train_r2))

print("----------------------------------------------------------------------------")

# Test1 Data
test1_mae = mean_absolute_error(dc.trans.undo_transforms(test1_y,[transformers_train]),
                                dc.trans.undo_transforms(test1_pred,[transformers_train]))
test1_rmse = mean_squared_error(dc.trans.undo_transforms(test1_y,[transformers_train]),
                                dc.trans.undo_transforms(test1_pred,[transformers_train]), squared=False)
test1_r2 = r2_score(dc.trans.undo_transforms(test1_y,[transformers_train]),
                                dc.trans.undo_transforms(test1_pred,[transformers_train]))
print('##########################  Scores of Test1 Data  ##########################')
print('Test1 set MAE of {}: {:.3f}'.format(model, test1_mae))
print('Test1 set RMSE of {}: {:.3f}'.format(model, test1_rmse))
print('Test1 set R2 Score of {}: {:.3f}'.format(model, test1_r2))

print("----------------------------------------------------------------------------")

# Test2 Data
test2_mae = mean_absolute_error(dc.trans.undo_transforms(test2_y,[transformers_train]),
                                dc.trans.undo_transforms(test2_pred,[transformers_train]))
test2_rmse = mean_squared_error(dc.trans.undo_transforms(test2_y,[transformers_train]),
                                dc.trans.undo_transforms(test2_pred,[transformers_train]), squared=False)
test2_r2 = r2_score(dc.trans.undo_transforms(test2_y,[transformers_train]),
                                dc.trans.undo_transforms(test2_pred,[transformers_train]))
print('##########################  Scores of Test2 Data  ##########################')
print('Test2 set MAE of {}: {:.3f}'.format(model, test2_mae))
print('Test2 set RMSE of {}: {:.3f}'.format(model, test2_rmse))
print('Test2 set R2 Score of {}: {:.3f}'.format(model, test2_r2))

print("----------------------------------------------------------------------------")


print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%    SAVING THE MODEL and PREDICTIONS   %%%%%%%%%%%%%%%%%%%%%%%%%%%%")


test1_yo = dc.trans.undo_transforms(test1_y,[transformers_train])
test1_predo = dc.trans.undo_transforms(test1_pred,[transformers_train])
test2_yo = dc.trans.undo_transforms(test2_y,[transformers_train])
test2_predo = dc.trans.undo_transforms(test2_pred,[transformers_train])

# ------------------------    Saving TEST 1 Predictions   ------------------------

df_test1 = pd.read_csv(input_data_test1)
df_test1_pred = pd.DataFrame()    
df_test1_pred['reaction_id'] = df_test1['reaction_id']
df_test1_pred['reactant_smiles'] = df_test1['reactant_smiles']
df_test1_pred['reaction_energy'] = test1_yo
df_test1_pred['pred_test1'] = test1_predo
df_test1_pred.to_csv('final_models/GCN_result_test1.csv')


# ------------------------    Saving TEST 2 Predictions   ------------------------

df_test2 = pd.read_csv(input_data_test2)
df_test2_pred = pd.DataFrame()    
df_test2_pred['reaction_id'] = df_test2['reaction_id']
df_test2_pred['reactant_smiles'] = df_test2['reactant_smiles']
df_test2_pred['reaction_energy'] = test2_yo
df_test2_pred['pred_test2'] = test2_predo
df_test2_pred.to_csv('final_models/GCN_result_test2.csv')




