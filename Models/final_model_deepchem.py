#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:38:09 2021

@author: elham
"""

## import our libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import deepchem as dc
from deepchem.splits.splitters import RandomSplitter
from deepchem.models import GraphConvModel
from deepchem.utils.evaluate import Evaluator
from deepchem.models.losses import Loss
"""
This is an initial GCN model to train, with guessed parameters for batch_size
 and dropuout.
We use suffled train dataset and two test data sets.    
""" 

## import our train data from file
import time
start_time = time.time()


#Get and Arrange data

print("Get and Arrange data")
print(" ")

df_data= pd.read_csv('all_data.csv')

input_data_train = df_data[df_data['data_type'] == 0]
input_data_test1 = df_data[df_data['data_type'] == 1]
input_data_test2 = df_data[df_data['data_type'] == 2]

"""

####Read data from csv files , train, test1 and test2

input_data_train ="../final_data/train_s.csv"
input_data_test1= '../final_data/test1.csv'
input_data_test2= '../final_data/test2.csv'
 """

# Run before every test for reproducibility
def seed_all():
    np.random.seed(123)
    tf.random.set_seed(123)
    
    
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
# transformers_test1 = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset_test1, move_mean=True)
# transformers_test2 = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset_test2,move_mean=True)


#There is a "split" field in the dataset file where I  defined the training/valid/test set
seed_all()
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset_train,frac_train = 0.8,frac_valid = 0.2, frac_test = 0.0,seed=0)

#Normalizes data set to zero mean
train_dataset = transformers_train.transform(train_dataset)
valid_dataset= transformers_train.transform(valid_dataset)
test1_dataset = transformers_train.transform(dataset_test1)
test2_dataset = transformers_train.transform(dataset_test2)


## chose a directory to save train model
model_dir = "./tf_chp_initial"

model = GraphConvModel(n_tasks=1, batch_size=100, mode='regression', dropout=0.25,model_dir= model_dir,random_seed=0)


# list of evaluation metrics = ["pearson_r2_score", "r2_score", "mean_squared_error","mean_absolute_error", "rms_score", "mae_score", "pearsonr",
#                            "concordance_index" ]

metric = dc.metrics.Metric(dc.metrics.r2_score, mode='regression')
ckpt = tf.train.Checkpoint(step=tf.Variable(1))
manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=20)

# max_checkpoints_to_keep  = 5 (defult)
num_epochs = 100
losses_train = []
score_train = []
score_valid = []
for i in range(num_epochs):
    loss_train = model.fit(train_dataset, nb_epoch=1,deterministic=True)
    ckpt.step.assign_add(1)
    save_path = manager.save()
    print("Saved checkpoint for step {}: {} ".format(int(ckpt.step), save_path))
    model.save_checkpoint(max_checkpoints_to_keep=20 , model_dir = save_path )
    
    
    R2_valid = model.evaluate(valid_dataset,[metric])['r2_score']
    R2_train = model.evaluate(train_dataset,[metric])['r2_score']
    print("Epoch %d loss_train: %f R2_train %f R2_valid: %f  " % (i, loss_train,R2_train,R2_valid))

    losses_train.append(loss_train)
    score_train.append(R2_train)
    score_valid.append(R2_valid)


## save loss and score in a file
df = pd.DataFrame(list(zip(losses_train,score_train,score_valid)),columns = ['train-loss','train-R2score','valid-R2score'])
df.to_csv('loss-score-train-valid.csv')
print("--- %s seconds ---" % (time.time() - start_time))

## figure for loss and score
import matplotlib.pyplot as plt

#print(len(losses))
fig, ax = plt.subplots(2, sharex='col', sharey='row')
x = range(num_epochs)
y_train = losses_train
ax[0].plot(x, y_train, c='b', alpha=0.6, label='loss_train')
ax[0].set(xlabel='epoch', ylabel='loss')

y_test = score_valid
ax[1].plot(x, y_test,c='r', alpha=0.6, label='score_valid')
ax[1].set(xlabel='epoch', ylabel='R2 score')
p1 = plt.show()


##prediction value for train data
train_y = train_dataset.y
train_pred = model.predict(train_dataset)
## using test data to evaluate model
test1_y = test1_dataset.y
test1_pred = model.predict(test1_dataset)

test2_y = test2_dataset.y
test2_pred = model.predict(test2_dataset)

train_smile = train_dataset.ids
train_yo = dc.trans.undo_transforms(train_y,[transformers_train])
train_predo = dc.trans.undo_transforms(train_pred,[transformers_train])
train_res = zip (train_smile,train_yo,train_predo)
df_train_pred = pd.DataFrame(train_res, columns=('smile','train_y','train_pred'))

test1_yo = dc.trans.undo_transforms(test1_y,[transformers_train])
test1_predo = dc.trans.undo_transforms(test1_pred,[transformers_train])
test1_smile = dataset_test1.ids
test1_res = zip(test1_smile,test1_yo,test1_predo)
df_test1_pred = pd.DataFrame(test1_res,columns=('smile','test1_y','test1_pred'))
                         

test2_yo = dc.trans.undo_transforms(test2_y,[transformers_train])
test2_predo = dc.trans.undo_transforms(test2_pred,[transformers_train])
test2_smile = dataset_test2.ids  
test2_res = zip(test2_smile,test2_yo,test2_predo)
df_test2_pred = pd.DataFrame(test2_res,columns=('smile','test2_y','test2_pred'))
                      
 
df_train_pred.to_csv('init_pred_train.csv')
df_test1_pred.to_csv('init_pred_test1.csv')
df_test2_pred.to_csv('init_pred_test2.csv')


## evaluation using sklearn 
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
dcmodel = "DeepChem"
 # Scores of Train Data 
train_mae = mean_absolute_error(dc.trans.undo_transforms(train_y,[transformers_train]), 
                                dc.trans.undo_transforms(train_pred,[transformers_train]))
train_rmse = mean_squared_error(dc.trans.undo_transforms(train_y,[transformers_train]), 
                                dc.trans.undo_transforms(train_pred,[transformers_train]) , squared=False)
train_r2 = r2_score(dc.trans.undo_transforms(train_y,[transformers_train]), 
                                dc.trans.undo_transforms(train_pred,[transformers_train]))
print('##########################  Scores of Train Data  ##########################')
print('Train set MAE of {}: {:.3f}'.format(dcmodel, train_mae))
print('Train set RMSE of {}: {:.3f}'.format(dcmodel, train_rmse))
print('Train set R2 Score of {}: {:.3f}'.format(dcmodel, train_r2))

print("----------------------------------------------------------------------------")

# Test1 Data
test1_mae = mean_absolute_error(dc.trans.undo_transforms(test1_y,[transformers_train]),
                                dc.trans.undo_transforms(test1_pred,[transformers_train]))
test1_rmse = mean_squared_error(dc.trans.undo_transforms(test1_y,[transformers_train]),
                                dc.trans.undo_transforms(test1_pred,[transformers_train]), squared=False)
test1_r2 = r2_score(dc.trans.undo_transforms(test1_y,[transformers_train]),
                                dc.trans.undo_transforms(test1_pred,[transformers_train]))
print('##########################  Scores of Test1 Data  ##########################')
print('Test1 set MAE of {}: {:.3f}'.format(dcmodel, test1_mae))
print('Test1 set RMSE of {}: {:.3f}'.format(dcmodel, test1_rmse))
print('Test1 set R2 Score of {}: {:.3f}'.format(dcmodel, test1_r2))

print("----------------------------------------------------------------------------")

# Test2 Data
test2_mae = mean_absolute_error(dc.trans.undo_transforms(test2_y,[transformers_train]),
                                dc.trans.undo_transforms(test2_pred,[transformers_train]))
test2_rmse = mean_squared_error(dc.trans.undo_transforms(test2_y,[transformers_train]),
                                dc.trans.undo_transforms(test2_pred,[transformers_train]), squared=False)
test2_r2 = r2_score(dc.trans.undo_transforms(test2_y,[transformers_train]),
                                dc.trans.undo_transforms(test2_pred,[transformers_train]))
print('##########################  Scores of Test2 Data  ##########################')
print('Test2 set MAE of {}: {:.3f}'.format(dcmodel, test2_mae))
print('Test2 set RMSE of {}: {:.3f}'.format(dcmodel, test2_rmse))
print('Test2 set R2 Score of {}: {:.3f}'.format(dcmodel, test2_r2))

print("----------------------------------------------------------------------------")



## evaluation using deepchem model metrics 

metric_mae = dc.metrics.Metric(dc.metrics.mean_absolute_error)
train_mae = model.evaluate(train_dataset, [metric_mae],[transformers_train])

test1_mae = model.evaluate(test1_dataset, [metric_mae],[transformers_train])

test2_mae = model.evaluate(test2_dataset, [metric_mae],[transformers_train])


metric_mse = dc.metrics.Metric(dc.metrics.mean_squared_error)

train_mse = model.evaluate(train_dataset, [metric_mse],[transformers_train])

test1_mse = model.evaluate(test1_dataset, [metric_mse],[transformers_train])

test2_mse = model.evaluate(test2_dataset, [metric_mse],[transformers_train])


metric_r2 = dc.metrics.Metric(dc.metrics.r2_score)
train_r2 = model.evaluate(train_dataset, [metric_r2],[transformers_train])

test1_r2 = model.evaluate(test1_dataset, [metric_r2],[transformers_train])

test2_r2 = model.evaluate(test2_dataset, [metric_r2],[transformers_train])


print("Train evaluation")
print(train_mae)
print(train_mse)
print(train_r2)

print("Test1 evaluation")
print(test1_mae)
print(test1_mse)
print(test1_r2)


print("Test2 evaluation")
print(test2_mae)
print(test2_mse)
print(test2_r2)

## plot for all train and test data
###Show the results in figure
plt.figure(1)
plt.xlim((-0.2,0.3))
plt.ylim((-0.2,0.2))
plt.title("GCN Regression Prediction")
plt.xlabel("Real E (hartree)")
plt.ylabel("Predicted E (hartree)")
plt.grid(color='w', linestyle='--', linewidth=1)
plt.scatter(dc.trans.undo_transforms(train_y,[transformers_train]),
                                dc.trans.undo_transforms(train_pred,[transformers_train]), 
            color="blue", alpha=0.8, label="train")
plt.scatter(dc.trans.undo_transforms(test1_y,[transformers_train]),
                                dc.trans.undo_transforms(test1_pred,[transformers_train]), 
            color="red", alpha=0.8, label="test1")
plt.scatter(dc.trans.undo_transforms(test2_y,[transformers_train]),
                                dc.trans.undo_transforms(test2_pred,[transformers_train]), 
            color="lightgreen", alpha=0.8, label="test2")
plt.legend(loc = 'best')
plt.show()
