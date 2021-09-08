"""
Created on Sun Oct 18 14:54:37 2020
@author: Cihan Yatbaz, Elham Nour Gassemi
"""


print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%    IMPORTING FILES and PREPROCESSING   %%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(" ")


import pickle
import numpy
import pandas as pd

#For KERAS
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout




#Get and Arrange data

print("Get and Arrange data")
print(" ")

df_data= pd.read_csv('../Data/all_data.csv')

test1_data = df_data[df_data['data_type'] == 1]
test2_data = df_data[df_data['data_type'] == 2]


# SMILES Lists
test1_SMILES = test1_data["reactant_smiles"]
test2_SMILES = test2_data["reactant_smiles"]




# Function to create model, required for KerasClassifier


def create_model(optimizer='RMSprop', learn_rate=0.1, momentum=0.4, activation='sigmoid', dropout_rate=0.0):
    
    keras_model = Sequential()
    keras_model.add(Dense(128, input_dim=df_data.shape[1], activation=activation))
    keras_model.add(Dropout(dropout_rate))
    keras_model.add(Dense(32, activation=activation)) 
    keras_model.add(Dropout(dropout_rate))
    keras_model.add(Dense(8,activation=activation)) 
    keras_model.add(Dropout(dropout_rate))
    keras_model.add(Dense(1,activation='linear'))
    keras_model.summary()
    # Compile model
    keras_model.compile(loss='mean_squared_error', optimizer=optimizer)

    return keras_model




#%%   ENCODING and MODELING with reactant_smiles
print(" ")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%    ENCODING and MODELING with reactant_smiles   %%%%%%%%%%%%%%%%%%%%%%%%%%%%")



# import get_ecfc encoder from fingerprints  
from fingerprints import get_ecfc



## generate dataset it is diffrent from origin one  
import deepchem as dc






def generate(SMILES, verbose=False):

    featurizer = dc.feat.ConvMolFeaturizer()
    gcn = featurizer.featurize(SMILES)
    properties = [random.randint(-1,1)/100  for i in range(0,len(SMILES))]
    dataset = dc.data.NumpyDataset(X=gcn, y=np.array(properties))
    
    return dataset




#---------------------------------------------------------------------------------
### generate dataset from SMILES and function generate
test1_generated_dataset = generate(test1_SMILES)
test2_generated_dataset = generate(test2_SMILES)



## Calculate molecular descriptors
test1_encoded = get_ecfc(test1_SMILES)
test2_encoded = get_ecfc(test2_SMILES)

#%%
# MODELING

def score_results(pred_test1, pred_test2, model_name):
    
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import time
    
    model_name = model_name 
    start_time = time.time()
       
    
    # Scores of Test1 and Test2 Data 
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





#%%
#Import pretrained models

print(" ")
print(" ")
print(" ")
print("-------------    GCN Test1 and Test2 Scores   -------------")


### transformer for gcn 
filename = 'final_models/transformers.pkl'
infile = open(filename,'rb')
transformers = pickle.load(infile)
infile.close()


## model for gcn 
model_dir = 'final_models/tf_chp_initial'
gcne_model = dc.models.GraphConvModel(n_tasks=1, batch_size=100, mode='regression', dropout=0.25,model_dir= model_dir,random_seed=0)
gcne_model.restore('final_models/tf_chp_initial/ckpt-94/ckpt-197')
#print(gcne_model)
model_name = "GCN"

## predict energy from gcn model 
test1_pred_gcne = gcne_model.predict(test1_generated_dataset, transformers)
test2_pred_gcne = gcne_model.predict(test2_generated_dataset, transformers)

score_results(pred_test1=test1_pred_gcne, pred_test2=test2_pred_gcne, model_name=model_name)


#%%

#---------------------------------------------------------------------------------


print(" ")
print(" ")
print(" ")
print("-------------    Keras Test1 and Test2 Scores   -------------")
print(" ")

##keras model load
from keras.models import model_from_json

keras_final_model = model_from_json(open('./final_models/keras_final_model_architecture.json').read())
keras_final_model.load_weights('./final_models/keras_final_model_weights.h5')

model_name = "KERAS"

#keras_final_model = pickle.load(open(r'./final_models/keras_final_model.txt', "rb"))

test1_pred_keras = keras_final_model.predict(test1_encoded)
test2_pred_keras = keras_final_model.predict(test2_encoded)

score_results(pred_test1=test1_pred_keras, pred_test2=test2_pred_keras, model_name=model_name)


#%%

#---------------------------------------------------------------------------------

print(" ")
print(" ")
print(" ")
print("-------------    RF Test1 and Test2 Scores   -------------")
print(" ")

rf_final_model = pickle.load(open(r'./final_models/rf_final_model.txt', "rb"))
model_name = "RF"
   
test1_pred_rf = rf_final_model.predict(test1_encoded)
test2_pred_rf = rf_final_model.predict(test2_encoded)


##reshape (n,)    ----> (n,1)

test1_pred_rf_r = test1_pred_rf.reshape((len(test1_pred_rf),1))
test2_pred_rf_r = test2_pred_rf.reshape((len(test2_pred_rf),1))


score_results(pred_test1=test1_pred_rf_r, pred_test2=test2_pred_rf_r, model_name=model_name)

#%% Weighted Ensemble Model

#------------------------------------------------------------------------------------------------------------------

## Test 1 Experiments

test1_mae = []

test1_mae.append(0.00705) # 0 - GCN
test1_mae.append(0.00416) # 1 - Keras
test1_mae.append(0.0035) # 3 - RF



## Test 2 Experiments

test2_mae = []

test2_mae.append(0.00589) # 0 - GCN
test2_mae.append(0.00483) # 1 - Keras
test2_mae.append(0.00799) # 3 - RF



### if it is weightred prediction  check array shape?


# For Test1
test1_weighted_pred_0_1_3=( np.power(2/(test1_mae[0]+test2_mae[0]),3) * test1_pred_gcne + 
            np.power(2/(test1_mae[1]+test2_mae[1]),3) * test1_pred_keras + 
            np.power(2/(test1_mae[2]+test2_mae[2]),3) * test1_pred_rf_r ) / (
            np.power(2/(test1_mae[0]+test2_mae[0]),3) + np.power(2/(test1_mae[1]+test2_mae[1]),3) + np.power(2/(test1_mae[2]+test2_mae[2]),3)) 



#--------

# For Test2
test2_weighted_pred_0_1_3=( np.power(2/(test1_mae[0]+test2_mae[0]),3) * test2_pred_gcne + 
            np.power(2/(test1_mae[1]+test2_mae[1]),3) * test2_pred_keras + 
            np.power(2/(test1_mae[2]+test2_mae[2]),3) * test2_pred_rf_r ) / (
            np.power(2/(test1_mae[0]+test2_mae[0]),3) + np.power(2/(test1_mae[1]+test2_mae[1]),3) + np.power(2/(test1_mae[2]+test2_mae[2]),3)) 




#%% 
print(" ")
print(" ")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%    SAVING THE PREDICTIONS for ENSEMBLING  %%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(" ")
print(" ")


# ------------------------    Saving TEST 1 Predictions   ------------------------

ensemble_test1_results = pd.DataFrame(test1_SMILES, columns=['reactant_smiles'])
ensemble_test1_results["reaction_energy"]= test1_weighted_pred_0_1_3.reshape(-1)

ensemble_test1_results = ensemble_test1_results.round(6)

# saving the dataframe
ensemble_test1_results.to_csv(r'.\final_models\ensemble_test1_results.csv', index=False)



# ------------------------    Saving TEST 2 Predictions   ------------------------

ensemble_test2_results = pd.DataFrame(test2_SMILES, columns=['reactant_smiles'])
ensemble_test2_results["reaction_energy"]= test2_weighted_pred_0_1_3.reshape(-1)

ensemble_test2_results = ensemble_test2_results.round(6)

# saving the dataframe
ensemble_test2_results.to_csv(r'.\final_models\ensemble_test2_results.csv', index=False)


print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%    END   %%%%%%%%%%%%%%%%%%%%%%%%%%%%")



