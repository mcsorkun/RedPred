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






df_data= pd.read_csv('reddb-smiles.csv')

SMILES = df_data["reactant_smiles"]




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
generated_dataset = generate(SMILES)

## Calculate molecular descriptors
ecfc_encoder = get_ecfc(SMILES)


#%%

# MODELING
#Import pretrained models

print(" ")
print(" ")
print(" ")
print("-------------    GCN with ECFC Encoder   -------------")


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


## predict energy from gcn model 
pred_gcne = gcne_model.predict(generated_dataset, transformers)


#---------------------------------------------------------------------------------


print(" ")
print(" ")
print(" ")
print("-------------    Keras with ECFC Encoder   -------------")


##keras model load
from keras.models import model_from_json

keras_final_model = model_from_json(open('./final_models/keras_final_model_architecture.json').read())
keras_final_model.load_weights('./final_models/keras_final_model_weights.h5')

#keras_final_model = pickle.load(open(r'./final_models/keras_final_model.txt', "rb"))

pred_keras = keras_final_model.predict(ecfc_encoder)

#---------------------------------------------------------------------------------

print(" ")
print(" ")
print(" ")
print("-------------    RF with ECFC Encoder   -------------")


rf_final_model = pickle.load(open(r'./final_models/rf_final_model.txt', "rb"))

   
pred_rf  = rf_final_model.predict(ecfc_encoder)

##reshape (n,)    ----> (n,1)

pred_rf_r = pred_rf.reshape((len(pred_rf),1))


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


weighted_pred_0_1_3=( np.power(2/(test1_mae[0]+test2_mae[0]),3) * pred_gcne + 
            np.power(2/(test1_mae[1]+test2_mae[1]),3) * pred_keras + 
            np.power(2/(test1_mae[2]+test2_mae[2]),3) * pred_rf_r ) / (
            np.power(2/(test1_mae[0]+test2_mae[0]),3) + np.power(2/(test1_mae[1]+test2_mae[1]),3) + np.power(2/(test1_mae[2]+test2_mae[2]),3)) 



#--------



#%% 
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%    SAVING THE PREDICTIONS   %%%%%%%%%%%%%%%%%%%%%%%%%%%%")



df_results = pd.DataFrame(SMILES, columns=['reactant_smiles'])
df_results["reaction_energy"]= weighted_pred_0_1_3.reshape(-1)

df_results=df_results.round(6)

# saving the dataframe
df_results.to_csv(r'.\final_models\ensemble_prediction.csv', index=False)

# df_results.to_csv("results/predicted-"+test_data_name+".csv",index=False)




print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%    END   %%%%%%%%%%%%%%%%%%%%%%%%%%%%")



