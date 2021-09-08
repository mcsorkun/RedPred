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

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
#tf.random.set_seed(seed_value)
# for later versions: 
tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
"""from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
"""
# for later versions:
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


# Resource
# https://stackoverflow.com/questions/45230448/how-to-get-reproducible-result-when-running-keras-with-tensorflow-backend?rq=1
# https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras/52897216#52897216



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
print("-------------    Keras with reactant_smiles & ecfc fingerprint   -------------")
### Keras with reactant_smiles & ecfc fingerprint

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error



def BuildModel(input_dim=None):
    
    def model():
        keras_model = Sequential()
        keras_model.add(Dense(128, input_dim=input_dim,activation='relu')) 
        keras_model.add(Dense(32, activation='relu')) 
        keras_model.add(Dense(8,activation='relu')) 
        keras_model.add(Dense(1,activation='linear'))
        keras_model.summary()
        keras_model.compile(loss="mean_squared_error", optimizer="adam")   
        return keras_model
    return model




# Model
model = KerasRegressor(build_fn=BuildModel(input_dim = 2048), epochs=10, batch_size=5)
model_name = "KERAS" 

# Training
modeling(train_encoded=train_encoded, test1_encoded=test1_encoded, test2_encoded=test2_encoded, model=model, model_name=model_name)




#%% 
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%    SAVING THE MODEL and PREDICTIONS   %%%%%%%%%%%%%%%%%%%%%%%%%%%%")


# ------------------------    Saving TEST 1 Predictions   ------------------------
pred_test1 = model.predict(test1_encoded.values)
keras_result_test1 = test1_data[["reaction_id", "reactant_smiles", "reaction_energy"]]

keras_result_test1["pred_test1"] = pred_test1

keras_result_test1.to_csv(r'.\final_models\keras_result_test1.csv', index=False)



# ------------------------    Saving TEST 2 Predictions   ------------------------

pred_test2 = model.predict(test2_encoded.values)
keras_result_test2 = test2_data[["reaction_id", "reactant_smiles", "reaction_energy"]]
keras_result_test2["pred_test2"] = pred_test2

# saving the dataframe
keras_result_test2.to_csv(r'.\final_models\keras_result_test2.csv', index=False)



# ------------------------    Saving the Model   ------------------------

"""
import pickle
# save
pickle.dump(model, open(r'.\final_models\keras_final_model.txt', "wb"))
"""

json_model = model.model.to_json()
open(r'.\final_models\keras_final_model_architecture.json', 'w').write(json_model)
# saving weights
model.model.save_weights(r'.\final_models\keras_final_model_weights.h5', overwrite=True)


"""
#### Read Saved model 
from keras.models import model_from_json

keras_final_model = model_from_json(open('keras_final_model_architecture.json').read())
keras_final_model.load_weights('keras_final_model_weights.h5')
keras_final_model

s_pred_test1 = keras_final_model.predict(test1_encoded.values)
s_pred_test1

"""












