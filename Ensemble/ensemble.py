# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:09:36 2021

@author: mcsor
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


gcn_test1 = pd.read_csv('final_models/GCN_pred_test1.csv')
gcn_test1=gcn_test1.sort_values(by=['reaction_id'])
gcn_test1.reset_index(drop=True, inplace=True)        

keras_test1  = pd.read_csv('final_models/keras_result_test1.csv')
keras_test1 =keras_test1.sort_values(by=['reaction_id'])
keras_test1.reset_index(drop=True, inplace=True)   

lgbm_test1  = pd.read_csv('final_models/lgbm_result_test1.csv')
lgbm_test1=lgbm_test1.sort_values(by=['reaction_id'])
lgbm_test1.reset_index(drop=True, inplace=True)   

rf_test1 = pd.read_csv('final_models/rf_result_test1.csv')
rf_test1=rf_test1.sort_values(by=['reaction_id'])
rf_test1.reset_index(drop=True, inplace=True)   

xgbm_test1 = pd.read_csv('final_models/xgbm_result_test1.csv')
xgbm_test1=xgbm_test1.sort_values(by=['reaction_id'])
xgbm_test1.reset_index(drop=True, inplace=True)   


all_results=[]
all_results.append(gcn_test1)
all_results.append(keras_test1)
all_results.append(lgbm_test1)
all_results.append(rf_test1)
all_results.append(xgbm_test1)

test1_mae = []

test1_mae.append(mean_absolute_error(gcn_test1["reaction_energy"], gcn_test1["pred_test1"]))
test1_mae.append(mean_absolute_error(keras_test1["reaction_energy"], keras_test1["pred_test1"]))
test1_mae.append(mean_absolute_error(lgbm_test1["reaction_energy"], lgbm_test1["pred_test1"]))
test1_mae.append(mean_absolute_error(rf_test1["reaction_energy"], rf_test1["pred_test1"]))
test1_mae.append(mean_absolute_error(xgbm_test1["reaction_energy"], xgbm_test1["pred_test1"]))



for i in range (0,len(all_results)):
    for j in range(i+1,len(all_results)):
        print(i,"-",j,":",mean_absolute_error(all_results[i]["pred_test1"], all_results[j]["pred_test1"]))
        


for i in range (0,len(all_results)):
    for j in range(i+1,len(all_results)):
        pred_x_x=(all_results[i]["pred_test1"]+all_results[j]["pred_test1"]) / 2
        # print(i,"-",j,":",mean_absolute_error(pred_x_x, all_results[0]["reaction_energy"]))   
        # print(mean_squared_error(pred_x_x, all_results[0]["reaction_energy"],squared=False))  
        print(r2_score( all_results[0]["reaction_energy"],pred_x_x)) 
        
        
for i in range (0,len(all_results)):
    for j in range(i+1,len(all_results)):
        for k in range(j+1,len(all_results)):
            pred_x_x=(all_results[i]["pred_test1"]+all_results[j]["pred_test1"]+all_results[k]["pred_test1"]) / 3
            # print(i,"-",j,"-",k)   
            # print(mean_absolute_error(pred_x_x, all_results[0]["reaction_energy"]))  
            # print(mean_squared_error(pred_x_x, all_results[0]["reaction_energy"],squared=False))  
            print(r2_score( all_results[0]["reaction_energy"],pred_x_x)) 
        

## Test 2 Experimens


gcn_test2 = pd.read_csv('final_models/GCN_pred_test2.csv')
gcn_test2=gcn_test2.sort_values(by=['reaction_id'])
gcn_test2.reset_index(drop=True, inplace=True)        

keras_test2  = pd.read_csv('final_models/keras_result_test2.csv')
keras_test2 =keras_test2.sort_values(by=['reaction_id'])
keras_test2.reset_index(drop=True, inplace=True)   

lgbm_test2  = pd.read_csv('final_models/lgbm_result_test2.csv')
lgbm_test2=lgbm_test2.sort_values(by=['reaction_id'])
lgbm_test2.reset_index(drop=True, inplace=True)   

rf_test2 = pd.read_csv('final_models/rf_result_test2.csv')
rf_test2=rf_test2.sort_values(by=['reaction_id'])
rf_test2.reset_index(drop=True, inplace=True)   

xgbm_test2 = pd.read_csv('final_models/xgbm_result_test2.csv')
xgbm_test2=xgbm_test2.sort_values(by=['reaction_id'])
xgbm_test2.reset_index(drop=True, inplace=True)   


all_results2=[]
all_results2.append(gcn_test2)
all_results2.append(keras_test2)
all_results2.append(lgbm_test2)
all_results2.append(rf_test2)
all_results2.append(xgbm_test2)

test2_mae = []

test2_mae.append(mean_absolute_error(gcn_test2["reaction_energy"], gcn_test2["pred_test2"]))
test2_mae.append(mean_absolute_error(keras_test2["reaction_energy"], keras_test2["pred_test2"]))
test2_mae.append(mean_absolute_error(lgbm_test2["reaction_energy"], lgbm_test2["pred_test2"]))
test2_mae.append(mean_absolute_error(rf_test2["reaction_energy"], rf_test2["pred_test2"]))
test2_mae.append(mean_absolute_error(xgbm_test2["reaction_energy"], xgbm_test2["pred_test2"]))



for i in range (0,len(all_results2)):
    for j in range(i+1,len(all_results2)):
        print(i,"-",j,":",mean_absolute_error(all_results2[i]["pred_test2"], all_results2[j]["pred_test2"]))
        


for i in range (0,len(all_results2)):
    for j in range(i+1,len(all_results2)):
        pred_x_x=(all_results2[i]["pred_test2"]+all_results2[j]["pred_test2"]) / 2
        # print(mean_absolute_error(pred_x_x, all_results2[0]["reaction_energy"]))   
        # print(mean_squared_error(pred_x_x, all_results2[0]["reaction_energy"],squared=False))  
        print(r2_score( all_results2[0]["reaction_energy"],pred_x_x)) 


for i in range (0,len(all_results2)):
    for j in range(i+1,len(all_results2)):
        for k in range(j+1,len(all_results2)):
            pred_x_x=(all_results2[i]["pred_test2"]+all_results2[j]["pred_test2"]+all_results2[k]["pred_test2"]) / 3
            # print(i,"-",j,"-",k)   
            # print(mean_absolute_error(pred_x_x, all_results2[0]["reaction_energy"]))  
            # print(mean_squared_error(pred_x_x, all_results2[0]["reaction_energy"],squared=False))  
            print(r2_score( all_results2[0]["reaction_energy"],pred_x_x)) 
        

for i in range (0,len(all_results2)):
    for j in range(i+1,len(all_results2)):
        for k in range(j+1,len(all_results2)):
            for l in range(k+1,len(all_results2)):
                pred_x_x=(all_results2[i]["pred_test2"]+all_results2[j]["pred_test2"]+all_results2[k]["pred_test2"]+all_results2[l]["pred_test2"]) / 4
                # print(i,"-",j,"-",k,"-",l)   
                # print(mean_absolute_error(pred_x_x, all_results2[0]["reaction_energy"]))  
                # print(mean_squared_error(pred_x_x, all_results2[0]["reaction_energy"],squared=False))  
                print(r2_score( all_results2[0]["reaction_energy"],pred_x_x)) 
            


## Ingore Outliers

def avg_wo_outlier(data_raw,true_val):

    data = np.sort(data_raw)
    Q1 = np.percentile(data, 25, interpolation = 'midpoint') 
    Q2 = np.percentile(data, 50, interpolation = 'midpoint') 
    Q3 = np.percentile(data, 75, interpolation = 'midpoint') 
      
    # print('Q1 25 percentile of the given data is, ', Q1)
    # print('Q1 50 percentile of the given data is, ', Q2)
    # print('Q1 75 percentile of the given data is, ', Q3)
      
    IQR = Q3 - Q1 
    # print('Interquartile range is', IQR)
    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR
    
    outlier = []
    not_outlier =[]
    for x in data:
        if ((x> up_lim) or (x<low_lim)):
             outlier.append(x)
        else:
             not_outlier.append(x)
             
    if(len(outlier)>0):
        print('\n Data raw:', data_raw)
        print(' True:', true_val)
        print(' outlier in the dataset is', outlier)
        
    return np.mean(not_outlier)             
             


# data=[1, 2, 3, 4, 5, 6, 7, 12]
# avg_wo_outlier(data)


ens_pred_list=[]
for row_id in range (0,len(all_results[0])):  
    
    row_preds=[]
    for model_id in range(0,len(all_results)):
        row_preds.append(all_results[model_id]["pred_test1"][row_id])
     
    ens_pred=avg_wo_outlier(row_preds,all_results[0]["reaction_energy"][row_id])
    ens_pred_list.append(ens_pred)
        
        

print("Ensemble remove outliers MAE:",mean_absolute_error(ens_pred_list, all_results[0]["reaction_energy"]))
print("Ensemble remove outliers RMSE:",mean_squared_error(ens_pred_list, all_results[0]["reaction_energy"],squared=False))
print("Ensemble remove outliers: R2",r2_score( all_results[0]["reaction_energy"]),ens_pred_list)



ens_pred_list2=[]
for row_id in range (0,len(all_results2[0])):  
    
    row_preds=[]
    for model_id in range(0,len(all_results2)):
        row_preds.append(all_results2[model_id]["pred_test2"][row_id])
     
    ens_pred=avg_wo_outlier(row_preds,all_results2[0]["reaction_energy"][row_id])
    ens_pred_list2.append(ens_pred)
        
        

print("Ensemble remove outliers:",mean_absolute_error(ens_pred_list2, all_results2[0]["reaction_energy"]))
print("Ensemble remove outliers RMSE:",mean_squared_error(ens_pred_list2, all_results2[0]["reaction_energy"],squared=False))
print("Ensemble remove outliers: R2",r2_score(all_results2[0]["reaction_energy"],ens_pred_list2))






## Weighted test-1 and test-2 avg

#1-3 example
pred_1_3=( np.power(2/(test1_mae[1]+test2_mae[1]),3) * all_results[1]["pred_test1"] + 
          np.power(2/(test1_mae[3]+test2_mae[3]),3) * all_results[3]["pred_test1"]) / (np.power(2/(test1_mae[1]+test2_mae[1]),3) +np.power(2/(test1_mae[3]+test2_mae[3]),3)) 
print("1-3 weight:",mean_absolute_error(pred_1_3, all_results[0]["reaction_energy"]))

pred2_1_3=( np.power(2/(test1_mae[1]+test2_mae[1]),3) * all_results2[1]["pred_test2"] + 
          np.power(2/(test1_mae[3]+test2_mae[3]),3) * all_results2[3]["pred_test2"]) / (np.power(2/(test1_mae[1]+test2_mae[1]),3) +np.power(2/(test1_mae[3]+test2_mae[3]),3)) 
print("1-3 weight:",mean_absolute_error(pred2_1_3, all_results2[0]["reaction_energy"]))

#Loop for all combinations

#test-1 (two combination)
for i in range (0,len(all_results2)): 
    for j in range(i+1,len(all_results2)):

        pred1=( np.power(2/(test1_mae[i]+test2_mae[i]),3) * all_results[i]["pred_test1"] + 
          np.power(2/(test1_mae[j]+test2_mae[j]),3) * all_results[j]["pred_test1"]) / (np.power(2/(test1_mae[i]+test2_mae[i]),3) +np.power(2/(test1_mae[j]+test2_mae[j]),3)) 
        # print(i,"-",j,":",mean_absolute_error(pred1, all_results[0]["reaction_energy"]))

        # print(mean_absolute_error(pred1, all_results[0]["reaction_energy"]))   
        # print(mean_squared_error(pred1, all_results[0]["reaction_energy"],squared=False))  
        print(r2_score( all_results[0]["reaction_energy"],pred1)) 

#test-2 (two combination)
for i in range (0,len(all_results2)):
    for j in range(i+1,len(all_results2)):

        pred2=( np.power(2/(test1_mae[i]+test2_mae[i]),3) * all_results2[i]["pred_test2"] + 
          np.power(2/(test1_mae[j]+test2_mae[j]),3) * all_results2[j]["pred_test2"]) / (np.power(2/(test1_mae[i]+test2_mae[i]),3) +np.power(2/(test1_mae[j]+test2_mae[j]),3)) 
        # print(i,"-",j,":",mean_absolute_error(pred1, all_results[0]["reaction_energy"]))

        
        # print(mean_absolute_error(pred2, all_results2[0]["reaction_energy"]))   
        # print(mean_squared_error(pred2, all_results2[0]["reaction_energy"],squared=False))  
        print(r2_score( all_results2[0]["reaction_energy"],pred2)) 
        
        
        
        
        pred_x_x=(all_results2[i]["pred_test2"]+all_results2[j]["pred_test2"]) / 2
        # print(mean_absolute_error(pred_x_x, all_results2[0]["reaction_energy"]))   
        # print(mean_squared_error(pred_x_x, all_results2[0]["reaction_energy"],squared=False))  
        print(r2_score( all_results2[0]["reaction_energy"],pred_x_x)) 




#test-1 (three combination)

for i in range (0,len(all_results2)):
    for j in range(i+1,len(all_results2)):
        for k in range(j+1,len(all_results2)):
            
            pred=( np.power(3/(test1_mae[i]+test2_mae[i]),3) * all_results[i]["pred_test1"] + 
              np.power(3/(test1_mae[j]+test2_mae[j]),3) * all_results[j]["pred_test1"] + 
              np.power(3/(test1_mae[k]+test2_mae[k]),3) * all_results[k]["pred_test1"]
              ) / (np.power(3/(test1_mae[i]+test2_mae[i]),3) +np.power(3/(test1_mae[j]+test2_mae[j]),3)+np.power(3/(test1_mae[k]+test2_mae[k]),3)) 
            # print(i,"-",j,":",mean_absolute_error(pred1, all_results[0]["reaction_energy"]))  
         
            
            # pred_x_x=(all_results2[i]["pred_test2"]+all_results2[j]["pred_test2"]+all_results2[k]["pred_test2"]) / 3
            # print(i,"-",j,"-",k)   
            # print(mean_absolute_error(pred, all_results[0]["reaction_energy"]))  
            # print(mean_squared_error(pred, all_results[0]["reaction_energy"],squared=False))  
            print(r2_score( all_results[0]["reaction_energy"],pred)) 




#test-2 (three combination)

for i in range (0,len(all_results2)):
    for j in range(i+1,len(all_results2)):
        for k in range(j+1,len(all_results2)):
            
            pred2=( np.power(3/(test1_mae[i]+test2_mae[i]),3) * all_results2[i]["pred_test2"] + 
              np.power(3/(test1_mae[j]+test2_mae[j]),3) * all_results2[j]["pred_test2"] + 
              np.power(3/(test1_mae[k]+test2_mae[k]),3) * all_results2[k]["pred_test2"]
              ) / (np.power(3/(test1_mae[i]+test2_mae[i]),3) +np.power(3/(test1_mae[j]+test2_mae[j]),3)+np.power(3/(test1_mae[k]+test2_mae[k]),3)) 
            # print(i,"-",j,":",mean_absolute_error(pred1, all_results[0]["reaction_energy"]))  
         
            
            # pred_x_x=(all_results2[i]["pred_test2"]+all_results2[j]["pred_test2"]+all_results2[k]["pred_test2"]) / 3
            # print(i,"-",j,"-",k)   
            print(mean_absolute_error(pred2, all_results2[0]["reaction_energy"]))  
            # print(mean_squared_error(pred2, all_results2[0]["reaction_energy"],squared=False))  
            # print(r2_score( all_results2[0]["reaction_energy"],pred2)) 
        




#test-1 (four combination)
for i in range (0,len(all_results2)):
    for j in range(i+1,len(all_results2)):
        for k in range(j+1,len(all_results2)):
            for l in range(k+1,len(all_results2)):
                
                pred=( np.power(2/(test1_mae[i]+test2_mae[i]),3) * all_results[i]["pred_test1"] + 
                  np.power(4/(test1_mae[j]+test2_mae[j]),3) * all_results[j]["pred_test1"] + 
                  np.power(4/(test1_mae[k]+test2_mae[k]),3) * all_results[k]["pred_test1"]+ 
                  np.power(4/(test1_mae[l]+test2_mae[k]),3) * all_results[l]["pred_test1"]
                  ) / (np.power(4/(test1_mae[i]+test2_mae[i]),3) +np.power(4/(test1_mae[j]+test2_mae[j]),3)+np.power(4/(test1_mae[k]+test2_mae[k]),3)+np.power(4/(test1_mae[l]+test2_mae[l]),3)) 
                # print(i,"-",j,":",mean_absolute_error(pred1, all_results[0]["reaction_energy"]))  
             
                
                # pred_x_x=(all_results2[i]["pred_test2"]+all_results2[j]["pred_test2"]+all_results2[k]["pred_test2"]) / 3
                # print(i,"-",j,"-",k)   
                print(mean_absolute_error(pred, all_results[0]["reaction_energy"]))  
                # print(mean_squared_error(pred, all_results[0]["reaction_energy"],squared=False))  
                # print(r2_score( all_results[0]["reaction_energy"],pred))                
                    
                    
#test-2 (four combination)
for i in range (0,len(all_results2)):
    for j in range(i+1,len(all_results2)):
        for k in range(j+1,len(all_results2)):
            for l in range(k+1,len(all_results2)):
                
                pred2=( np.power(2/(test1_mae[i]+test2_mae[i]),3) * all_results2[i]["pred_test2"] + 
                  np.power(4/(test1_mae[j]+test2_mae[j]),3) * all_results2[j]["pred_test2"] + 
                  np.power(4/(test1_mae[k]+test2_mae[k]),3) * all_results2[k]["pred_test2"]+ 
                  np.power(4/(test1_mae[l]+test2_mae[k]),3) * all_results2[l]["pred_test2"]
                  ) / (np.power(4/(test1_mae[i]+test2_mae[i]),3) +np.power(4/(test1_mae[j]+test2_mae[j]),3)+np.power(4/(test1_mae[k]+test2_mae[k]),3)+np.power(4/(test1_mae[l]+test2_mae[l]),3)) 
                # print(i,"-",j,":",mean_absolute_error(pred1, all_results[0]["reaction_energy"]))  
             
                
                # pred_x_x=(all_results2[i]["pred_test2"]+all_results2[j]["pred_test2"]+all_results2[k]["pred_test2"]) / 3
                # print(i,"-",j,"-",k)   
                # print(mean_absolute_error(pred2, all_results2[0]["reaction_energy"]))  
                # print(mean_squared_error(pred2, all_results2[0]["reaction_energy"],squared=False))  
                print(r2_score( all_results2[0]["reaction_energy"],pred2))                
                                        
                


## Weighted test-1

pred_1_3=( np.power(1/test1_mae[1],3) * all_results[1]["pred_test1"] + 
          np.power(1/test1_mae[3],3) * all_results[3]["pred_test1"]) / (np.power(1/test1_mae[1],3) +np.power(1/test1_mae[3],3)) 
print("1-3 weight:",mean_absolute_error(pred_1_3, all_results[0]["reaction_energy"]))




pred_1_2_3=( np.power(1/test1_mae[1],3) * all_results[1]["pred_test1"] + 
            np.power(1/test1_mae[2],3) * all_results[2]["pred_test1"] + 
          np.power(1/test1_mae[3],3) * all_results[3]["pred_test1"]) / (np.power(1/test1_mae[1],3) +np.power(1/test1_mae[2],3) + np.power(1/test1_mae[3],3)) 
print("1-2-3 weight:",mean_absolute_error(pred_1_2_3, all_results[0]["reaction_energy"]))


pred_0_1_2_3=( np.power(1/test1_mae[0],3) * all_results[0]["pred_test1"] + 
              np.power(1/test1_mae[1],3) * all_results[1]["pred_test1"] + 
              np.power(1/test1_mae[2],3) * all_results[2]["pred_test1"] + 
              np.power(1/test1_mae[3],3) * all_results[3]["pred_test1"]) / (np.power(1/test1_mae[0],3) +np.power(1/test1_mae[1],3) +np.power(1/test1_mae[2],3) + np.power(1/test1_mae[3],3)) 
print("0-1-2-3 weight:",mean_absolute_error(pred_0_1_2_3, all_results[0]["reaction_energy"]))


pred_0_1_2_3_4=( np.power(1/test1_mae[0],3) * all_results[0]["pred_test1"] + 
              np.power(1/test1_mae[1],3) * all_results[1]["pred_test1"] + 
              np.power(1/test1_mae[2],3) * all_results[2]["pred_test1"] + 
              np.power(1/test1_mae[3],3) * all_results[3]["pred_test1"] + 
              np.power(1/test1_mae[4],3) * all_results[4]["pred_test1"]) / (
              np.power(1/test1_mae[0],3) +np.power(1/test1_mae[1],3) +np.power(1/test1_mae[2],3) + np.power(1/test1_mae[3],3) + np.power(1/test1_mae[4],3)) 
print("0-1-2-3-4 weight:",mean_absolute_error(pred_0_1_2_3_4, all_results[0]["reaction_energy"]))



