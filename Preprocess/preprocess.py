# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 16:39:00 2021
@author: Murat Cihan Sorkun
"""

from chemplot import Plotter
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
# import cphtml
from sklearn.model_selection import train_test_split



df_reactions = pd.read_csv('../data/RedDB_reaction.csv')
print(df_reactions.isna().sum())

# Remove if reaction energy is NULL
df_reactions_clean=df_reactions[df_reactions['reaction_energy'].notna()]
df_reactions_clean=df_reactions_clean.reset_index(drop=True)
# print(df_reactions_clean.isna().sum())

# Plot histogram to see is the distribution and is there any outliers
sns.histplot(data=df_reactions_clean, x="reaction_energy")
plt.show()
plt.savefig('../results/data-reaction.png')


# create UMAP and PLOT 
cp_reddb_reaction = Plotter.from_smiles(df_reactions_clean["reactant_smiles"])
cp_reddb_reaction.umap(random_state=0)
plt.show()
plt.savefig('../results/data-umap.png')

# ADD UMAP components to dataframe
result = pd.concat([df_reactions_clean, cp_reddb_reaction.df_2_components], axis=1, join="inner")


# Arrange TEST-SET-2
data_type=[]
for umap1,umap2 in zip(result['UMAP-1'],result['UMAP-2']):
    if(umap1>20.5 and umap2>-5 and umap2<2.5 ):
        data_type.append(2)
    elif(umap1>22 and umap2>2.5 and umap2<6 ):
        data_type.append(2)
    elif(umap1>21 and umap2>6 and umap2<9 ):
        data_type.append(2)
    elif(umap1>19.5 and umap2>9 and umap2<12 ):
        data_type.append(2)
    elif(umap1>21 and umap2>12 and umap2<14 ):
        data_type.append(2)
    elif(umap1>18.5 and umap2>14 and umap2<16.5 ):
        data_type.append(2)
    elif(umap1>22 and umap2>16.5):
        data_type.append(2)
    else:
        data_type.append(0)

# ADD data-type in dataframe (0:Train 1:Test-1 2:Test-2)
result["data_type"]=data_type

# Plot only Train + Test-2
cp_reddb_reaction.target_type="C"
cp_reddb_reaction.target=data_type
cp_reddb_reaction.umap(random_state=0)
plt.show()
plt.savefig('../results/data-test2.png')

# Select TEST-SET-1
train_i_list, test_i_list = [],[]
for parent_id in range(1,53):
    index_list=result[(result['data_type']==0) & (result['data_package_id']==parent_id)].index
    if len(index_list)>8:
        train_i, test_i = train_test_split(index_list, test_size = 0.11, random_state=1)
        train_i_list = train_i_list + list(train_i)
        test_i_list = test_i_list + list(test_i)
    elif len(index_list)>0 :
        test_i = index_list.sample(n=1, random_state = 1)
        train_i = index_list.loc[~index_list.index.isin(test_i.index)]
        train_i_list.append(train_i)
        test_i_list = test_i_list + test_i

# Update TEST-SET-1 in the dataframe        
result.loc[test_i_list,["data_type"]] = 1


# Plot ALL
cp_reddb_reaction.target_type="C"
cp_reddb_reaction.target=result["data_type"]
cp_reddb_reaction.umap(random_state=0)
plt.show()
plt.savefig('../results/data-all-split.png')


# Plot Distribution
sns.countplot(x="data_type",data=result)
plt.show()
plt.savefig('../results/data-counts.png')
        

# Get UFF and MMFF values from ff_molecules.csv
df_molecules= pd.read_csv('../data/ff_molecules.csv')
r_UFF, r_MMFF, p_UFF, p_MMFF = [], [], [], []

for inchi_r, inchi_p in zip(result['reactant_inchiKey'],result['product_inchiKey']):
     
    reactant=df_molecules[df_molecules["inchiKey"] == inchi_r]
    if(len(reactant)>0):
        r_UFF.append(reactant.iloc[0]["uff_energy"])
        r_MMFF.append(reactant.iloc[0]["mmff_energy"])
    else:
        r_UFF.append(None)
        r_MMFF.append(None)    
    
    product=df_molecules[df_molecules["inchiKey"] == inchi_p]
    if(len(product)>0):
        p_UFF.append(product.iloc[0]["uff_energy"])
        p_MMFF.append(product.iloc[0]["mmff_energy"])
    else:
        p_UFF.append(None)
        p_MMFF.append(None)    

# Append UFF and MMFF values to results DF
result["reactantUFF"]=r_UFF
result["reactantMMFF"]=r_MMFF
result["productUFF"]=p_UFF
result["productMMFF"]=p_MMFF


print(result.isna().sum())


#Export CSV
result.to_csv("../results/all_data.csv",index=False)


train_data = result[result['data_type'] == 0]
test1_data = result[result['data_type'] == 1]
test2_data = result[result['data_type'] == 2]

train_data.to_csv("../results/train.csv",index=False)
test1_data.to_csv("../results/test1.csv",index=False)
test2_data.to_csv("../results/test2.csv",index=False)
