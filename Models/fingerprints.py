# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 21:43:01 2021
@author: Murat Cihan Sorkun

Fingerprint funtions for encoding molecules
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
#from mhfp.encoder import MHFPEncoder ---- FOR SECFP ENCODER



def get_ecfp(smiles_list, radius=2, nBits=2048, useCounts=False):
    """
    Calculates the ECFP fingerprint for given SMILES list
    
    :param smiles_list: List of SMILES
    :type smiles_list: list
    :param radius: The ECPF fingerprints radius.
    :type radius: int
    :param nBits: The number of bits of the fingerprint vector.
    :type nBits: int
    :param useCounts: Use count vector or bit vector.
    :type useCounts: bool
    :returns: The calculated ECPF fingerprints for the given SMILES
    :rtype: Dataframe
    """     
    
    ecfp_fingerprints=[]
    erroneous_smiles=[]
    for smiles in smiles_list:
        mol=Chem.MolFromSmiles(smiles)
        if mol is None:
            ecfp_fingerprints.append([None]*nBits)
            erroneous_smiles.append(smiles)
        else:
            mol=Chem.AddHs(mol)
            if useCounts:
                ecfp_fingerprints.append(list(AllChem.GetHashedMorganFingerprint(mol, radius, nBits)))  
            else:    
                ecfp_fingerprints.append(list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits).ToBitString()))  
    
    # Create dataframe of fingerprints
    df_ecfp_fingerprints = pd.DataFrame(data = ecfp_fingerprints, index = smiles_list)
    # Remove erroneous data
    if len(erroneous_smiles)>0:
        print("The following erroneous SMILES have been found in the data:\n{}.\nThe erroneous SMILES will be removed from the data.".format('\n'.join(map(str, erroneous_smiles))))           
        df_ecfp_fingerprints = df_ecfp_fingerprints.dropna(how='any')    
    
    return df_ecfp_fingerprints



# ECFC Encoder

def get_ecfc(smiles_list, radius=2, nBits=2048, useCounts=True):
    """
    Calculates the ECFP fingerprint for given SMILES list
    
    :param smiles_list: List of SMILES
    :type smiles_list: list
    :param radius: The ECPF fingerprints radius.
    :type radius: int
    :param nBits: The number of bits of the fingerprint vector.
    :type nBits: int
    :param useCounts: Use count vector or bit vector.
    :type useCounts: bool
    :returns: The calculated ECPF fingerprints for the given SMILES
    :rtype: Dataframe
    """     
    
    ecfp_fingerprints=[]
    erroneous_smiles=[]
    for smiles in smiles_list:
        mol=Chem.MolFromSmiles(smiles)
        if mol is None:
            ecfp_fingerprints.append([None]*nBits)
            erroneous_smiles.append(smiles)
        else:
            mol=Chem.AddHs(mol)
            if useCounts:
                ecfp_fingerprints.append(list(AllChem.GetHashedMorganFingerprint(mol, radius, nBits)))  
            else:    
                ecfp_fingerprints.append(list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits).ToBitString()))  
    
    # Create dataframe of fingerprints
    df_ecfp_fingerprints = pd.DataFrame(data = ecfp_fingerprints, index = smiles_list)
    # Remove erroneous data
    if len(erroneous_smiles)>0:
        print("The following erroneous SMILES have been found in the data:\n{}.\nThe erroneous SMILES will be removed from the data.".format('\n'.join(map(str, erroneous_smiles))))           
        df_ecfp_fingerprints = df_ecfp_fingerprints.dropna(how='any')    
    
    return df_ecfp_fingerprints




#simple test
test_smiles_list=["CCCC","CO","ON(c1ccccc1)C(=O)c2ccc(Cl)cc2Cl","CN1C=NC2=C1C(=O)N(C(=O)N2C)C","XXXX"]
simple_ecfp_result=get_ecfp(test_smiles_list)
simple_ecfp_count_result=get_ecfp(smiles_list=test_smiles_list,useCounts=True)
#simple_maccs_result=get_maccs(test_smiles_list)
#simple_secfp_result=get_secfp(test_smiles_list)

#reddb SMILES test
#reddb_smiles = pd.read_csv("reddb-smiles.csv") 
# reddb_ecfp_result=get_ecfp(reddb_smiles["smiles"])
# reddb_maccs_result=get_maccs(reddb_smiles["smiles"])
# reddb_ecfp_count_result=get_ecfp(smiles_list=reddb_smiles["smiles"],useCounts=True)
# reddb_secfp_result=get_secfp(smiles_list=reddb_smiles["smiles"])
