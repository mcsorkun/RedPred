# RedPred 
RedPred: Redox Energy Prediction Tool for Redox Flow Battery Molecules

------------------------------------------

## About RedPred Project:

* RedPred is an redox energy prediction model for redox flow battery molecules that consists ensemble of 3 ML algorithms (Graph Conv Neural Nets, Random Forest, and Deep Neural Nets).
 
* The model takes the SMILES notations of reactant molecules of the redox reaction as an input and predicts the redox reaction energy (Hartree).

* RedPred is trained on RedDB [1] publicly available redox flow battery candidate molecules dataset.

* The performance of the RedPred is 0.0036 and 0.0043 Hartree MAE on the test-1 and test-2 sets, respectively.

* If you are using the predictions from RedPred on your work, please cite these papers: [1, 2] 

  * [1] Sorkun, Elif, et al. (2021). RedDB, a computational database of electroactive molecules for aqueous redox flow batteries.

  * [2] In preparation (will be updated soon)


------------------------------------------
## Workflow - IMAGE :

![](redpred_app.gif)

------------------------------------------

## Project Files:


**Data:** Contains the row and processed data files 

**Preprocess:** Contains data preprocessing including removing missing values and test/train splitting (requires different dependencies, to reproduce it please check the requirements on the folder)

**Ensemble:** Contains ensembling process of the selected 3 models. 

**Models:** Contains final code file of 5 models that we used for RedPred project and ECFC encoder file.

------------------------------------------


## Dependencies:

- python=3.7.9 (requires "conda install python=3.7.9")
- rdkit=2020.09.1.0 (requires "conda install -c conda-forge rdkit=2020.09.1")
- scikit-learn=0.22.1
- deepchem==2.4.0
- numpy==1.18.5
- pandas==1.1.3
- tensorflow==2.3.2
- keras==2.4.3
- lightgbm==2.3.1
- xgboost==1.4.2
- h5py==2.10.0

------------------------------------------


## Web application:


You can use the RedPred web application by following [this link](https://share.streamlit.io/mcsorkun/redpred-web/main/app.py).

------------------------------------------

## Report an Issue:
             
You are welcome to report a bug or contribuite to the RedPred project by filing an [issue](https://github.com/mcsorkun/RedPred/issues).


------------------------------------------

## References:


[1]: 



------------------------------------------

## Developers:

* **Murat Cihan Sorkun :** [![](https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/murat-cihan-sorkun/) 

* **Cihan Yatbaz :** [![](https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/cihanyatbaz/) 

* **Elham Nour Ghassemi :** [![](https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/elhamnourghassemi/)
      


------------------------------------------


This project developed at AMD LAB : https://www.amdlab.nl/
