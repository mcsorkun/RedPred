# RedPred 
RedPred: Redox Energy Prediction Tool for Redox Flow Battery Molecules

------------------------------------------

## About RedPred Project:

* RedPred is an reaction energy prediction model for redox flow battery molecules that consists consensus of 3 ML algorithms (Graph Conv Neural Nets, Random Forest, and Deep Neural Nets).

* You can upload or type your SMILES used as a reactant in the redox reaction to get the reaction energy (Hartree).

* RedPred is trained on RedDB [1] publicly available redox flow battery candidate molecules dataset.

* The performance of the RedPred is 0.0036 Hartree MAE on the test set.

* If you are using the predictions from RedPred on your work, please cite these papers: [1, 2] 

  * [1] Sorkun, Elif, et al. (2021). RedDB, a computational database of electroactive molecules for aqueous redox flow batteries.

  * [2] In preparation (will be updated soon)


------------------------------------------
## IMAGE

![](redpred_app.gif)

------------------------------------------

## Project Files


**force_field:** Force field calculations using Rdkit UFF and MMFF methods.

**merge_data:** Code for combining molecular and reaction datafiles

**outlier:** Outlier analysis

**split_data:** Contains a custom split method ensures each 



Note: Please put explanation for your new folders


**Saved-model:** Contains a saved trained GCN model for initial prediction with first guess of hyperparameters 


------------------------------------------

## Web application


You can use the RedPred web application by following [this link](https://share.streamlit.io/mcsorkun/redpred-web/main/app.py).

------------------------------------------

## Report an Issue 
             
You are welcome to report a bug or contribuite to the web application by filing an [issue](https://github.com/mcsorkun/RedPred/issues).



------------------------------------------

## Developers

* **Murat Cihan Sorkun :** [![](https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/murat-cihan-sorkun/) 

* **Cihan Yatbaz :** [![](https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/cihanyatbaz/) 

* **Elham Nour Ghassemi :** [![](https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/elhamnourghassemi/)
      


------------------------------------------


