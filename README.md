# MICROPHERRET
MICROPHERRET: MICRObial PHEnotypic tRait ClassifieR using machine lEarning Techniques

MICROPHERRET is a machine learning-based for the prediction of 89 metabolic and ecological prokaryotic phenotypes from gene annotations. Starting from KEGG Orthologs, it can efficiently classify genomes and metagenomes by integrating different supervised optimized machine learning models - logistic regression, random forest, support vector machines and neural networks - maximizing the prediction accuracy. MICROPHERRET is an efficient approach for functional prediction of MAGs till 70% of genome completeness. The tool can be easily expanded and refined by the users to predict functions of interest.

## Installation
Create a conda environment with Python 3.9 and packages:
```
conda create -n micropherret python=3.9 pandas biopython scikit-learn=1.1.2 numpy tensorflow tensorflow_addons
pip install tensorflow-addons
```
To enter the environment:
```
conda activate micropherret
```
Clone this git repository
```
git clone https://github.com/BizzoTL/MICROPHERRET/
```
The files needed to run the scripts can be retrieved here:  https://mega.nz/folder/FAVSRYTT#ElNlwvpSMuXSZu9NDoIp3w

To run the scripts, create a folder called "MICROPHERRET" and add the files and script into it.

## Requirements
Random forest models were saved in .sav files with the parameter n_job set to 6 or fewer. Consequently, training MICROPHERRET requires at least 6 CPUs. To modify this parameter, users can retrain and optimize the models by following the previously described procedure.

## Usage
### Functional prediction
For simply predicting the functions of your genome of interest you need to provide the .annotations file from EggNOG-mapper. In the MICROPHERRET folder, put the annotations of your genomes or MAGs in a folder called "annotations". Run "get_annotation_matrix.ipynb" and specify the folder where the files are located. After obtaining the output, called "ko_df_all.csv", run "predict_functions.py" from the "annotation" folder. The script takes two arguments, the user-generated annotation matrix and the matrix containing functions as columns and genomes as rows (function_db.csv) to check the prediction results. 

Example:
python predict_functions.py -i ko_df_all.csv -f function_db.csv

The resulting .csv file contains the genomes used in input as rows and the prediction for each functional class of MICROPHERRET in the columns. If the prediction is positive and the genome performs a particular function, a "1" is present in the cell, otherwise a "0".

### Class generation and refinement
MICROPHERRET's models can be refined to fit the user's needs by changing the sets of genomes performing the functions stored in the training set. At the same time, the training set can be expanded by modifying the training set adding genomes linked to the function the user wishes to predict. In both cases, the users can follow the instructions in "create_new_class.py". The inputs required to run the script are .annotations files from EggNOG-mapper for all the genomes linked to the function of interest. The script will then open and parse .annotations files and from each file it retrieves the KOs. Those are stored in a dictionary with KOs as keys and copy numbers as values. The dictionary is then used with the main script KO matrix to train the different machine-learning algorithms and after selecting the best-performing model it saves it. The resulting model and scaler can then be added to the folder containing MICROPHERRET models.

Parameters to use:
'-g' Folder containing the .annotations files from EggNOG-mapper of the genomes that will be used for model training

'--from_faprotax' Dataset of genomes from FAPROTAX used for MICROPHERRET training, i.e. genome_ko_all.csv from the matrix folder

'-f' Name of the function-specific classifier that will be trained with the genomes. If it is new, a new model will be trained. Otherwise, a previous model will be retrained using the provided genomes 

'-r' .txt file with the list of genomes that we want to remove from the previous training set, one per line. If genomes present in FAPROTAX are suspected to perform the function but were not provided in -g, they should be absolutely removed


Example:
python create_new_class.py -g genomes_folder/ --from_faprotax faprotax_annotation_matrix -f new_function -r genomes_to_remove

After creating the class, add the saved model and scaler with the others in the MICROPHERRET folder to run your predictions.

#### Class refinement example: acetoclastic methanogenesis
The code to run the script is available with the other files in the "Example" folder. After installing MICROPHERRET and activating the environment, simply run create_new_class.py from the folder to obtain a refined version of the classifier to predict acetoclastic methanogenesis. The exact command to run is the following:
```
python3 create_new_class.py -g annotations/ --from_faprotax my_faprotax_db.csv  -f acetoclastic_methanogenesis 
```
The resulting files will be the .sav and .scaler models that need to be put into the model folder in order to predict MICROPHERRET functions.
