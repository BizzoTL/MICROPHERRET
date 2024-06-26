# MICROPHERRET
MICROPHERRET: MICRObial PHEnotypic tRait ClassifieR using machine lEarning Techniques

## Installation
Create a conda environment with Python 3.9 and packages:
```
conda create -n microrferret python=3.9 pandas biopython scikit-learn=1.1.2 numpy tensorflow tensorflow_addons
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

## Usage
### Functional prediction
For simply predicting the functions of your genome of interest you need to provide the .annotations file from EggNOG. In the MICROPHERRET folder, put the annotations of your genomes or MAGs in a folder called "annotations". Run "get_annotation_matrix.ipynb" and specify the folder where the files are located. After obtaining the output, called "ko_df_all.csv", run "predict_functions.py" from the "annotation" folder. The script takes two arguments, the user-generated annotation matrix and the matrix containing functions as columns and genomes as rows (function_db.csv) to check the prediction results. 

Example:
python predict_functions.py -i ko_df_all.csv -f function_db.csv

The resulting .csv file contains the genomes used in input as rows and the prediction for each functional class of MICROPHERRET in the columns. If the prediction is positive and the genome performs a particular function, a "1" is present in the cell, otherwise a "0".

### Class generation

To generate a new class, follow the instructions in "create_new_class.py". The inputs required to run the file are .annotations files from EggNOG for all the genomes to be included in the model. The script will then open and parse annotations files and from each .annotations file it retrieves the KOs. Those are stored in a dictionary with KOs as keys and copy numbers as values. The dictionary is then used with the main script KO matrix to train the different machine-learning algorithms and after selecting the best-performing model it saves it. The resulting model and scaler can then be added to the folder containing MICROPHERRET models.

Parameters to use:
'-g' Folder containing the genomes that will be used for model training

'--from_faprotax' Dataset of genomes and functions from FAPROTAX

'-f' Function-specific classifier that will be trained with the genomes

'-r' Genomes that we want to remove from previous training set

Example:
python create_new_class.py -g genomes_folder/ --from_faprotax faprotax_annotation_matrix -f new_function -r genomes_to_remove

After creating the class, add the saved model and scaler with the others in the MICROPHERRET folder to run your predictions.

