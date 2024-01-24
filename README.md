# MICROPHERRET
MICROPHERRET: MICRObial PHEnotypic tRait ClassifieR using machine lEarning Techniques

''Installation''
Create a conda environment with Python 3.9 and packages:

conda create -n microrferret python=3.9 pandas biopython scikit-learn=1.1.2 numpy tensorflow tensorflow_addons pip install tensorflow-addons

To enter the environment:

conda activate micropherret

Clone this git repository

git clone https://github.com/BizzoTL/MICROPHERRET/

''Usage''

For simply predicting the functions of your genome of interest you need to have the .annotations file from EggNOG. Run "get_annotation_matrix.ipynb" and specify the folder where the files are locates. After obtaining the output, run "predict_functions.py".

To generate a new class, follow the instructions in "create_new_class.py". The resulting model and scaler can then be added to the folder containing MICROPHERRET models.
