#script to get a new class
#load modules
import pandas as pd
import os 
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV,  StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
import random as rd
from sklearn.metrics import matthews_corrcoef, f1_score
from pickle import dump
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2, l1
from keras.metrics import CategoricalAccuracy, AUC, BinaryCrossentropy, MeanSquaredError, TruePositives, FalsePositives, TrueNegatives,FalseNegatives, BinaryAccuracy,Precision,Recall,AUC
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import keras_tuner
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

#metrics for NN
METRICS = [BinaryCrossentropy(name='Binary crossentropy'),
      TruePositives(name='tp'),
      FalsePositives(name='fp'),
      TrueNegatives(name='tn'),
      FalseNegatives(name='fn'),
      Precision(name='precision'),
      Recall(name='recall'),
      AUC(name='auc')]

#open and parse annotations file
def get_file(path):
    comments = []
    data_list = []
    annotated_file = {}

    with open(path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            if line[0].startswith('#'): comments.append(line[0])
            #store lines 
            data_list.append(line)

    #check length file using comments
    if len(comments) != len(set(comments)):
        indexes = []
        for i in range(len(data_list)):
            if comments[1] in data_list[i]:
                indexes.append(i)
        index_to_divide = indexes[-1] -1
    else:
        index_to_divide = 0

    for line in data_list[index_to_divide:]:
        if line[0].startswith('#'):continue
        annotated_file[line[0]] = line[1:]
    
    return annotated_file


#from each annotations file retrieves the KOs
def get_kos(path):
    ind = path.index('G')
    genome = path[ind:(ind+15)]
    #open and parse the file
    annotated_file = get_file(path)  
    #list of detected kos 
    ko_list = []
    
    #for each line in the file, KO are at the 10th position 
    for query in annotated_file.keys():
        ko = annotated_file[query][10]
        #if not empty and not multiple kos are present
        if ko != '-' and ',' not in ko:
            ko_list.append(ko)
        #if multiple kos are present
        elif ',' in ko:
            ko_list += ko.split(',')
    
    ko_list = [k[3:] for k in ko_list]

    #store data into dictionary with ko as keys and copy numbers as values
    genome_data_ko = {}
    for ko in ko_list:
        #for each detected ko count how many times is present
        genome_data_ko[ko] = ko_list.count(ko)

    return genome, genome_data_ko, len(set(ko_list))

def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Choice('units', [16, 32, 64, 128, 256]), activation='relu',kernel_regularizer=l2(l2=0.01)))
    model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, default=0.2, step=0.05)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid',kernel_regularizer=l2(l2=0.01)))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=METRICS)
    
    return model

def nn_model(dataset, function):
    results = {}
    for i in range(5):
        print("Model for ", function)
        earlyStopping = EarlyStopping(monitor='precision', patience=3, verbose=0, mode='max')
        namemodel = str(function)+"_"+str(i)+"_tuner_"+'.mdl_wts.hdf5'
        mcp_save = ModelCheckpoint(namemodel, save_best_only=True, monitor='precision', mode='max')

        training_dataset = dataset.sample(frac = 0.8)
        test_dataset = dataset.drop(training_dataset.index)
        train_labels = training_dataset.iloc[:, [-1]]
        test_labels = test_dataset.iloc[:, [-1]]
        sc_out = StandardScaler()
        training_dataset_1 = training_dataset.drop(columns=[training_dataset.columns[0],training_dataset.columns[1], training_dataset.columns[-1]])
        training_dataset_x = sc_out.fit_transform(training_dataset_1.values.reshape(training_dataset_1.shape[0],training_dataset_1.shape[1]))
        test_dataset = test_dataset.drop(columns=[test_dataset.columns[0],test_dataset.columns[1], test_dataset.columns[-1]])
        test_dataset_x = sc_out.fit_transform(test_dataset.values.reshape(test_dataset.shape[0],test_dataset.shape[1]))
        class_weights = compute_class_weight('balanced', classes=[0,1], y=dataset[function].values.reshape(-1))
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        tuner = keras_tuner.RandomSearch(build_model, overwrite=True, objective=keras_tuner.Objective("precision", direction="max"), max_trials=10)
        tuner.search(training_dataset_x, train_labels, epochs=10)

        best_model = tuner.get_best_models()[0]
        test_eval = best_model.evaluate(test_dataset_x,test_labels)
        true_positive, false_positive, true_negative, false_negative = test_eval[3], test_eval[4], test_eval[5], test_eval[6]
        denominator = ((true_positive + false_positive) * (true_positive + false_negative) * 
                       (true_negative + false_positive) * (true_negative + false_negative))
        if denominator == 0:
            mcc = 0
        else:
            numerator = (true_positive * true_negative) - (false_positive * false_negative)
            mcc = numerator / (denominator ** 0.5)
            
        best_hp = tuner.get_best_hyperparameters()[0]
        results[i] = [mcc, best_hp.values]
    max_mcc = -1
    for j in range(5):
    	if results[j][0] > max_mcc:
       		max_mcc = results[j][0]
       		best_model_all = results[j][1]
    return max_mcc, best_model_all

def nested_cv(x, y, clf, params):
    best_params = []
    best_scores = []
    train_scores = {'mcc':[], 'f1':[]}
    validation_scores = {'mcc':[],'f1':[]}
    
    cv_outer = StratifiedKFold(n_splits = 3, shuffle = True)
    cv_inner = StratifiedKFold(n_splits= 3, shuffle= True) 
    
    pointer = 1

    #outer loop
    for train_index, val_index in cv_outer.split(x, y):
        print('NestedCV: {} of {}'.format(pointer, cv_outer.get_n_splits()))
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        sc_inner = StandardScaler()
        inner_pipeline = Pipeline(steps=[('sc', sc_inner), ('model', clf)])

        #GridSearchCv -- inner loop
        model = GridSearchCV(estimator=inner_pipeline, param_grid=params, scoring='matthews_corrcoef', cv = cv_inner,verbose=3, n_jobs = 3).fit(x_train, y_train)
        best_params.append(model.best_params_)
        best_scores.append(model.best_score_)

        train_scores['mcc'].append(matthews_corrcoef(y_train, model.predict(x_train)))
        train_scores['f1'].append(f1_score(y_train, model.predict(x_train)))
        validation_scores['mcc'].append(matthews_corrcoef(y_val, model.predict(x_val)))  
        validation_scores['f1'].append(f1_score(y_val, model.predict(x_val)))

        pointer += 1      

    return best_params, best_scores, train_scores, validation_scores
    
def start_nestedCV(x, y, clfs, params):
    best_params = {}
    best_scores = {}
    train_scores = {}
    validation_scores = {}

    for model in clfs.keys():
        print('Nested CV...')
        print(model)
        #entire x and y, not separated in train and test set
        best_params[model], best_scores[model], train_scores[model], validation_scores[model] = nested_cv(x, y, clfs[model], params[model])
    
    return best_params, best_scores, train_scores, validation_scores, clfs

def get_best_best_params(best_pars, best_scores, val_scores):
    best_par_values = [tuple(i[key] for key in i.keys()) for i in best_pars]

    if len(set(best_par_values)) == 1:
        print('One best parameter')
        best_best_score = sum(best_scores)/len(best_scores)
        best_val_scores = [sum(val_scores[scoring])/len(val_scores[scoring]) for scoring in val_scores.keys()]
        return best_pars[0], best_best_score, best_val_scores
    
    elif len(set(best_par_values)) == len(best_par_values):
        print('no best parameter...')
        max_mcc_ind = val_scores['mcc'].index(max(val_scores['mcc']))
        return best_pars[max_mcc_ind], best_scores[max_mcc_ind], [val_scores[scoring][max_mcc_ind] for scoring in val_scores.keys()]
    
    else:
        print('counting')
        counts = {best_par_values.count(par):par for par in set(best_par_values)}
        max_count = max(list(counts.keys()))
        best_best_par = best_pars[best_par_values.index(counts[max_count])]
        indexes = [i[0] for i in list(enumerate(best_pars)) if i[1] == best_best_par]
        avg_best_score = sum([best_scores[i] for i in indexes]) / len([best_scores[i] for i in indexes])
        avg_best_val_sores = [sum([val_scores[scoring][i] for i in indexes]) / len([val_scores[scoring][i] for i in indexes]) for scoring in val_scores.keys()]
        return best_best_par, avg_best_score, avg_best_val_sores 

def get_best_model(clfs, best_pars, best_scores, val_scores):
    model_pars = {}
    model_scores = {}
    model_vals = {}
    for model in best_pars.keys():
        print(model)
        print(best_pars[model])
        #find best parameters of each model
        model_pars[model], model_scores[model], model_vals[model] = get_best_best_params(best_pars[model], best_scores[model], val_scores[model])
    print(model_pars)
    print(model_scores)
    print(model_vals)
    #consider MCC only
    vals = [model_vals[model][0] for model in model_vals.keys()]
    print(vals)
    #look for max MCC among the clfs
    max_values = max(vals)
    print(max_values)
    ind = vals.index(max_values)
    chosen_model = list(model_vals.keys())[ind]
    print(chosen_model)
    chosen_pars = model_pars[chosen_model]
    print(chosen_pars)
    par_to_pass = {key[7:]:chosen_pars[key] for key in chosen_pars}
    print(par_to_pass)
    #add best parameters to chosen classifier
    chosen_classifier = clfs[chosen_model].set_params(**par_to_pass)
    print(chosen_classifier)
    chosen_score = model_scores[chosen_model]
    print(chosen_score)
    chosen_val = model_vals[chosen_model]
    print(chosen_val)
    return chosen_classifier, chosen_score, chosen_val, model_pars, model_scores, model_vals

def classifier_NN(function, dataset, units, dropout):
	earlyStopping = EarlyStopping(monitor='precision', patience=3, verbose=0, mode='max')
	namemodel = "new_"+ str(function)+'.mdl_wts.hdf5'
	mcp_save = ModelCheckpoint(namemodel, save_best_only=True, monitor='precision', mode='max')
	training_dataset = dataset
	train_labels = training_dataset.iloc[:, [-1]]
	sc_out = StandardScaler()
	training_dataset_1 = training_dataset.drop(columns=[training_dataset.columns[0],training_dataset.columns[1], training_dataset.columns[-1]])
	training_dataset_x = sc_out.fit_transform(training_dataset_1.values.reshape(training_dataset_1.shape[0],training_dataset_1.shape[1]))
	class_weights = compute_class_weight('balanced', classes=[0,1], y=dataset[function].values.reshape(-1))
	class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
	model = Sequential()
	model.add(Dense(units, activation='relu', input_shape=(training_dataset_x.shape[1], 1),kernel_regularizer=l2(l2=0.01)))
	model.add(Dropout(dropout))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid',kernel_regularizer=l2(l2=0.01)))
	model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=METRICS)
	model.fit(training_dataset_x, train_labels, epochs=10, batch_size=32,callbacks=[earlyStopping, mcp_save,],class_weight =class_weight_dict)
	dump(scaler_fitted, open('scaler_'+func_class+str(i)+'.sav', 'wb'))

def classifier(function, x,y, chosen_model):
    sc = StandardScaler()
    x_train_normalized = sc.fit_transform(x)
    print(x_train_normalized.shape)
    print(chosen_model)
    best_model_fitted = chosen_model.fit(x_train_normalized, y)
    # save the model
    dump(best_model_fitted, open('new_model_'+function+'_2.sav', 'wb'))
    # save the scaler
    dump(sc, open('new_scaler_'+function+'_2.sav', 'wb'))
    print(function + ': classifier saved!')
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--genomes_folder', required=True, help = 'Folder containing the genomes that will be used for model training')
    parser.add_argument('--from_faprotax', required= True, help = 'Dataset of genomes and functions from FAPROTAX')
    parser.add_argument('-f', '--function', help='Function-specific classifier that will be train with the genomes')
    parser.add_argument('-r', '--to_remove', help = 'Genomes that we want to remove from previous training set. If not given only the ones in genomes_folder will be removed.')
    args = parser.parse_args()

    #file con annotazioni in una cartella
    files = [args.genomes_folder + i for i in os.listdir(args.genomes_folder) if i.endswith('.annotations')]
    print('{} annotated genomes available in given folder'.format(len(files)))

    data_ko = {}
    ko_number = {}
    print('Retrieving KOs from given files...')
    for f in files:
        genome, data_ko[genome], ko_number[genome] = get_kos(f)

    ko_df = pd.DataFrame(data = data_ko).T
    ko_df_new = ko_df.fillna(0)
    print('{} KOs were detected'.format(ko_df_new.shape[1]))
    print('Genetic information saved in features matrix called ko_df.csv')
    ko_df_new.to_csv('ko_df_2.csv')
    #joined everything and create training dataset
    function_df = pd.Series(data = 1, index = ko_df_new.index, name = args.function) #create function column for ko_df
    #print('Label matrix function_df.csv created')
    #function_df.to_csv('function_df_2.csv')
    #merge to ko_df
    data_all = ko_df.merge(function_df, left_index=True, right_index=True)
    data_all = data_all.fillna(0)
    data_all.to_csv('dataset_2.csv')

    #change data from faprotax
    #drop genomes if they are present in faprotax
    print('downloading')
    faprotax = pd.read_csv(args.from_faprotax).set_index('Unnamed: 0')
    if args.function not in faprotax.columns:
        func_to_add = pd.Series(data = 0, index = faprotax.index, name = args.function)
        faprotax[args.function] = func_to_add

    #my faprotax should have all the faprotax genomes (14364) as rows, their kos and the desired function as columns 
    my_faprotax = faprotax[[col for col in faprotax.columns if col.startswith('K') or col == args.function]]
    #my_faprotax.to_csv('my_faprotax_2.csv')

    #drop genomes who were assigned to the given function in faprotax
    genomes_to_drop = my_faprotax.loc[my_faprotax[args.function] == 1].index
    my_faprotax_2 = my_faprotax.drop(genomes_to_drop)

    print(my_faprotax.shape)
    print(my_faprotax_2.shape)
    #drop genomes that will be added now if present -- means that they were negatevely assigned to  function in faprotax 
    for g in ko_df_new.index:
        print(g)
        try: my_faprotax_2.drop(g, inplace = True)
        except: continue
    print(my_faprotax_2.shape)

    #remove genomes that will be used in validation from dataset
    if args.to_remove:
        to_remove_list = []
        with open(args.to_remove) as file:
            for line in file:
                line = line.strip().split('\t')
                to_remove_list.append(line[0])
        for g in to_remove_list:
            try: my_faprotax_2.drop(g, inplace= True)
            except: continue
    print(my_faprotax_2.shape)

    #join to faprotax to obtain dataset with organisms both doing and not doing the function
    everything = pd.concat([my_faprotax_2, data_all])
    everything_new = everything.fillna(0)
    everything_new.to_csv('everything_2.csv')
    print('Entired dataset saved in everything.csv')
    print(everything_new.shape)
    
    
    #separate label and feature matrixes 
    everything_x = everything_new.drop(args.function, axis = 1)
    everything_y = everything_new[args.function]

    clfs = {'lr': LogisticRegression(solver = 'liblinear', max_iter= 10000, random_state= 0), 
    'rf': RandomForestClassifier(random_state=0, bootstrap=True, n_jobs = 5), 'svm': SVC(cache_size=1000, random_state=0, kernel='linear')}

    params_grid = {'lr': {'model__C': [0.001, 0.01, 1, 10, 100], 'model__penalty': ['l1', 'l2']}, 
    'rf': {'model__n_estimators': [100, 200, 500, 800, 1000], 'model__max_features' : ['sqrt', None, 'log2']}, 
    'svm' : {'model__C': [1, 0.1, 0.001, 10], 'model__kernel' : ['linear', 'rbf', 'sigmoid', 'poly']}}    

    #do nested cv for the given clfs on the entire created dataset
    params, scores, train, vals, clfs = start_nestedCV(everything_x, everything_y, clfs, params_grid)
    print('Nested CV data stored in data_nested_cv.csv')
    data_nested_cv = pd.DataFrame([params, scores, train, vals])
    print('Detected parameters per model: \n {}'.format(params))
    print('Corrisponding scores calculated in NestedCV: \n {}'.format(scores))
    print('MCC and F1 scores calculated on train set: \n {}'.format(train))
    print('MCC and F1 scores calcukated on validation set: \n {}'.format(vals))
    data_nested_cv.to_csv('data_nested_cv_2.csv')
    
    best_nn_mcc, best_nn_model = nn_model(everything_new, args.function)
    
    chosen_model, chosen_score, chosen_val, model_pars, model_scores, model_vals = get_best_model(clfs, params, scores, vals)
    best_nested_cv = pd.DataFrame([model_pars, model_scores, model_vals])
    best_nested_cv.to_csv('best_nested_cv_2.csv')
    
    if best_nn_mcc > chosen_val[0]:
    	print('Chosen model: Neural Network')
    	print('Score on validation set: ', best_nn_mcc)

    	print('Storing information in choosen_model_2.txt')
    	with open('chosen_model_2.txt', 'w') as file:
        	file.write('classifier: Neural Network'+'\n')
        	file.write('score: ' + str(best_nn_mcc) + '\n')
        	file.close()

    	print('Saving chosen classifier...')
    	classifier_NN(args.function, best_nn_model["units"], best_nn_model["dropout"])
    else:
    	print('Chosen model: ', chosen_model)
    	print('Nested CV score: ', chosen_score)
    	print('Score on validation set: ', chosen_val)

    	print('Storing information in choosen_model_2.txt')
    	with open('chosen_model_2.txt', 'w') as file:
        	file.write('classifier: ' + str(chosen_model) + '\n')
        	file.write('score: ' + str(chosen_score) + '\n')
        	file.write('validation score: ' + str(chosen_val)+'\n')
        	file.close()

    	print('Saving chosen classifier...')
    	classifier(args.function, everything_x, everything_y, chosen_model)
