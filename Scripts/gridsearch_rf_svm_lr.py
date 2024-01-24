import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import random as rd
from sklearn.metrics import matthews_corrcoef, f1_score
#import shap
import argparse
import os
import time
import signal
import random
import warnings

def nested_cv(x, y, penalty):
    best_params = []
    best_scores = []
    train_scores = {'mcc':[], 'f1':[]}
    validation_scores = {'mcc':[],'f1':[]}
    cv_results = []
    
    cv_outer = StratifiedKFold(n_splits = 3, shuffle = True)
    cv_inner = StratifiedKFold(n_splits= 3, shuffle= True)

    seed = np.random.randint(0,42)
    print(seed)

    pointer = 1
    #outer loop
    for train_index, val_index in cv_outer.split(x, y):
        print('NestedCV: {} of {}'.format(pointer, cv_outer.get_n_splits()))
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        for (i,j) in zip(class_parameters['Classifiers'], class_parameters['Parameters']):

            print('Classifier: {}'.format(i))
        
            sc = StandardScaler()
            model = GridSearchCV(estimator=i, param_grid=j, scoring = 'matthews_corrcoef',n_jobs = -1, cv=cv_inner, verbose = 5)
            inner_pipeline = Pipeline(steps = [('Scale', sc), ('Tuning', model)])
            search = inner_pipeline.fit(x_train, y_train)
            %time search
            best_params.append(search.best_params_)
            best_scores.append(search.best_score_)
            cv_results.append(search.cv_results_)
            
            train_scores['mcc'].append(matthews_corrcoef(y_train, search.predict(x_train)))
            train_scores['f1'].append(f1_score(y_train, search.predict(x_train)))
            validation_scores['mcc'].append(matthews_corrcoef(y_val, search.predict(x_val)))  
            validation_scores['f1'].append(f1_score(y_val, search.predict(x_val)))

        pointer += 1      

    return best_params, best_scores, cv_results, train_scores, validation_scores

def start_nestedCV(x, y, penalty):
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size = 0.2, random_state=1)
    best_params, best_scores, cv_results, train_scores, validation_scores = nested_cv(x_train, y_train, penalty)
    return best_params, best_scores, cv_results, train_scores, validation_scores

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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--function', required = True)
    parser.add_argument('-o', '--output_folder', required = True)
    parser.add_argument('-p', '--penalty')
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    kos = pd.read_csv('./input_classifier/genome_ko_all.csv').set_index('Genome')
    functions = pd.read_csv('./function_db.csv').set_index('genome_id')
    functions.insert(0, 'ID', functions['Unnamed: 0'])
    functions.drop('Unnamed: 0', axis = 1, inplace = True)
    data = kos.merge(functions,left_index=True, right_index = True)
    ko_input = data[kos.columns].drop('Species', axis = 1)
    function_input = data[functions.columns].drop(['scientific_name', 'ID'], axis = 1)
    y_col = [i for i in function_input.columns if function_input[i].sum() > 3]
    y_input = function_input[y_col]
    discarded_class = [i for i in function_input.columns if i not in y_input.columns]

    print(args.function)
    optimal = {}
    classifier_res = {}
    best_params, best_scores, cv_results, train_scores_nCV, val_scores = start_nestedCV(ko_input, y_input[args.function], args.penalty)
    best_best_params, best_best_score, best_best_val_score = get_best_best_params(best_params, best_scores, val_scores)
    if os.path.exists(args.output_folder) == False:
        os.makedirs(args.output_folder)

    pd.Series(best_best_params).to_csv(os.path.join(args.output_folder,'best_params_NEW_' + args.penalty + '.csv'))
    pd.Series(best_best_scores).to_csv(os.path.join(args.output_folder,'best_scores_NEW_' + args.penalty + '.csv'))

    pd.Series(train_scores_nCV).to_csv(os.path.join(args.output_folder,'train_scores_cv_NEW_' + args.penalty + '.csv'))
    pd.Series(val_scores).to_csv(os.path.join(args.output_folder,'val_scores_NEW_' + args.penalty + '.csv'))