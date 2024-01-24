#to predict
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix, accuracy_score, roc_auc_score, jaccard_score, zero_one_loss, hamming_loss
from keras.models import load_model
from pickle import dump, load
import keras.backend as K

def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())

def get_validation_set(to_validate, training_set):
    training_kos = training_set.columns
    validation_kos = to_validate.columns
    if len(set(training_kos).intersection(set(validation_kos))) == 0:
        print('no common kos between provided ones and training')
        return
    else:
        common = list(set(training_kos).intersection(set(validation_kos)))
        print('{} common kos'.format(len(common)))
        common_table = to_validate[common]
        #remove orthologs in validation not in the training
        to_remove = set(validation_kos) - set(training_kos)
        print('{} kos present in user set but not in training set will be removed'.format(len(to_remove)))
        missing = list(set(training_kos) - set(validation_kos))
        print('{} kos missing from the users et will be add to train the classifiers'.format(len(missing)))
        #missing_df = pd
        missed = pd.DataFrame(0, index = to_validate.index, columns= missing)
        to_submit = common_table.merge(missed, left_index = True, right_index = True)
        #print(list(validation_kos))
        to_submit = to_submit[list(training_kos)] #change order columns
        print('Shape of training dataset: {}, Shape of user dataset: {}'.format(training_set.shape, to_submit.shape))
        if list(to_submit.columns) != list(training_set.columns): 
            print('ERRORRRRR')
            return
    return to_submit


def validate(classes, ko_validation, function_validation = 0):
    results_per_class = {}
    scores = {}
    for c in classes[::-1]:
        print(c)
        if c in ["anoxygenic_photoautotrophy_Fe_oxidizing","dark_sulfite_oxidation","oil_bioremediation","dark_sulfur_oxidation","dark_thiosulfate_oxidation","anoxygenic_photoautotrophy_S_oxidizing"]:
            model = "../saved_models/"+c+".mdl_wts.hdf5"
            scaler = load(open('../saved_models/scaler_'+c+'.sav', 'rb'))
            modelo = load_model(model, compile = True, custom_objects={"matthews_correlation_coefficient": matthews_correlation_coefficient })
            to_validate_norm = scaler.transform(ko_validation)
            pred1 = (modelo.predict(to_validate_norm) > 0.5).astype(np.int32)
            pred = []
            for i in pred1:
                pred.append(i[0])
        else:
            model = load(open('../saved_models/model_'+c+'.sav', 'rb'))
            scaler = load(open('../saved_models/scaler_'+c+'.sav', 'rb'))
            to_validate_norm = scaler.transform(ko_validation)
            pred = model.predict(to_validate_norm)

        results_per_class[c] = pred
        
        if type(function_validation) != int:
            	scores[c] = [matthews_corrcoef(function_validation[c], pred), f1_score(function_validation[c], pred, zero_division=1), confusion_matrix(function_validation[c], pred), accuracy_score(function_validation[c], pred),                        hamming_loss(function_validation[c], pred), zero_one_loss(function_validation[c], pred),  function_validation[c].sum()]
    return results_per_class, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--user_dataset', required = True, help = 'csv file containing a matrix with KOs as columns and genomes as rows. ')
    parser.add_argument('-f', '--functions', help = 'matrix containing functions as columns and genomes as rows, used for checking the results of classifier')
    args = parser.parse_args()

    classes = ['anoxygenic_photoautotrophy_Fe_oxidizing', 'oil_bioremediation', 'dark_sulfite_oxidation', 'arsenate_respiration', 
               'manganese_respiration', 'dark_sulfur_oxidation', 'knallgas_bacteria', 'reductive_acetogenesis', 'dark_iron_oxidation', 
               'dark_thiosulfate_oxidation', 'chlorate_reducers', 'iron_respiration', 'anoxygenic_photoautotrophy_H2_oxidizing', 
               'nitrate_denitrification', 'chitinolysis', 'aerobic_anoxygenic_phototrophy', 'denitrification', 'dissimilatory_arsenate_reduction', 
               'dark_sulfide_oxidation', 'ureolysis', 'cellulolysis', 'thiosulfate_respiration', 'nitrous_oxide_denitrification', 
               'plastic_degradation', 'sulfur_respiration', 'aromatic_hydrocarbon_degradation', 'acetoclastic_methanogenesis', 'xylanolysis', 
               'sulfite_respiration', 'fumarate_respiration', 'dark_hydrogen_oxidation', 'nitrification', 'methanol_oxidation', 'sulfate_respiration',
                 'dark_oxidation_of_sulfur_compounds', 'nitrite_denitrification', 'arsenate_detoxification', 'anoxygenic_photoautotrophy_S_oxidizing',
                   'nitrate_respiration', 'nitrite_respiration', 'aromatic_compound_degradation', 'nitrate_ammonification', 'ligninolysis', 
                   'nitrite_ammonification', 'phototrophy', 'respiration_of_sulfur_compounds', 'anoxygenic_photoautotrophy', 'methylotrophy', 
                   'nitrogen_fixation', 'invertebrate_parasites', 'nitrogen_respiration', 'photoheterotrophy', 'chemoheterotrophy', 'nitrate_reduction'
                   , 'aerobic_ammonia_oxidation', 'predatory_or_exoparasitic', 'methanogenesis_using_formate', 'plant_pathogen', 
                   'human_pathogens_meningitis', 'human_pathogens_gastroenteritis', 'hydrocarbon_degradation', 'manganese_oxidation', 
                   'animal_parasites_or_symbionts', 'human_pathogens_all', 'photoautotrophy', 'human_pathogens_septicemia', 'aerobic_chemoheterotrophy',
                     'human_associated', 'aliphatic_non_methane_hydrocarbon_degradation', 'human_pathogens_pneumonia', 'fermentation', 
                     'human_pathogens_diarrhea', 'mammal_gut', 'methanotrophy', 'human_gut', 'intracellular_parasites', 'methanogenesis_by_CO2_reduction_with_H2',
                     'methanogenesis_by_disproportionation_of_methyl_groups', 'methanogenesis_by_reduction_of_methyl_compounds_with_H2', 
                     'hydrogenotrophic_methanogenesis', 'oxygenic_photoautotrophy', 'aerobic_nitrite_oxidation', 'methanogenesis', 
                     'arsenite_oxidation_detoxification', 'arsenite_oxidation_energy_yielding', 'fish_parasites', 'dissimilatory_arsenite_oxidation', 
                     'photosynthetic_cyanobacteria', 'human_pathogens_nosocomia'][::-1]

    training_dataset = pd.read_csv('../matrix/genome_ko_all.csv').set_index('Genome').drop('Species', axis = 1)
    user_dataset = pd.read_csv(args.user_dataset).set_index('Unnamed: 0')

    validation_set = get_validation_set(user_dataset, training_dataset)

    if args.functions:
        functions = pd.read_csv(args.functions).drop(['Unnamed: 0', 'scientific_name'], axis =1).set_index('genome_id')
        data_all = validation_set.merge(functions, left_index=True, right_index=True)
        kos_val = data_all[validation_set.columns]
        print(kos_val)
        functions_val = data_all[functions.columns]
        print(functions_val)
        results_per_class, scores = validate(classes, kos_val, functions_val)
        results_df = pd.DataFrame(results_per_class, index = kos_val.index)
        results_df.to_csv('predict_functions_exact.csv')
        sums = results_df.sum()
        sums.to_csv('predict_sum_exact.csv')
        scores_df = pd.DataFrame(scores).T
        scores_df.columns = ['MCC', 'f1_score', 'confusion_matrix', 'accuracy', 'hamming_loss', 'zero_one_loss','genome_n']
        scores_df.to_csv('validation_scores.csv')
    else:
        results_per_class, scores = validate(classes, validation_set)
        results_df = pd.DataFrame(results_per_class, index = validation_set.index)
        indexon = pd.DataFrame(validation_set)
        results_df.to_csv('predict_functions.csv')
        sums = results_df.sum()
        sums.to_csv('predict_sum.csv')
