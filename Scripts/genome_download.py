#Genomes rank analysis and get genomes RefSeq accession IDs 

#Load modules
import pandas as pd
import argparse
import os
import random

#Download NCBI taxonomy IDs and ranks 
def download_taxonomy():
    names_NCBI = {} #names as keys and NCBI IDs as values
    nodes_NCBI = {} #NCBI IDs as keys and ranks as values
    ranks = set()

    with open('./names.dmp', 'r') as file: #new updated ones
        for line in file:
            line = line.strip().split('|')
            line = [i.rstrip().lstrip() for i in line]
            name = line[1]
            id = line[0]
            info = line[3]

            if info == 'scientific name':
                try:
                    names_NCBI[name].append(id)
                except:
                    names_NCBI[name] = [id]

    with open('./nodes.dmp', 'r') as file:
        for line in file:
            line = line.strip().split('|')
            line = [i.rstrip().lstrip() for i in line[0:3]] 
            nodes_NCBI[line[0]] =  line[1], line[2]
            ranks.add(line[2])
            
    return names_NCBI, nodes_NCBI, ranks


#Get genome Refseq accession IDs per taxon name
def get_genomes_by_name(genomes):
    gen_by_name = {}

    for genome in genomes:
        list_acc_ids = []
        for i in genomes[genome]:
            acc_id = i[0]
            list_acc_ids.append(acc_id)
        gen_by_name[genome] = list_acc_ids

    return gen_by_name

#Get genome Refseq accession IDs per taxon ID using NCBI taxonomy
def get_gen_ids(genomes):
    #Call get_genomes_by_name and substitute name with NCBI ID
    gen_by_name = get_genomes_by_name(genomes)
    gen_by_id = {}

    for name in gen_by_name:
        genomes = gen_by_name[name]
        for id in taxonomy[name]:
            gen_by_id[id] = genomes

    return gen_by_id

#Parse file obtained from dry run of ncbi-genome-download
def get_genomes(path):
    genomes = {}
    count = 0

    with open(path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            if not line[0].startswith('Considering'):
                count += 1
                name = line[1]
                try: 
                    genomes[name].append([line[0], line[2]])

                except KeyError:
                    genomes[name] = [[line[0], line[2]]]

    print('{} genomes present in {}'.format(count, path))

    #Return list of Refseq accession ID per taxa ID
    #dictionary with taxaID as keys and genomes as values
    gen_by_id = get_gen_ids(genomes)

    return gen_by_id


#Retrieve IDs with right rank among the present genomes IDs or their parent IDs
def right_rank(id, rank):
    rank_index = my_lineage.index(rank)
    ancestors = my_lineage[:rank_index]
    
    if id == '447024':
        res = 'species'
    else:
        res = nodes_NCBI[id][1] #rank
    if res in ancestors: 
        return
    elif res == rank:
        return id
    else:
        if id == '447024': 
            parent = '2614943'
        else:
            parent = nodes_NCBI[id][0] #parent id

        return right_rank(parent,rank)
    
#For each ID at right rank, store list of genomes belonging to that taxa 
def ranking_analysis(rank, test):
    data = {}

    for id in test:
        result = right_rank(id, rank)
        if result is not None:
            for c in range(len(test[id])):
                try:
                    data[result].append(id)
                except:
                    data[result] = [id]
        else: continue

    return data

#Perform distribution analysis of genomes at given rank
def genome_analysis(rank, genomes_id):

     print('Analyse {} distribution...'.format(rank))
     result = ranking_analysis(rank, genomes_id)

     print('Create outputs...')
     df = {}
     df['ID'] = [id for id in result]
     names = {i:name for name in taxonomy for i in taxonomy[name]}
     df['Names'] = [names[id] for id in result]
     df['Number'] = [len(result[id]) for id in result] #set!!!!

     return pd.DataFrame(data = df).sort_values(by = 'Number', ascending=False), result



def retrieve_descen_ids(descendants):
    genomes_complete = []
    genomes = []
    for id in descendants:
        g = genomes_id[id]
        genomes += g

        if id in complete_genomes_id:
            genomes_complete += complete_genomes_id[id]
    
    return genomes, genomes_complete

#Retrieve max n accession IDs per taxa 
def get_acc_ids(species_df, species_dict, n = 30):
    species = species_df['ID']
    info_genomes = {}
    genomes_of_species = {}
    acc_ids = []

    under = list(species_df.loc[species_df['Number'] <= n]['ID']) #7868
    up = list(species_df.loc[species_df['Number'] > n]['ID']) #84

    for s in species:
        des_set = set(species_dict[s]) #descendants

        s_acc_ids, complete_acc_ids = retrieve_descen_ids(des_set)

        #if taxa genomes < n then all genomes are taken
        if s in under:
            genomes_of_species[s] = s_acc_ids
            info_genomes[s] = [str(len(genomes_of_species[s])) , str(len(complete_acc_ids))]
            acc_ids += s_acc_ids
        
        #if taxa has too many genomes, we check for number of complete genomes
        #if complete genomes are too many, we pick randomly 
        elif s in up and len(complete_acc_ids) > n:
            taken_acc_ids = random.sample(complete_acc_ids, n)
            genomes_of_species[s] = taken_acc_ids
            info_genomes[s] = [str(len(taken_acc_ids)), str(len(complete_acc_ids))]
            acc_ids += taken_acc_ids
        
        #if no complete genomes are available, we pick randomly among the not completed genomes
        elif s in up and len(complete_acc_ids) == 0:
            taken_acc_ids = random.sample(s_acc_ids, n)
            genomes_of_species[s] = taken_acc_ids
            info_genomes[s] = [str(len(taken_acc_ids)), str(len(complete_acc_ids))]
            acc_ids += taken_acc_ids

        else: #pick available complete genomes
            genomes_of_species[s] = complete_acc_ids
            info_genomes[s] = [str(len(complete_acc_ids)), str(len(complete_acc_ids))]
            acc_ids += complete_acc_ids

    print('{} RefSeq IDs stored...'.format(len(acc_ids)))

    return acc_ids, genomes_of_species, info_genomes



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Genomes distribution analysis')
    parser.add_argument('-i', '--genome_list', required=True, help="Filepath to list of genomes obtained from dry run (-n command) of ncbi-genome-download")
    parser.add_argument('-c', '--complete_list', required=True, help="Filepath to list of complete genomes obtained from dry run (-n command) of ncbi-genome-download")
    parser.add_argument('-r', '--rank', default= 'species', choices =['superkingdom', 'kingdom', 'subkingdom', 'superphylum', 'phylum', 'subphylum', 'infraphylum', 'superclass', 'class', 'subclass', 'infraclass', 'cohort', 'subcohort', 'superorder', 'order', 'suborder', 'infraorder', 'parvorder', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'subgenus', 'section', 'subsection', 'series', 'subseries', 'species group', 'species subgroup','species', 'forma specialis', 'subspecies', 'morph','varietas', 'subvariety', 'forma', 'serogroup', 'serotype', 'biotype','strain', 'pathogroup','isolate', 'genotype', 'no rank', 'clade'],help = "Taxonomic rank whose distribution among genomes must be analysed, species by default." )
    parser.add_argument('-o', '--output_folder', required = True, help = "Output folder. Outputs include .csv files to analyse the distribution of desired rank in given genomes, accession_ID.txt with list of RefSeq genome accession IDs useful to download genomes from NCBI. ")
    parser.add_argument('-n', '--number_genomes', type=int, help = "Maximum number of genomes to be downloaded per taxon.")
    args = parser.parse_args()

    print('Download taxonomy from NCBI...')
    taxonomy, nodes_NCBI, ranks = download_taxonomy()
        
    #if output folder does not exist, create it
    if os.path.exists(args.output_folder) == False:
        os.makedirs(args.output_folder)

    print('Retrieving genomic information from provided data...')
    genomes_id = get_genomes(args.genome_list)
    complete_genomes_id = get_genomes(args.complete_list)

    #NCBI taxonomic rank hierarchy
    my_lineage = ['superkingdom', 'kingdom', 'subkingdom', 'superphylum', 'phylum', 'subphylum', 'infraphylum', 'superclass', 'class', 'subclass', 'infraclass', 'cohort', 'subcohort', 'superorder', 'order', 'suborder', 'infraorder', 'parvoorder', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'subgenus', 'section', 'subsection', 'series', 'subseries', 'species group', 'species subgroup','species', 'forma specialis', 'subspecies', 'morph','varietas', 'subvariety', 'forma', 'serogroup', 'serotype', 'biotype','strain', 'pathogroup','isolate', 'genotype', 'no rank', 'clade']

    results_df, result_dict = genome_analysis(args.rank, genomes_id)
    
    print('{} {} detected'.format(len(result_dict), args.rank))

    results_df.to_csv(os.path.join(args.output_folder,'id_name_number.csv'))
    counts_df = pd.DataFrame(data={'Species': results_df.groupby(['Number']).count()['Names']})
    counts_df['Percentage'] = counts_df['Species']/counts_df['Species'].sum()*100
    counts_df.to_csv(os.path.join(args.output_folder,'genome_numbers.csv'))

    if args.number_genomes:
        print('Retrieve {} RefSeq accession IDs per {}... '.format(args.number_genomes, args.rank))
        acc_ids, genomes_of_species, info_genomes = get_acc_ids(results_df,result_dict, args.number_genomes)


        with open(os.path.join(args.output_folder,'accession_ID.txt'), 'w') as file:
            file.writelines([id + '\n' for id in acc_ids])
            file.close()

        with open(os.path.join(args.output_folder,'genomes_per_species.txt'), 'w') as file:
            file.write('Species : NCBI accession IDs per species \n')
            out = [s + '\t:\t' + ' '.join(genomes_of_species[s]) + '\n' for s in genomes_of_species]
            file.writelines(out)
            file.close()

        with open(os.path.join(args.output_folder,'genomes_info.txt'), 'w') as file:
            file.write('Species : Number of genomes, Number of complete genomes \n')
            file.writelines([str(s) + '\t:\t' + ' '.join(info_genomes[s]) + '\n' for s in info_genomes])
            file.close()

        with open(os.path.join(args.output_folder,'genomes_id.txt'), 'w') as file:
            file.write('NCBI taxaID : NCBI accession IDs \n')
            out = [s + '\t:\t' + ' '.join(genomes_of_species[s]) + '\n' for s in genomes_of_species]
            file.writelines(out)
            file.close()