import os
from urllib import request
import subprocess
import csv
import sys
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import spacy
from functools import partial

from multiprocessing import Manager,Process

from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import numpy as np
import time

# nohup python imbd_data_loader.py > myprogram.out 2> myprogram.err &
csv.field_size_limit(sys.maxsize)
nbr_processes = 90
def embed_genre_dic(embed_dict, id2class_dic):
    mlb = MultiLabelBinarizer(classes=list(embed_dict.values()), sparse_output=True)
    def embed_genre(embed_dict, classes_str, mlb):
        igenre=[embed_dict[genre] for genre in classes_str]
        return mlb.fit_transform([set(igenre)])
    newD = {k: embed_genre(embed_dict, v, mlb) for k, v in id2class_dic.items()}
    return newD

def embed_word2vec(title, nlps):
    vector=None
    for nlp in nlps:
        #print(title)
        #s=time.time()

        doc = nlp(title)
        #e=time.time()
        #print('nlp time')
        #print(e-s)
        if vector is None:
            vector=doc.vector
        else:
            vector=np.concatenate((vector, doc.vector))
    return vector
IMDB_DIR= '../../data/imdb_data/'
def _download_imdb(flag_overwrite=False):
    if os.path.isdir(IMDB_DIR) is False:
        os.mkdir(IMDB_DIR)
    DOWNLOAD_INFO = [('title.basics.tsv.gz', 'https://datasets.imdbws.com/title.basics.tsv.gz'),
                     ('title.akas.tsv.gz', 'https://datasets.imdbws.com/title.akas.tsv.gz'),
                     ('title.crew.tsv.gz', 'https://datasets.imdbws.com/title.crew.tsv.gz'),
                     ('name.basics.tsv.gz', 'https://datasets.imdbws.com/name.basics.tsv.gz')]
    for save_name, url in DOWNLOAD_INFO:
        if os.path.isfile(os.path.join(IMDB_DIR, save_name[:-3])):
            print("Found {}, Skip".format(os.path.join(IMDB_DIR, save_name)))
        else:
            data_file = request.urlopen(url)
            with open(os.path.join(IMDB_DIR, save_name), 'wb') as output:
                output.write(data_file.read())
            subprocess.call(['gunzip', '{}/{}'.format(IMDB_DIR, save_name)])
            subprocess.call(['rm', '{}/{}'.format(IMDB_DIR, save_name)])

def read_imdb2dic(IMDB_DIR='../data/imdb_data/'):
    _download_imdb()
    titles2id_dic = {}
    titles2year_dic = {}
    _id2year_dic = {}
    id2info_dic = {}
    # TODO genre to multihot encoding
    id2genre_dic = {}
    with open(os.path.join(IMDB_DIR, "title.basics.tsv"), newline='', encoding='utf-8') as csvfile:
        IMDB_title_name =  (csv.reader(csvfile, delimiter='\t'))
        next(IMDB_title_name)
        for row in IMDB_title_name:
            str_id = row[0]
            title_type = row[1].lower()
            title1 = row[2].lower()
            title2 = row[3].lower()
            assert "\n" not in title1 and "\n" not in title2
            is_adult=row[4]
            start_year = row[5]
            end_year = row[6]
            runtime=row[7]

            start_year = None if start_year == '\\N' else float(start_year)
            end_year = None if end_year == '\\N' else float(end_year)
            is_adult = None if is_adult == '\\N' else float(is_adult)
            runtime = None if runtime == '\\N' else float(runtime)
            if start_year is not None and len(row) == 9:
                if str_id not in _id2year_dic:
                    _id2year_dic[str_id] = start_year
                if str_id not in id2info_dic:
                    id2info_dic[str_id] = [title1,title2,( (start_year), (end_year)), title_type, (is_adult), (runtime)]
                if str_id not in id2genre_dic:
                    id2genre_dic[str_id] = row[8].lower().split(",")

                if title1 not in titles2id_dic:
                    titles2id_dic[title1] = {}
                    titles2id_dic[title1][str_id] = start_year
                    titles2year_dic[title1] = {}
                    titles2year_dic[title1][start_year] = [str_id]
                else:
                    if str_id not in titles2id_dic[title1]:
                        titles2id_dic[title1][str_id] = start_year
                    if start_year not in titles2year_dic[title1]:
                        titles2year_dic[title1][start_year] = [str_id]
                    else:
                        titles2year_dic[title1][start_year].append(str_id)

                if title2 != title1:
                    if title2 not in titles2id_dic:
                        titles2id_dic[title2] = {}
                        titles2id_dic[title2][str_id] = start_year
                        titles2year_dic[title2] = {}
                        titles2year_dic[title2][start_year] = [str_id]
                    else:
                        if str_id not in titles2id_dic[title2]:
                            titles2id_dic[title2][str_id] = start_year
                        if start_year not in titles2year_dic[title2]:
                            titles2year_dic[title2][start_year] = [str_id]
                        else:
                            titles2year_dic[title2][start_year].append(str_id)
                else:
                    continue
            else:
                continue
    with open(os.path.join(IMDB_DIR, "title.ratings.tsv"), newline='', encoding='utf-8') as csvfile:
        IMDB_title_rating =  (csv.reader(csvfile, delimiter='\t'))
        next(IMDB_title_rating)
        for row in IMDB_title_rating:
            str_id = row[0]
            average_rating = row[1]
            num_votes = row[2]
        # TODO Check if this title is important or not.. does it exist in other dictionaries

            if str_id not in id2info_dic:
                id2info_dic[str_id] = [float(average_rating),float(num_votes)]
            else:
                id2info_dic[str_id]+=[float(average_rating),float(num_votes)]




    with open(os.path.join(IMDB_DIR, "title.akas.tsv"), newline='', encoding='utf-8') as csvfile2:
        IMDB_akas_name =  (csv.reader(csvfile2, delimiter="\t"))
        next(IMDB_akas_name)
        for row in IMDB_akas_name:
            str_id = row[0]
            title3 = row[2].lower()
            region=row[3].lower()
            language=row[4].lower

            # TODO decide if these attributes are needed.

            if "\n" in title3:
                print("len(title3)", len(title3))
                continue
            assert "\n" not in title3
            if str_id in _id2year_dic:
                year = _id2year_dic[str_id]
                if title3 not in titles2id_dic:
                    titles2id_dic[title3] = {}
                    titles2id_dic[title3][str_id] = year
                    titles2year_dic[title3] = {}
                    titles2year_dic[title3][year] = [str_id]
                else:
                    if str_id not in titles2id_dic[title3]:
                        titles2id_dic[title3][str_id] = year
                    if year not in titles2year_dic[title3]:
                        titles2year_dic[title3][year] = [str_id]
                    else:
                        titles2year_dic[title3][year].append(str_id)
            else:
                # only the akas with basic info are retained
                continue

    print("#title name: {}".format(len(titles2id_dic)))
    print("#movie id: {}".format(len(id2info_dic)))
    with open(os.path.join(IMDB_DIR,'_title_name2idsdic_dic.pkl'), 'wb') as f:
        pickle.dump(titles2id_dic, f)
    with open(os.path.join(IMDB_DIR,'_title_name2yeardic_dic.pkl'), 'wb') as f:
        pickle.dump(titles2year_dic, f)
    with open(os.path.join(IMDB_DIR, '_id2info_dic.pkl'), 'wb') as f:
        pickle.dump(id2info_dic, f)
    with open(os.path.join(IMDB_DIR, '_id2genre_dic.pkl'), 'wb') as f:
        pickle.dump(id2genre_dic, f)
    ###################################################################################
    ###################################################################################

    id2l_director_dic = {}
    id2l_writer_dic = {}
    id2l_principal_dic={}
    with open(os.path.join(IMDB_DIR, "title.principals.tsv"), newline='', encoding='utf-8') as csvfile:
        IMDB_title_principals =  (csv.reader(csvfile, delimiter='\t'))
        next(IMDB_title_principals)
        for row in IMDB_title_principals:
            str_id = row[0]
            ordering = row[1]
            person_id = row[2]
            job_category=row[3].lower()
            job_title = row[4]
            job_title = None if job_title == '\\N' else job_title.lower()
            character = row[5]
            character = None if character == '\\N' else character.lower()
            # TODO is it an entry per person and movie or list of persons
            if  str_id not in id2l_principal_dic:
                id2l_principal_dic[str_id] = [[person_id,job_category,job_title, character]]
            else:
                id2l_principal_dic[str_id] += [[person_id, job_category, job_title, character]]

    # TODO Possible Bug. How about when only one or 2 directors or writers exists?
    with open(os.path.join(IMDB_DIR, "title.crew.tsv"), newline='', encoding='utf-8') as csvfile:
        file_rows =  (csv.reader(csvfile, delimiter='\t'))
        next(file_rows)
        for row in file_rows:
            id = row[0]
            director_str = row[1]
            writer_str = row[2]

            if id in id2l_director_dic:
                print(id, id2l_director_dic[id])
            else:
                if director_str != "\\N" and len(director_str) > 2:
                    director_vec = director_str.split(",")
                    id2l_director_dic[id] = director_vec


            if id in id2l_writer_dic:
                print(id, id2l_writer_dic[id])
            else:
                if writer_str != "\\N" and len(writer_str) > 2:
                    writer_vec = writer_str.split(",")
                    id2l_writer_dic[id] = writer_vec
    with open(os.path.join(IMDB_DIR, '_id2director_dic.pkl'), 'wb') as f:
        pickle.dump(id2l_director_dic, f)
    with open(os.path.join(IMDB_DIR, '_id2writer_dic.pkl'), 'wb') as f:
        pickle.dump(id2l_writer_dic, f)
    # TODO test data dumping and loading
    with open(os.path.join(IMDB_DIR, '_id2_principal_dic.pkl'), 'wb') as f:
        pickle.dump(id2l_principal_dic, f)
    ###################################################################################
    ###################################################################################

    people_id2name_dic = {}
    with open(os.path.join(IMDB_DIR, "name.basics.tsv"), newline='', encoding='utf-8') as csvfile:
        file_rows =  (csv.reader(csvfile, delimiter='\t'))
        next(file_rows)
        for row in file_rows:
            id = row[0]
            name = row[1]
            birthyear=row[2]
            deathyear=row[3]
            primaryProfession=row[4]
            knownfortitles=row[5]
            if id in people_id2name_dic:
                print(id, people_id2name_dic[id])
            else:
                people_id2name_dic[id] = [name, birthyear, deathyear, primaryProfession, knownfortitles]
    with open(os.path.join(IMDB_DIR, '_people_id2name_dic.pkl'), 'wb') as f:
        pickle.dump(people_id2name_dic, f)

    print("IMDb dics generated ...")
    return titles2id_dic, titles2year_dic, id2info_dic, id2genre_dic, \
           id2l_director_dic, id2l_writer_dic,id2l_principal_dic, people_id2name_dic


def load_nlp_models(nlp_model='en_fr_lang'):
    if nlp_model == 'small':
        nlps_title = [spacy.load('xx_ent_wiki_sm')]
        nlps_characters=nlps_title
        nlps_primary_profession=nlps_title
    elif nlp_model=='en_fr_lang':
        nlp_eng = spacy.load('en_core_web_lg')
        nlp_french = spacy.load('fr_core_news_md')
        nlps_title= [nlp_eng,nlp_french]#,nlp_german,nlp_it,nlp_spanish]
        nlps_characters = [nlp_eng]
        nlps_primary_profession = [nlp_eng]
    elif nlp_model=='various_lang':
        nlp_eng = spacy.load('en_core_web_lg')
        nlp_french = spacy.load('fr_core_news_md')
        nlp_german = spacy.load('de_core_news_md')
        nlp_it = spacy.load('it_core_news_sm')
        nlp_spanish = spacy.load('es_core_news_md')
        nlps_title= [nlp_eng,nlp_french,nlp_german,nlp_it,nlp_spanish]
        nlps_characters = [nlp_eng]
        nlps_primary_profession = [nlp_eng]
    else:
        raise ValueError("Not supported.")
    return  nlps_title,nlps_characters,nlps_primary_profession

def parallel_dict_nlp_processing(dictionary_to_transform, nlps):
        '''
            Parallel processing of dict rows with shared memory for obtaining the nlp representation
            from the string entry of the dictionary
        :param dictionary_to_transform: The dictionary whose entries will be transformed
        :param nlps: the nlp models to use in the transformation
        :return:
        '''
        def embed_titles(d, vs, ks):
            for v, k in zip(vs, ks):
                d[k] = embed_word2vec(v, nlps)
        manager = Manager()
        d = manager.dict()
        keys, values = zip(*dictionary_to_transform.items())
        len_per_split = len(keys) // nbr_processes
        if len_per_split==0:
            len_per_split=1
        # TODO do not pass the whole value key only the subsets...
        job=[]
        for i in range(1, nbr_processes + 2):
            if len_per_split * i >= len(values):
                vs = values[len_per_split * (i - 1):]
                ks = keys[len_per_split * (i - 1):]
            else:
                vs = values[len_per_split * (i - 1):len_per_split * (i)]
                ks = keys[len_per_split * (i - 1):len_per_split * (i)]
            job+= [Process(target=embed_titles, args=(d, vs,ks,))]
        _ = [p.start() for p in job]
        _ = [p.join() for p in job]

        #print(d)
        return d


def read_subset_imdb2dic(nlp_model='en_fr_lang', IMDB_DIR='../data/imdb_data/'):
    #_download_imdb()
    id2numer_info_dic = {}
    id2str_info_dic = {}
    id2genre_dic = {}
    # TODO load first to dictionary and then process in parallel for the nlp model ...
    nlps_title, nlps_characters, nlps_primary_profession = load_nlp_models(nlp_model)

    with open(os.path.join(IMDB_DIR, "title.basics.tsv"), newline='', encoding='utf-8') as csvfile:
        IMDB_title_name =  (csv.reader(csvfile, delimiter='\t'))
        next(IMDB_title_name)
        genre_embed_dict={}
        count=0 #30 total
        for row in IMDB_title_name:
            #if count== 10:
                # for debugging
            #    break
            if len(row)==9:
                str_id = row[0]
                title_type = row[1].lower()
                title1 = row[2].lower()
                title2 = row[3].lower()
                assert "\n" not in title1 and "\n" not in title2
                is_adult=row[4]
                start_year = row[5]
                end_year = row[6]
                runtime=row[7]
                #print(row)
                start_year = None if start_year == '\\N' else float(start_year)
                end_year = None if end_year == '\\N' else float(end_year)
                is_adult = None if is_adult == '\\N' else float(is_adult)
                runtime = None if runtime == '\\N' else float(runtime)
                if start_year is not None and len(row) == 9:
                    if str_id not in id2numer_info_dic:
                        id2numer_info_dic[str_id] = [start_year, end_year, title_type, (is_adult), (runtime)]
                        id2str_info_dic[str_id] =title1+' '+title2

                    if str_id not in id2genre_dic:
                        id2genre_dic[str_id] = row[8].lower().split(",")
                        for gen in id2genre_dic[str_id]:
                            if gen not in genre_embed_dict:
                                genre_embed_dict[gen]=count
                                count+=1
                else:
                    continue
            else:
                continue
    s=time.time()
    id2genre_dic=embed_genre_dic(genre_embed_dict, id2genre_dic)
    e=time.time()
    print('Genre embedding runtime')
    print(e - s)
    s = time.time()
    # get the real dict from the proxy dictionary of the multiprocessing
    id2str_info_dic=dict(parallel_dict_nlp_processing(id2str_info_dic,nlps=nlps_title))
    e = time.time()
    print('Parallel processing runtime')
    print(e - s)
    print(len(id2str_info_dic.values()))
    #print(id2str_info_dic.keys())

    #print(id2str_info_dic)


    with open(os.path.join(IMDB_DIR, "title.ratings.tsv"), newline='', encoding='utf-8') as csvfile:
        IMDB_title_rating =  (csv.reader(csvfile, delimiter='\t'))
        next(IMDB_title_rating)
        for row in IMDB_title_rating:
            str_id = row[0]
            average_rating = row[1]
            num_votes = row[2]
        # TODO Check if this title is important or not.. does it exist in other dictionaries

            if str_id in id2numer_info_dic:
                id2numer_info_dic[str_id] +=\
                    [float(average_rating),float(num_votes)]
            else:
                id2numer_info_dic[str_id] = [float(average_rating), float(num_votes)]



    print("#movie id: {}".format(len(id2numer_info_dic)))

    with open(os.path.join(IMDB_DIR, '_id2numer_info_dic.pkl'), 'wb') as f:
        pickle.dump(id2numer_info_dic, f)
    with open(os.path.join(IMDB_DIR, '_id2str_info_dic.pkl'), 'wb') as f:
        pickle.dump(id2str_info_dic, f)   
    with open(os.path.join(IMDB_DIR, '_id2genre_dic.pkl'), 'wb') as f:
        pickle.dump(id2genre_dic, f)
    sys.stdout.flush()
    ###################################################################################
    ###################################################################################
    id2l_director_dic = {}
    id2l_writer_dic = {}
    id2l_principal_dic={}
    # for different categories
    mlb = MultiLabelBinarizer(classes=list(range(15)), sparse_output=True)
    s = time.time()
    with open(os.path.join(IMDB_DIR, "title.principals.tsv"), newline='', encoding='utf-8') as csvfile:
        IMDB_title_principals =  (csv.reader(csvfile, delimiter='\t'))
        next(IMDB_title_principals)
        total_category = 15
        job_category_count = 0
        job_category_dict = {}
        job_title_dict = {}
        zero_category=csr_matrix((total_category), dtype=np.int8)
        for row in IMDB_title_principals:
            if len(row)==6:
                str_id = row[0]
                ordering = row[1]
                person_id = row[2]
                job_category=row[3].lower()
                job_title = row[4]
                job_title = None if job_title == '\\N' else job_title.lower()
                character = row[5]
                character = None if character == '\\N' else character.lower()
                # TODO is it an entry per person and movie or list of persons
                if job_category is not None:
                    if job_category not in job_category_dict:
                        job_category_dict[job_category] = job_category_count
                        job_category_count += 1
                if job_category_count>total_category:
                    print(job_category_count)
                    total_category=job_category_count

                if  job_category is not None:
                    #s=time.time()
                    job_category_one_hot=mlb.fit_transform([set(([job_category_dict[job_category]]))])
                    #e=time.time()
                    #print(e-s)
                else:
                    job_category_one_hot=zero_category
                job_category=job_category_one_hot
                # TODO should we include job title and character played ? job category, more meaningfull
                if  str_id not in id2l_principal_dic:
                    id2l_principal_dic[str_id] = [[person_id,job_category]]#,job_title,character ]]
                else:
                    id2l_principal_dic[str_id] += [[person_id, job_category]]#, job_title,character]]

            else:
                continue
    e = time.time()
    print('Read and process principals')
    print(e - s)
    #print(len(id2str_info_dic.values()))

    # TODO Possible Bug. How about when only one or 2 directors or writers exists?
    with open(os.path.join(IMDB_DIR, "title.crew.tsv"), newline='', encoding='utf-8') as csvfile:
        file_rows =  (csv.reader(csvfile, delimiter='\t'))
        next(file_rows)
        for row in file_rows:
            id = row[0]
            director_str = row[1]
            writer_str = row[2]

            if id in id2l_director_dic:
                print(id, id2l_director_dic[id])
            else:
                if director_str != "\\N" and len(director_str) > 2:
                    director_vec = director_str.split(",")
                    id2l_director_dic[id] = director_vec


            if id in id2l_writer_dic:
                print(id, id2l_writer_dic[id])
            else:
                if writer_str != "\\N" and len(writer_str) > 2:
                    writer_vec = writer_str.split(",")
                    id2l_writer_dic[id] = writer_vec
    with open(os.path.join(IMDB_DIR, '_id2director_dic.pkl'), 'wb') as f:
        pickle.dump(id2l_director_dic, f)
    with open(os.path.join(IMDB_DIR, '_id2writer_dic.pkl'), 'wb') as f:
        pickle.dump(id2l_writer_dic, f)
    # TODO test data dumping and loading
    with open(os.path.join(IMDB_DIR, '_id2_principal_dic.pkl'), 'wb') as f:
        pickle.dump(id2l_principal_dic, f)
    sys.stdout.flush()
    ###################################################################################
    ###################################################################################

    people_id2name_dic = {}
    people_id2primaryProfession={}
    # for different proffesions
    mlb = MultiLabelBinarizer(classes=list(range(43)), sparse_output=True)
    s=time.time()
    with open(os.path.join(IMDB_DIR, "name.basics.tsv"), newline='', encoding='utf-8') as csvfile:
        file_rows =  (csv.reader(csvfile, delimiter='\t'))
        count_professions = 0
        professions_dict = {}
        next(file_rows)
        for row in file_rows:
            id = row[0]
            name = row[1]
            birthyear=row[2]
            deathyear=row[3]
            primaryProfession=row[4]
            knownfortitles=row[5]
            birthyear = None if birthyear == '\\N' else float(birthyear)
            deathyear = None if deathyear == '\\N' else float(deathyear)
            for prof in primaryProfession.split(","):
                if prof not in professions_dict:
                    professions_dict[prof] = count_professions
                    count_professions += 1
            #primaryProfession= None if len(primaryProfession)==0 else embed_word2vec(primaryProfession,nlps_primary_profession)
            if id in people_id2name_dic:
                print(id, people_id2name_dic[id])
            else:
                people_id2name_dic[id] = [name, birthyear, deathyear, knownfortitles]
                iprof=[professions_dict[prof] for prof in primaryProfession.split(',')]
                people_id2primaryProfession[id]=mlb.fit_transform([set(iprof)])
    with open(os.path.join(IMDB_DIR, '_people_id2name_dic.pkl'), 'wb') as f:
        pickle.dump(people_id2name_dic, f)
    with open(os.path.join(IMDB_DIR, '_people_id2primaryProfession.pkl'), 'wb') as f:
        pickle.dump(people_id2primaryProfession, f)
    print('Read and process people info')
    e = time.time()
    print(e - s)

    #print(len(id2str_info_dic.values()))

    print("IMDb dics generated ...")
    sys.stdout.flush()
    return  id2numer_info_dic,id2str_info_dic, id2genre_dic, id2l_director_dic, id2l_writer_dic,\
            id2l_principal_dic, people_id2name_dic, people_id2primaryProfession

def load_imdb_dics():
    with open(os.path.join(IMDB_DIR,'_title_name2idsdic_dic.pkl'), 'rb') as f:
        titles2id_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR,'_title_name2yeardic_dic.pkl'), 'rb') as f:
        titles2year_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR, '_id2info_dic.pkl'), 'rb') as f:
        id2info_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR, '_id2genre_dic.pkl'), 'rb') as f:
        id2genre_dic = pickle.load(f)

    with open(os.path.join(IMDB_DIR,'_id2director_dic.pkl'), 'rb') as f:
        id2l_director_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR,'_id2_principal_dic.pkl'), 'rb') as f:
        id2l_principal_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR,'_id2writer_dic.pkl'), 'rb') as f:
        id2l_writer_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR,'_people_id2name_dic.pkl'), 'rb') as f:
        people_id2name_dic = pickle.load(f)
    print("IMDb dics loaded ...")


    return titles2id_dic, titles2year_dic, id2info_dic, id2genre_dic, \
           id2l_director_dic, id2l_writer_dic,id2l_principal_dic, people_id2name_dic
def load_imdb_small_subset_dics():
    with open(os.path.join(IMDB_DIR,'_id2numer_info_dic_small.pkl'), 'rb') as f:
        id2numer_info_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR,'_id2str_info_dic_small.pkl'), 'rb') as f:
        id2str_info_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR, '_id2genre_dic_small.pkl'), 'rb') as f:
        id2genre_dic = pickle.load(f)

    with open(os.path.join(IMDB_DIR,'_id2l_director_dic_small.pkl'), 'rb') as f:
        id2l_director_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR,'_id2l_principal_dic_small.pkl'), 'rb') as f:
        id2l_principal_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR,'_id2l_writer_dic_small.pkl'), 'rb') as f:
        id2l_writer_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR,'_people_id2name_dic_small.pkl'), 'rb') as f:
        people_id2name_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR, '_people_id2primaryProfession_small.pkl'), 'rb') as f:
        people_id2primaryProfession = pickle.load(f)
    print("IMDb dics loaded ...")

    return id2numer_info_dic,id2str_info_dic, id2genre_dic, id2l_director_dic, id2l_writer_dic,\
            id2l_principal_dic, people_id2name_dic, people_id2primaryProfession

def load_imdb_subset_dics():
    with open(os.path.join(IMDB_DIR,'_id2numer_info_dic.pkl'), 'rb') as f:
        id2numer_info_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR,'_id2str_info_dic.pkl'), 'rb') as f:
        id2str_info_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR, '_id2genre_dic.pkl'), 'rb') as f:
        id2genre_dic = pickle.load(f)

    with open(os.path.join(IMDB_DIR,'_id2director_dic.pkl'), 'rb') as f:
        id2l_director_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR,'_id2_principal_dic.pkl'), 'rb') as f:
        id2l_principal_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR,'_id2writer_dic.pkl'), 'rb') as f:
        id2l_writer_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR,'_people_id2name_dic.pkl'), 'rb') as f:
        people_id2name_dic = pickle.load(f)
    with open(os.path.join(IMDB_DIR, '_people_id2primaryProfession.pkl'), 'rb') as f:
        people_id2primaryProfession = pickle.load(f)
    print("IMDb dics loaded ...")

    return id2numer_info_dic,id2str_info_dic, id2genre_dic, id2l_director_dic, id2l_writer_dic,\
            id2l_principal_dic, people_id2name_dic, people_id2primaryProfession


if __name__ == '__main__':
    print("==================================================")
    #load_imdb_subset_dics()
    #titles2id_dic, titles2year_dic, id2info_dic, id2genre_dic,\
    #id2l_director_dic, id2l_writer_dic, id2l_principal_dic,people_id2name_dic = read_imdb2dic()
    id2numer_info_dic, id2str_info_dic, id2genre_dic, id2l_director_dic, id2l_writer_dic, \
    id2l_principal_dic, people_id2name_dic, people_id2primaryProfession = read_subset_imdb2dic()