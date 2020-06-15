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
title_dict = {}
name_dict = {}

def handle_title_basics(IMDB_DIR='../data/imdb_data/'):
    min_sy = 2020
    min_ey = 2020
    min_runtime = 120
    records = []
    with open(os.path.join(IMDB_DIR, "title.basics.tsv"), newline='', encoding='utf-8') as csvfile:
        IMDB_title_name =  (csv.reader(csvfile, delimiter='\t'))
        row = next(IMDB_title_name)
        records.append("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(row[0], row[1], row[2], row[3],
            row[4], row[5], row[6], row[7]))
        for i, row in enumerate(IMDB_title_name):
            if len(row) < 9:
                print(i)
            str_id = row[0]
            title_dict[str_id] = 0
            title_type = row[1].lower()
            title1 = row[2].lower()
            title2 = row[3].lower()
            assert "\n" not in title1 and "\n" not in title2
            is_adult=row[4]
            start_year = row[5]
            end_year = row[6]
            runtime=row[7]

            start_year = -1 if start_year == '\\N' else int(start_year)
            end_year = -1 if end_year == '\\N' else int(end_year)
            is_adult = -1 if is_adult == '\\N' else int(is_adult)
            runtime = -1 if runtime == '\\N' else int(runtime)
            if start_year != -1 and start_year < min_sy:
                min_sy = start_year
            if end_year != -1 and end_year < min_ey:
                min_ey = end_year
            if runtime != -1 and runtime < min_runtime:
                min_runtime = runtime

            records.append("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(str_id, title_type,
                title1, title2, is_adult, start_year, end_year, runtime))

    with open("title_basisc_info.tsv", "w+", encoding='utf-8') as f:
        f.writelines(records)

def handle_title_akas(IMDB_DIR='../data/imdb_data/'):
    records = []
    with open(os.path.join(IMDB_DIR, "title.akas.tsv"), newline='', encoding='utf-8') as csvfile:
        IMDB_akas_name = csv.reader(csvfile, delimiter="\t")
        row = next(IMDB_akas_name)
        print(row)
        records.append("{}\t{}\n".format(row[0], row[3]))

        region_dict = {}
        for row in IMDB_akas_name:
            str_id = row[0]
            region=row[3].lower()
            if region_dict.get(str_id, None) is None:
                region_dict[str_id] = []
            region_dict[str_id].append(region)

        for key, val in region_dict.items():
            if title_dict.get(key, None) is None:
                continue

            regions = "{}".format(val[0])
            for i, v in enumerate(val):
                if i == 0:
                    continue
                regions = "{},{}".format(regions, v)
            records.append("{}\t{}\n".format(key, regions))
            title_dict[key] = 1

        for key, val in title_dict.items():
            if val == 0:
                records.append("{}\t{}\n".format(key, "unknow"))

    with open("title_akas.tsv", "w+", encoding='utf-8') as f:
        f.writelines(records)

def handle_principals(IMDB_DIR='../data/imdb_data/'):
    records = []
    with open(os.path.join(IMDB_DIR, "title.principals.tsv"), newline='', encoding='utf-8') as csvfile:
        IMDB_title_pricipals = (csv.reader(csvfile, delimiter="\t"))
        row = next(IMDB_title_pricipals)
        records.append("{}\t{}\t{}\n".format(row[0], row[2], row[3])) 

        for row in IMDB_title_pricipals:
            str_id = row[0]
            person = row[2]
            job = row[3]
            if name_dict.get(person, None) is None:
                print("unknow user {}".format(person))
                continue 
            if title_dict.get(str_id, None) is None:
                print("unknown principal title {}".format(str_id))
                continue 
            records.append("{}\t{}\t{}\n".format(str_id, person, job))

    with open("title_principals.tsv", "w+", encoding='utf-8') as f:
        f.writelines(records)

def handle_crew(IMDB_DIR='../data/imdb_data/'):
    director_records = []
    writer_records = []
    with open(os.path.join(IMDB_DIR, "title.crew.tsv"), newline='', encoding='utf-8') as csvfile:
        IMDB_crew = (csv.reader(csvfile, delimiter="\t"))
        row = next(IMDB_crew)    
        director_records.append("{}\t{}\n".format(row[0], row[1]))
        writer_records.append("{}\t{}\n".format(row[0], row[2]))

        for row in IMDB_crew:
            str_id = row[0]
            if title_dict.get(str_id, None) is None:
                print("unknown crew  title {}".format(str_id))
                continue
            if row[1] != '\\N':
                directors = row[1].split(',')
                for d in directors:
                    if name_dict.get(d, None) is None:
                        print("unknow user {}".format(d))
                        continue
                    director_records.append("{}\t{}\n".format(str_id, d))

            if row[2] != '\\N':
                writers = row[2].split(',')
                for w in writers:
                    if name_dict.get(w, None) is None:
                        print("unknow user {}".format(w))
                        continue
                    writer_records.append("{}\t{}\n".format(str_id, w))

    with open("title_crew_director.tsv", "w+", encoding='utf-8') as f:
        f.writelines(director_records)

    with open("title_crew_writer.tsv", "w+", encoding='utf-8') as f:
        f.writelines(writer_records)

def handle_name(IMDB_DIR='../data/imdb_data/'):
    records = []
    know_records = []
    with open(os.path.join(IMDB_DIR, "name.basics.tsv"), newline='', encoding='utf-8') as csvfile:
        IMDB_name = (csv.reader(csvfile, delimiter='\t'))
        row = next(IMDB_name)
        records.append("{}\t{}\n".format(row[0], row[4]))
        know_records.append("{}\t{}\n".format(row[0], row[5]))

        for row in IMDB_name:
            str_id = row[0]
            primaryProfession=row[4]
            if primaryProfession == '\\N':
                primaryProfession = "none"
            knownfortitles=row[5]
            if knownfortitles != '\\N':
                kft = knownfortitles.split(',')
                for k in kft:
                    if title_dict.get(k, None) is None:
                        print("unknown name title {}".format(k))
                        continue
 
                    know_records.append("{}\t{}\n".format(str_id, k))

            records.append("{}\t{}\n".format(str_id, primaryProfession))
            name_dict[str_id] = 0

    with open("name_basics.tsv", "w+", encoding='utf-8') as f:
        f.writelines(records)

    with open("name_knowfortitles.tsv", "w+", encoding='utf-8') as f:
        f.writelines(know_records)

def handle_genre_label(IMDB_DIR='../data/imdb_data/'):
    records = []
    empty_records = []
    with open(os.path.join(IMDB_DIR, "title.basics.tsv"), newline='', encoding='utf-8') as csvfile:
        IMDB_title_name = (csv.reader(csvfile, delimiter='\t'))
        row = next(IMDB_title_name)
        records.append("{}\t{}\n".format(row[0], row[8]))
        empty_records.append("{}\n".format(row[0]))

        for row in IMDB_title_name:
            if len(row) < 9:
                continue
            str_id=row[0]
            genre=row[8]
            if title_dict.get(str_id, None) is None:
                print("unknown genre title {}".format(str_id))
                continue
            if genre == '\\N':
                empty_records.append("{}\n".format(str_id))
            else:
                records.append("{}\t{}\n".format(row[0], row[8].lower()))

    with open("title_genre.tsv", "w+", encoding='utf-8') as f:
        f.writelines(records)

    with open("title_no_genre.tsv", "w+", encoding='utf-8') as f:
        f.writelines(empty_records)

handle_title_basics("../../data/imdb_data/")
handle_name("../../data/imdb_data/")
handle_title_akas("../../data/imdb_data/")
handle_principals("../../data/imdb_data/")
handle_crew("../../data/imdb_data/")
handle_genre_label("../../data/imdb_data/")
