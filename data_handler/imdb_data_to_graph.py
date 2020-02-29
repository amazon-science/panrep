import numpy as np
from data_handler import imbd_data_loader
import dgl
import pickle
import os

def embed_genre():
    return
def embed_movie_info(movie_id,id2info_dic,id2genre_dic,feat_nbr):
    feat=np.zeros((feat_nbr))
    if movie_id in id2info_dic:
        feat[0] = len(id2info_dic[movie_id][0])
    return feat
def embed_person_info(person_id,people_id2name_dic,feat_nbr):
    feat = np.zeros((feat_nbr))
    if person_id in people_id2name_dic:
        feat[0]=len(people_id2name_dic[person_id][0])
    return feat
# TODO how to treat missing attributes

def graph_generator(IMDB_DIR=' ../data/imdb_data/'):
    _, _, id2info_dic, id2genre_dic, \
    id2l_director_dic,  id2l_writer_dic, id2l_principal_dic, people_id2name_dic= imbd_data_loader.load_imdb_dics()

    use_principals = False

    # Generate edge lists
    id2l_director_edge_list = [(k, d) for k, v in id2l_director_dic.items() for d in v]
    id2l_writer_edge_list = [(k, w) for k, v in id2l_writer_dic.items() for w in v]
    id2l_principal_edge_list = [(k, w) for k, v in id2l_principal_dic.items() for w in v]
    # extract ids that participate in the abore relationships
    # extract ids of movies
    moviesdirected = [i[0] for i in id2l_director_edge_list]
    movieswithprincipals=[i[0] for i in id2l_principal_edge_list]
    principals = [i[1][0] for i in id2l_principal_edge_list]
    # edge feature information among principal and movie
    dictOfPrincipals = {(i[0],i[1][0]):i[1][1:] for i in id2l_principal_edge_list}

    # extract ids of persons
    directors = [i[1] for i in id2l_director_edge_list]
    movieswritten = [i[0] for i in id2l_writer_edge_list]
    writers = [i[1] for i in id2l_writer_edge_list]

    # Need to map all ids to int
    # create list of all movies and people
    if use_principals:
        movie_imdb = np.concatenate((moviesdirected, movieswritten,movieswithprincipals))
        people_imdb = np.concatenate(( directors, writers,principals))
    else:
        movie_imdb = np.concatenate((moviesdirected, movieswritten))
        people_imdb = np.concatenate((directors, writers))
    # extract unique ids movies and people
    uniq_mov, edges_mov = np.unique(movie_imdb, return_inverse=True)
    uniq_people, edges_peopl = np.unique(people_imdb, return_inverse=True)

    # map the uniq ids to integers
    dictOfMovies = {uniq_mov[i]: i for i in range(len(uniq_mov))}
    dictOfPeople = {uniq_people[i]: i for i in range(len(uniq_people))}

    # map string edge list to int id edge list
    intid2l_writer_edge_list = [(dictOfMovies[k], dictOfPeople[v]) for (k, v) in id2l_writer_edge_list]
    intid2l_director_edge_list = [(dictOfMovies[k], dictOfPeople[v]) for (k, v) in id2l_director_edge_list]


    # create graph using edge lists
    if use_principals:
        intid2l_principal_edge_list = [(dictOfMovies[k], dictOfPeople[v[0]]) for (k, v) in id2l_principal_edge_list]
        g= dgl.heterograph({('movie','directed_by','person'):intid2l_director_edge_list,
                            ('movie','written_by','person'):intid2l_writer_edge_list,
                        ('movie','participated_by','person'):intid2l_principal_edge_list})
    else:
        g = dgl.heterograph({('movie', 'directed_by', 'person'): intid2l_director_edge_list,
                             ('movie', 'written_by', 'person'): intid2l_writer_edge_list})

    # create features
    nbr_person_feat=1
    nbr_movie_feat=1
    people_feat = np.zeros((len(uniq_people), nbr_person_feat))

    for i in range(len(uniq_people)):
        person_id=uniq_people[i]
        people_feat[i,:]=embed_person_info(person_id,people_id2name_dic,nbr_person_feat)
    movie_feat = np.zeros((len(uniq_mov), nbr_movie_feat))
    for i in range(len(uniq_mov)):
        movie_id=uniq_mov[i]
        movie_feat[i,:]=embed_movie_info(movie_id,id2info_dic,id2genre_dic,nbr_movie_feat)
    # TODO conver to torch th.array()
    g.nodes['movie'].data['features'] = movie_feat
    # TODO check datatype
    g.nodes['person'].data['features'] = people_feat
    dgl
    pickle.dump(g, open(os.path.join(IMDB_DIR, "graph.pickle"), "wb"),
                protocol=4);

if __name__ == '__main__':
    print("==================================================")
    graph_generator()