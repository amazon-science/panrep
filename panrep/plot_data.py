import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')
def panrep():
    def split_acc(acc):
        acc = float(acc.split(" ")[1].split("~")[0])
        return acc

    def plot_results(results, paramlist):
        plots = {}
        elems = list(results.keys())[0]
        sets = []
        for el in list(elems):
            sets += [set()]
        keys = list(results.keys())
        experiment = keys[-1]
        keys = keys[:-1]
        for key in keys:
            (n_epochs, n_layers, n_hidden, n_bases, fanout, lr, dropout,
             use_link_prediction, use_reconstruction_loss,
             use_infomax_loss, mask_links, use_self_loop,
             use_node_motif, num_cluster, single_layer, num_motif_cluster) = key
            for i in range(len(list(key))):
                sets[i].add(key[i])
            plots[key] = split_acc(results[key])

        count_confs = [len(conf) for conf in sets]
        plot_over = 0
        key = list(key)
        legend = []
        confs = [2, 15]

        for conf in sets[confs[0]]:
            for conf2 in sets[confs[1]]:
                legend += [paramlist[confs[0]] + ' : ' + str(conf) +
                           paramlist[confs[1]] + ' : ' + str(conf2)]
                y = []
                key[confs[0]] = conf
                key[confs[1]] = conf2
                x = sorted(sets[plot_over])
                for el in x:
                    key[plot_over] = el
                    y += [plots[tuple(key)]]
                plt.plot(x, y)
        plt.legend(legend, loc='center left', bbox_to_anchor=(0.1, 1.4))
        plt.show()

    paramlist = "n_epochs, n_layers, n_hidden, n_bases, fanout, lr, dropout,use_link_prediction, use_reconstruction_loss,use_infomax_loss, mask_links, use_self_loop,use_node_motif, num_cluster, single_layer, num_motif_cluster"
    paramlist = paramlist.split(',')
    file = "2020-04-15-22:46:17.306660.pickle"

    results = pickle.load(open("results/" + file, 'rb'))
    plot_results(results, paramlist)
def endtoend():
    def split_acc(acc):
        tes_acc = float(acc.split(" ")[2])
        endend_acc = float(acc.split(" ")[5].split("~")[0])
        return tes_acc,endend_acc

    def plot_results(results, paramlist):
        plots = {}
        elems = list(results.keys())[0]
        sets = []
        for el in list(elems):
            sets += [set()]
        keys = list(results.keys())
        experiment = keys[-1]
        keys = keys[:-1]
        for key in keys:
            (n_epochs, n_layers, n_hidden, n_bases, fanout, lr,
             dropout, use_self_loop, K)= key
            for i in range(len(list(key))):
                sets[i].add(key[i])

            plots[key] = split_acc(results[key])

        plot_over = 0
        key = list(key)
        legend = []
        confs = [8,2]

        for conf in sets[confs[0]]:
            title=paramlist[confs[0]] + ' : ' + str(conf)
            legend=[]
            for conf2 in sorted(sets[confs[1]]):
                legend+=[paramlist[confs[1]] + ' : ' + str(conf2)]
                #legend += [cur_legend]
                y = []
                key[confs[0]] = conf
                key[confs[1]] = conf2
                x = sorted(sets[plot_over])
                for el in x:
                    key[plot_over] = el
                    y += [plots[tuple(key)][1]]
                plt.plot(x, y)

            plt.legend(legend, loc='center left', bbox_to_anchor=(-0.1, 1.2))
            plt.xlabel('Epochs')
            plt.ylabel('Macro-F1')
            plt.title(title)
            plt.show()
        for conf in sets[confs[0]]:
            title = paramlist[confs[0]] + ' : ' + str(conf)
            legend = []
            for conf2 in sorted(sets[confs[1]]):
                legend += [paramlist[confs[1]] + ' : ' + str(conf2)]
                # legend += [cur_legend]
                y = []
                key[confs[0]] = conf
                key[confs[1]] = conf2
                x = sorted(sets[plot_over])
                for el in x:
                    key[plot_over] = el
                    y += [plots[tuple(key)][0]]
                plt.plot(x, y)

            plt.legend(legend, loc='center left', bbox_to_anchor=(-0.1, 1.2))
            plt.xlabel('Epochs')
            plt.ylabel('Macro-F1')
            plt.title(title +" direct RGCN classifier")
            plt.show()


    paramlist = "n_epochs, n_layers, n_hidden, n_bases, fanout, lr, dropout, use_self_loop, K"
    paramlist = paramlist.split(',')
    file = "2020-04-16-01:57:06.865085.pickle"

    results = pickle.load(open("results/end_to_end_node_classification/" + file, 'rb'))
    plot_results(results, paramlist)

def finetune():
    def split_acc(acc):
        panrep_acc = float(acc.split(" ")[4].split("~")[0])
        tes_acc = float(acc.split(" ")[24])
        finpanrep_acc = float(acc.split(" ")[30].split("~")[0])
        return panrep_acc,tes_acc,finpanrep_acc

    def plot_results(results, paramlist):
        plots = {}
        elems = list(results.keys())[0]
        sets = []
        for el in list(elems):
            sets += [set()]
        keys = list(results.keys())
        experiment = keys[-1]
        keys = keys[:-1]
        for key in keys:
            (n_epochs,n_fine_tune_epochs, n_layers, n_hidden, n_bases, fanout, lr, dropout,
             use_link_prediction, use_reconstruction_loss,
             use_infomax_loss, mask_links, use_self_loop,
             use_node_motif,num_cluster,single_layer,motif_cluster,k_fold) = key
            for i in range(len(list(key))):
                sets[i].add(key[i])

            plots[key] = split_acc(results[key])

        plot_over = 0
        key = list(key)

        confs = [1, 3,9,10,13,17]
        experiment=[n_layers]
        i=0
        for conf6 in sets[confs[5]]:

            for conf in sets[confs[0]]:


                for conf2 in sets[confs[1]]:
                    experiment = paramlist[confs[5]] + \
                                 ' : ' + str(conf6) + paramlist[confs[0]] + ' : ' + str(conf) + paramlist[
                                     confs[1]] + ' : ' + str(conf2)

                    legend = []
                    i+=1
                    plt.figure(num=i,figsize=(12,10))

                    for conf3 in sets[confs[2]]:

                        for conf4 in sets[confs[3]]:

                            for conf5 in sets[confs[4]]:
                                    skip_this=False
                                    cur_legend =  paramlist[confs[2]] + ':' + str(conf3)
                                    cur_legend += paramlist[confs[3]] + ':' + str(conf4)
                                    cur_legend += paramlist[confs[4]] + ':' + str(conf5)
                                    y = []
                                    key[confs[0]] = conf
                                    key[confs[1]] = conf2
                                    key[confs[2]] = conf3
                                    key[confs[3]] = conf4
                                    key[confs[4]] = conf5
                                    key[confs[5]] = conf6

                                    x = sorted(sets[plot_over])

                                    for el in x:
                                        key[plot_over] = el
                                        if tuple(key) not in plots:
                                            skip_this=True
                                            break
                                        y += [plots[tuple(key)][0]]
                                    if skip_this:
                                        break
                                    plt.plot(x, y)
                                    legend += ["PR" + cur_legend]
                                    legend+=["PR-FT" +cur_legend]
                                    y=[]
                                    for el in x:
                                        key[plot_over] = el
                                        y += [plots[tuple(key)][2]]
                                    plt.plot(x, y)

                    plt.legend(legend, loc='center left', bbox_to_anchor=(-0, 1.1), ncol=3,prop=fontP)
                    plt.xlabel('Epochs')
                    plt.ylabel('Macro-F1')
                    plt.title(experiment)
                    plt.show()
                    legend=[]
        '''
        for conf in sets[confs[0]]:
            cur_legend=paramlist[confs[0]] + ' : ' + str(conf)
            for conf2 in sets[confs[1]]:
                cur_legend+=paramlist[confs[1]] + ' : ' + str(conf2)
                #legend += [cur_legend]
                y = []
                key[confs[0]] = conf
                key[confs[1]] = conf2
                x = sorted(sets[plot_over])
                legend+=["MLP classifier"]
                for el in x:
                    key[plot_over] = el
                    y += [plots[tuple(key)][1]]
                plt.plot(x, y)

        plt.legend(legend, loc='center left', bbox_to_anchor=(-0.1, 1.1))
        plt.xlabel('Epochs')
        plt.ylabel('Macro-F1')
        plt.title(experiment)
        plt.show()
        '''

    paramlist = "n_epochs,n_ft_ep, n_layers, n_h, n_b, fanout, lr, dropout, use_link_prediction, R, I, mask_links, use_self_loop, M,num_cluster,single_layer, num_motif_clus,k_fold"
    paramlist = paramlist.split(',')
    file = "2020-04-18-15:45:02.991956.pickle"

    results = pickle.load(open("results/finetune_node_classification/" + file, 'rb'))
    plot_results(results, paramlist)

def finetune2():
    def split_acc(acc):
        panrep_acc = float(acc.split(" ")[4].split("~")[0])
        tes_acc = float(acc.split(" ")[24])
        finpanrep_acc = float(acc.split(" ")[30].split("~")[0])
        return panrep_acc,tes_acc,finpanrep_acc

    def plot_results(results, paramlist):
        plots = {}
        elems = list(results.keys())[0]
        sets = []
        for el in list(elems):
            sets += [set()]
        keys = list(results.keys())
        experiment = keys[-1]
        keys = keys[:-1]
        for key in keys:
            (n_epochs,n_fine_tune_epochs, n_layers, n_hidden, n_bases, fanout, lr, dropout,
             use_link_prediction, use_reconstruction_loss,
             use_infomax_loss, mask_links, use_self_loop,
             use_node_motif,num_cluster,single_layer,motif_cluster,k_fold) = key
            for i in range(len(list(key))):
                sets[i].add(key[i])

            plots[key] = split_acc(results[key])

        plot_over = 0
        key = list(key)
        legend = []
        confs = [2, 15]
        experiment=[n_fine_tune_epochs, n_layers, n_hidden]
        experiment=str(experiment)

        for conf in sets[confs[0]]:
            cur_legend=paramlist[confs[0]] + ' : ' + str(conf)
            for conf2 in sets[confs[1]]:
                cur_legend+=paramlist[confs[1]] + ' : ' + str(conf2)
                #legend += [cur_legend]
                y = []
                key[confs[0]] = conf
                key[confs[1]] = conf2
                x = sorted(sets[plot_over])
                legend+=["PanRepMI"]
                for el in x:
                    key[plot_over] = el
                    y += [plots[tuple(key)][0]]
                plt.plot(x, y)
                legend+=["PanRepMI+ Finetune"]
                y=[]
                for el in x:
                    key[plot_over] = el
                    y += [plots[tuple(key)][2]]
                plt.plot(x, y)

        plt.legend(legend, loc='center left', bbox_to_anchor=(-0.1, 1.1))
        plt.xlabel('Epochs')
        plt.ylabel('Macro-F1')
        plt.title(experiment)
        plt.show()
        legend=[]
        for conf in sets[confs[0]]:
            cur_legend=paramlist[confs[0]] + ' : ' + str(conf)
            for conf2 in sets[confs[1]]:
                cur_legend+=paramlist[confs[1]] + ' : ' + str(conf2)
                #legend += [cur_legend]
                y = []
                key[confs[0]] = conf
                key[confs[1]] = conf2
                x = sorted(sets[plot_over])
                legend+=["MLP classifier"]
                for el in x:
                    key[plot_over] = el
                    y += [plots[tuple(key)][1]]
                plt.plot(x, y)

        plt.legend(legend, loc='center left', bbox_to_anchor=(-0.1, 1.1))
        plt.xlabel('Epochs')
        plt.ylabel('Macro-F1')
        plt.title(experiment)
        plt.show()


    paramlist = "n_epochs,n_fine_tune_epochs, n_layers, n_hidden, n_bases, fanout, lr, dropout, use_link_prediction, use_reconstruction_loss, use_infomax_loss, mask_links, use_self_loop, use_node_motif,num_cluster,single_layer, motif_cluster,k_fold"
    paramlist = paramlist.split(',')
    file = "2020-04-15-23:20:15.052926.pickle"

    results = pickle.load(open("results/finetune_node_classification/" + file, 'rb'))
    plot_results(results, paramlist)

if __name__ == '__main__':
    finetune()