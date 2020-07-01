import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from classifiers import ClassifierMLP


def macro_micro_f1(y_test, y_pred):
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    print("Macro micro f1 " +str(macro_f1)+ " "+str(micro_f1))
    return macro_f1, micro_f1


def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def evaluate_results_nc(embeddings, labels, num_classes):
    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
    macro_str='Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])])

    micro_str='Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])])
    print(macro_str)
    print(micro_str)
    print('K-means test')
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std,macro_str,micro_str


def mlp_classifier(feats,use_cuda,n_hidden,lr_d,n_cepochs,multilabel,num_classes,labels,train_idx,val_idx,test_idx,device):
    ###
    # Use the encoded features for classification
    # Here we initialize the features using the reconstructed ones

    # feats = g.nodes[category].data['features']
    l2norm = 0.0001
    inp_dim = feats.shape[1]
    model = ClassifierMLP(input_size=inp_dim, hidden_size=n_hidden,out_size=num_classes)

    if use_cuda:
        model.cuda()
        feats=feats.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_d, weight_decay=l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()

    # TODO find all zero indices rows and remove.
    if len(labels.shape)>1:
        zero_rows=np.where(~(labels).cpu().numpy().any(axis=1))[0]

        train_idx=np.array(list(set(train_idx).difference(set(zero_rows))))
        val_idx = np.array(list(set(val_idx).difference(set(zero_rows))))
        test_idx = np.array(list(set(test_idx).difference(set(zero_rows))))

    train_indices = torch.tensor(train_idx).to(device).long()
    valid_indices = torch.tensor(val_idx).to(device).long()
    test_indices = torch.tensor(test_idx).to(device).long()




    best_val_acc = 0
    best_test_acc = 0
    labels_n=labels
    if multilabel is False:
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        loss_func = torch.nn.BCEWithLogitsLoss()

    for epoch in range(n_cepochs):
        optimizer.zero_grad()
        logits = model(feats)
        if multilabel:
            loss = loss_func(logits[train_idx].squeeze(1),
                             labels_n[train_idx])
        else:
            loss = loss_func(logits[train_idx].squeeze(1), torch.max(labels_n[train_idx], 1)[1])
        loss.backward()
        optimizer.step()
        pred = torch.sigmoid(logits).detach().cpu().numpy()

        train_acc = roc_auc_score(labels_n.cpu()[train_indices.cpu()].numpy(),
                                  pred[train_indices.cpu()],average='macro')
        val_acc = roc_auc_score(labels_n.cpu()[valid_indices.cpu()].numpy(),
                                pred[valid_indices.cpu()],average='macro')
        test_acc = roc_auc_score(labels_n.cpu()[test_indices.cpu()].numpy()
                                 , pred[test_indices.cpu()],average='macro')
        test_acc_w = roc_auc_score(labels_n.cpu()[test_indices.cpu()].numpy()
                                 , pred[test_indices.cpu()], average='weighted')
        macro_test, micro_test = macro_micro_f1(
            torch.max(labels[test_indices], 1)[1].cpu(), torch.max(logits[test_indices], 1)[1].cpu())
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 5 == 0:
            print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f), Weighted Test Acc %.4f' % (
                loss.item(),
                train_acc.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),test_acc_w.item()
            ))
    print()
    return best_test_acc