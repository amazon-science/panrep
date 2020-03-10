import snap
import os
import datetime

from utils.network_utils import load_mode_to_graph, load_crossnet_to_graph

today = datetime.date.today()
datestring = "20200304"
context = snap.TTableContext()
# Graph object to hold multimodal cancer network
Graph = snap.TMMNet.New()
mode_table_filenames = [
    "../aux_files/mambo_master/datasets/cancer_example/chemical/miner-chemical-%s.tsv" % datestring,
    "../aux_files/mambo_master/datasets/cancer_example/disease/miner-disease-%s.tsv" % datestring,
    "../aux_files/mambo_master/datasets/cancer_example/function/miner-function-%s.tsv" % datestring,
    "../aux_files/mambo_master/datasets/cancer_example/gene/miner-gene-%s.tsv" % datestring,
    "../aux_files/mambo_master/datasets/cancer_example/protein/miner-protein-%s.tsv" % datestring,
]
for mode_table in mode_table_filenames:
    splitName = mode_table.split('-')
    load_mode_to_graph(splitName[1], mode_table, Graph, context)

link_table_filenames = [
    "../aux_files/mambo_master/datasets/cancer_example/chemical-chemical/miner-chemical-chemical-%s.tsv" % datestring,
    "../aux_files/mambo_master/datasets/cancer_example/chemical-protein/miner-chemical-protein-%s.tsv" % datestring,
    "../aux_files/mambo_master/datasets/cancer_example/disease-chemical/miner-disease-chemical-%s.tsv" % datestring,
    "../aux_files/mambo_master/datasets/cancer_example/disease-disease/miner-disease-disease-%s.tsv" % datestring,
    "../aux_files/mambo_master/datasets/cancer_example/disease-function/miner-disease-function-%s.tsv" % datestring,
    "../aux_files/mambo_master/datasets/cancer_example/disease-protein/miner-disease-protein-%s.tsv" % datestring,
    "../aux_files/mambo_master/datasets/cancer_example/function-function/miner-function-function-%s.tsv" % datestring,
    "../aux_files/mambo_master/datasets/cancer_example/gene-protein/miner-gene-protein-%s.tsv" % datestring,
    "../aux_files/mambo_master/datasets/cancer_example/protein-function/miner-protein-function-%s.tsv" % datestring,
]

for link_table in link_table_filenames:
    link_table_name = os.path.basename(link_table)
    splitName = link_table_name.split('-')
    crossnetName = splitName[1] + "-" + splitName[2] + "-" + 'id'
    srcModeName = splitName[1]
    dstModeName = splitName[2]
    load_crossnet_to_graph(context, crossnetName, srcModeName, dstModeName, link_table, Graph)


types = [
'coexpression',
'colocalization',
'genetic_interactions',
'pathway',
'physical_interactions',
'predicted',
]

for typ in types:
    link_table = "../aux_files/mambo_master/datasets/cancer_example/gene-gene/%s_links/miner-gene-gene-%s.tsv" % (typ, datestring)
    link_table_name = os.path.basename(link_table)
    splitName = link_table_name.split('-')
    crossnetName = splitName[1] + '-' + splitName[2] + "-" + 'id'
    srcModeName = splitName[1]
    dstModeName = splitName[2]
    load_crossnet_to_graph(context, crossnetName, srcModeName, dstModeName, link_table, Graph, prefix=typ)




types = [
'coexpression',
'cooccurence',
'database',
'experimental',
'neighborhood',
'textmining',
]

for typ in types:
    link_table = "../aux_files/mambo_master/datasets/cancer_example/protein-protein/%s_links/miner-protein-protein-%s.tsv" % (typ, datestring)
    link_table_name = os.path.basename(link_table)
    splitName = link_table_name.split('-')
    cossnetName = splitName[1] + '-' + splitName[2] + "-" + 'id'
    srcModeName = splitName[1]
    dstModeName = splitName[2]
    load_crossnet_to_graph(context, crossnetName, srcModeName, dstModeName, link_table, Graph, prefix=typ)


output_dir = "output/cancer_example"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

graph_name = "cancer_example.graph"
outputPath = os.path.join(output_dir, graph_name)

print("Saved in: %s" % outputPath)
FOut = snap.TFOut(outputPath)
Graph.Save(FOut)
FOut.Flush()