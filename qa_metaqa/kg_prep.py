import torch
from tqdm import tqdm
from torch import LongTensor
from collections import defaultdict
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel
from utils import add_elems, read_obj, write_obj, stop_words, bert_encode
from argparse import ArgumentParser


def prep_kg() -> None:
    print("\n* preparing the knowledge graph ...")
    num_ents = 0  # num of entities
    ent2id = {}  # entity strings to ids

    num_rels = 0  # num of relations
    rel2id = {}  # relation strings to ids

    num_trps = 0  # num of triples
    trp2id = {}  # (head_id, rel_id, tail_id) to triple_id

    with open(args.in_path + "kb.txt", "r") as f:
        line = f.readline()
        while line:
            elems = line.rstrip().split("|")
            assert len(elems) == 3, "\t* found an unusual line: {}".format(line)
            num_ents = add_elems(elem=elems[0], data_dict=ent2id, num=num_ents)
            num_rels = add_elems(elem=elems[1], data_dict=rel2id, num=num_rels)
            num_ents = add_elems(elem=elems[2], data_dict=ent2id, num=num_ents)
            num_trps = add_elems(elem=(ent2id[elems[0]], rel2id[elems[1]], ent2id[elems[2]]), data_dict=trp2id, num=num_trps)
            line = f.readline()

    kg = {"num_ents": num_ents, "ent2id": ent2id, "num_rels": num_rels, "rel2id": rel2id, "num_trps": num_trps, "trp2id": trp2id}
    write_obj(obj=kg, obj_name="the knowledge graph", file_path=args.out_path + "kg.pickle")


def prep_geo_graph() -> None:
    print("\n* preparing the geometric graph (only original edges) ...")
    kg = read_obj(obj_name="the knowledge graph", file_path=args.out_path + "kg.pickle")
    num_trps = kg["num_trps"]
    trp2id = kg["trp2id"]
    num_ents = kg["num_ents"]

    x = torch.arange(num_ents, dtype=torch.long).unsqueeze(1)  # original ent ids, size: (num_ents, 1)
    edge_index = LongTensor(2, num_trps)  # head and tail entities
    edge_attr = LongTensor(num_trps)  # relations
    edge_type = LongTensor(num_trps)  # edge type, 0: original, 1: inverse, 2: self-loop

    id2trp = {}  # triple id to triple tuple (h, r, t)
    for key, value in trp2id.items():
        id2trp[value] = key

    # original edges
    for trp_id in range(num_trps):
        h = id2trp[trp_id][0]
        r = id2trp[trp_id][1]
        t = id2trp[trp_id][2]
        edge_index[0, trp_id] = h
        edge_index[1, trp_id] = t
        edge_attr[trp_id] = r
        edge_type[trp_id] = 0

    kg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type, num_nodes=num_ents)
    write_obj(obj=kg_graph, obj_name="geometric graph", file_path=args.out_path + "kg-geo_x.pickle")


def prep_subgraph_cans() -> None:
    print("\n* preparing subgraphs and candidate targets ...")
    kg = read_obj(obj_name="the knowledge graph", file_path=args.out_path + "kg.pickle")
    kg_graph = read_obj(obj_name="geometric graph", file_path=args.out_path + "kg-geo_x.pickle")
    edge_index = kg_graph.edge_index.to(args.device)
    edge_attr = kg_graph.edge_attr.to(args.device)
    edge_type = kg_graph.edge_type.to(args.device)
    num_ents = kg["num_ents"]

    ent2subgsz = defaultdict(list)  # {entity id: [1hop_subgraph_size(int), 2hop_subgraph_size, 3hop_subgraph_size]}
    ent2subg = defaultdict(list)  # {entity id: [1hop_subgraph(torch_geometric.data.Data), 2hop_subgraph, 3hop_subgraph]}
    ent2cans = defaultdict(list)  # {entity id: [1hop_neighbors(LongTensor), 2hop_neighbors, 3hop_neighbors]}

    for ent_idx in tqdm(range(num_ents)):
        # 1hop_triple_ids and 1hop_neighbors
        fhop_edge_ids, fhop_neis = find_neighbors(cen_ent=ent_idx, edge_index=edge_index)
        ent2subgsz[ent_idx].append(fhop_edge_ids.size(0))
        ent2subg[ent_idx].append(subg_con(nei_edge_ids=fhop_edge_ids, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type, num_ents=fhop_neis.size(0)))
        ent2cans[ent_idx].append(fhop_neis.cpu())

        # 2hop_triple_ids and 2hop_neighbors
        shop_edge_ids = [fhop_edge_ids]
        shop_neis = [fhop_neis]
        for ent in fhop_neis:
            tmp_edge_ids, tmp_neis = find_neighbors(cen_ent=ent, edge_index=edge_index)
            shop_edge_ids.append(tmp_edge_ids)
            shop_neis.append(tmp_neis)
        shop_edge_ids = torch.unique(torch.cat(shop_edge_ids, dim=0))
        shop_neis = torch.unique(torch.cat(shop_neis, dim=0))
        ent2subgsz[ent_idx].append(shop_edge_ids.size(0))
        ent2subg[ent_idx].append(subg_con(nei_edge_ids=shop_edge_ids, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type, num_ents=shop_neis.size(0)))
        ent2cans[ent_idx].append(shop_neis.cpu())

        # 3hop_triple_ids and 3hop_neighbors
        thop_edge_ids = [shop_edge_ids]
        thop_neis = [shop_neis]
        for ent in shop_neis:
            tmp_edge_ids, tmp_neis = find_neighbors(cen_ent=ent, edge_index=edge_index)
            thop_edge_ids.append(tmp_edge_ids)
            thop_neis.append(tmp_neis)
        thop_edge_ids = torch.unique(torch.cat(thop_edge_ids, dim=0))
        thop_neis = torch.unique(torch.cat(thop_neis, dim=0))
        ent2subgsz[ent_idx].append(thop_edge_ids.size(0))
        ent2subg[ent_idx].append(subg_con(nei_edge_ids=thop_edge_ids, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type, num_ents=thop_neis.size(0)))
        ent2cans[ent_idx].append(thop_neis.cpu())
    write_obj(obj=[ent2subgsz, ent2subg, ent2cans], obj_name="entity subgraphs and candidate answers", file_path=args.out_path + "kg-subg-cans.pickle")


def find_neighbors(cen_ent: int, edge_index: LongTensor) -> list:
    nei_edge_ids = torch.nonzero(edge_index == cen_ent)[:, 1]  # ids of edges containing the center entity, size: (num_neighbor_edges,)
    nei_edge_index = torch.index_select(input=edge_index, index=nei_edge_ids, dim=1)  # size: (num_neighbor_edges, 2)
    neighbors = torch.unique(torch.cat([nei_edge_index[0, :], nei_edge_index[1, :]], dim=0))  # center entity and its direct neighbors
    return nei_edge_ids, neighbors


def subg_con(nei_edge_ids: LongTensor, edge_index: LongTensor, edge_attr: LongTensor, edge_type: LongTensor, num_ents: int) -> Data:
    nei_edge_index = torch.index_select(input=edge_index, index=nei_edge_ids, dim=1)  # size: (num_neighbor_edges, 2)
    nei_edge_attr = torch.index_select(input=edge_attr, index=nei_edge_ids, dim=0)  # size: (num_neighbor_edges,)
    nei_edge_type = torch.index_select(input=edge_type, index=nei_edge_ids, dim=0)  # # size: (num_neighbor_edges,)
    subgraph = Data(edge_index=nei_edge_index.cpu(), edge_attr=nei_edge_attr.cpu(), edge_type=nei_edge_type.cpu(), num_nodes=num_ents)
    return subgraph


def prep_ent_rel_embeds() -> None:
    print("\n* preparing entity and relation embeddings")
    kg = read_obj(obj_name="the knowledge graph", file_path=args.out_path + "kg.pickle")
    num_ents = kg["num_ents"]
    ent2id = kg["ent2id"]
    num_rels = kg["num_rels"]
    rel2id = kg["rel2id"]

    id2ent = {}
    for key, value in ent2id.items():
        id2ent[value] = key
    id2rel = {}
    for key, value in rel2id.items():
        id2rel[value] = key

    ent_embeds = []
    rel_embeds = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(args.device)
    model.eval()
    with torch.no_grad():
        ent_labels = []
        for ent_idx in range(num_ents):
            ent_labels.append(id2ent[ent_idx])
            if len(ent_labels) == 5000 or ent_idx == num_ents - 1:
                ent_embeds.append(bert_encode(tokenizer=tokenizer, model=model, batch_str=ent_labels, device=args.device))
                ent_labels = []
        ent_embeds = torch.cat(ent_embeds, dim=0).cpu()

        rel_labels = []
        for rel_idx in range(num_rels):
            ori_label = id2rel[rel_idx].split("_")
            fil_label = []
            for word in ori_label:
                if word not in stop_words:
                    fil_label.append(word)
            if len(fil_label) == 0:
                print("\t\tfound a relation label with only stop words: {}, id: {}".format(ori_label, rel_idx))
                fil_label = ori_label
            rel_labels.append(" ".join(fil_label))
            if len(rel_labels) == 5000 or rel_idx == num_rels - 1:
                rel_embeds.append(bert_encode(tokenizer=tokenizer, model=model, batch_str=rel_labels, device=args.device))
                rel_labels = []
        rel_embeds = torch.cat(rel_embeds, dim=0).cpu()

    write_obj(obj=ent_embeds, obj_name="entity embeddings", file_path=args.out_path + "ent_label_embeds.pickle")
    write_obj(obj=rel_embeds, obj_name="relation embeddings", file_path=args.out_path + "rel_label_embeds.pickle")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--in_path', type=str, default='../data/metaqa/input/', help='input data path')
    args.add_argument('--out_path', type=str, default='../data/metaqa/output/', help='output data path')
    args.add_argument('--device', type=int, default=0)
    args = args.parse_args()

    prep_kg()
    prep_geo_graph()
    prep_subgraph_cans()
    prep_ent_rel_embeds()
