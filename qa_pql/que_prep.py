import os
import torch
import graph_tool
import random
from graph_tool import topology
from argparse import ArgumentParser
from tqdm import tqdm
from itertools import product
from collections import defaultdict
from torch.utils.data import random_split
from transformers import BertTokenizer, BertModel
from utils import write_obj, read_obj, stop_words, bert_encode


def read_qa():
    ent2id = read_obj(obj_name="{}-hop knowledge graph".format(hop),
                      file_path=os.path.join(args.out_path, "{}H-kg.pickle".format(hop)))['ent2id']

    with open(file=os.path.join(args.in_path, 'PQL-{}H.txt'.format(hop)), mode='r') as f:
        lines = f.readlines()
        qid2que = {}
        num_ques = 0
        for line in tqdm(lines):
            line = line.strip()
            que, ans, path = line.split('\t')

            first_ans = ans.replace('_(', '\t').split('(')[0].replace('\t', '_(')

            all_ans = [first_ans]
            for tmp_ans in ans.replace(first_ans, '', 1)[1:-1].split('/'):
                if tmp_ans not in all_ans and tmp_ans != '':
                    all_ans.append(tmp_ans)
            all_ans = [ent2id[_] for _ in all_ans]

            top_ent = ent2id[path.split('#')[0]]

            qid2que[num_ques] = {
                'question': que,
                'top_ents': top_ent,
                'glo_ans': all_ans,
                'path': path,
            }

            num_ques += 1
        print('\t* #total questions: {}'.format(num_ques))

        train_ids, test_ids, dev_ids = [list(_) for _ in torch.utils.data.random_split(list(qid2que.keys()), args.split)]

        for mode, ids in zip(['train', 'valid', 'test'], [train_ids, dev_ids, test_ids]):
            # print('\t\t* #{}: {}'.format(mode, len(ids)))

            tmp_que_count = 0
            tmp_qid2que = {}

            for idx in ids:
                tmp_qid2que[tmp_que_count] = qid2que[idx]
                tmp_que_count += 1

            que2id, qid2tops, qid2erps, qid2ans = {}, {}, {}, {}
            for qid, q_data in tmp_qid2que.items():
                que2id[q_data['question']] = qid
                qid2tops[qid] = q_data['top_ents']
                qid2erps[qid] = []
                for word in q_data['question'].split(" "):
                    if word not in stop_words and word != q_data['path'].split('#')[0] and word != '':
                        qid2erps[qid].append(word)
                qid2ans[qid] = torch.LongTensor(q_data['glo_ans'])
            ques = [que2id, qid2tops, qid2erps, qid2ans]
            print("\t* number of {} questions: {}".format(mode, len(ques[0])))
            write_obj(obj=ques, obj_name="preprocessed {}-hop {} questions".format(hop, mode),
                      file_path=os.path.join(args.out_path, "{}H-{}_ques.pickle".format(hop, mode)))


def prep_que_embeds() -> None:
    print("\n* computing question embeddings...")
    for mode in ["train", "valid", "test"]:
        qid2erps = read_obj(obj_name="preprocessed {}-hop {} questions".format(hop, mode), file_path=os.path.join(args.out_path, "{}H-{}_ques.pickle".format(hop, mode)))[2]

        qid2embeds = {}
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(args.device)
        model.eval()
        with torch.no_grad():
            for qid, erps in qid2erps.items():
                qid2embeds[qid] = bert_encode(tokenizer=tokenizer, model=model, batch_str=erps, device=args.device).cpu()
        write_obj(obj=qid2embeds, obj_name="{}-hop {} question embeddings".format(hop, mode), file_path=os.path.join(args.out_path, "{}H-{}_qid2embeds.pickle".format(hop, mode)))


def qa_batch_prep():
    print("\n* preparing question batch data...")
    num_rels = read_obj(obj_name="{}-hop knowledge graph".format(hop), file_path=os.path.join(args.out_path, "{}H-kg.pickle".format(hop)))["num_rels"]
    ent2subgsz, ent2subg, ent2cans = read_obj(obj_name="{}-hop subgraphs and candidate answers".format(hop), file_path=os.path.join(args.out_path, "{}H-kg-subg-cans.pickle".format(hop)))
    for mode in ["train", "valid", "test"]:
        qid2batch = {}  # qid to batch_data
        que2id, qid2tops, qid2erps, qid2ans = read_obj(obj_name="preprocessed {}-hop {} questions".format(hop, mode), file_path=os.path.join(args.out_path, "{}H-{}_ques.pickle".format(hop, mode)))
        qid2embeds = read_obj(obj_name="{}-hop {} question embeddings".format(hop, mode), file_path=os.path.join(args.out_path, "{}H-{}_qid2embeds.pickle".format(hop, mode)))
        for qid in range(len(qid2tops)):
            que_embed = qid2embeds[qid]  # size: (num_er_phrases, bert_out_size)

            top_ent = qid2tops[qid]

            glo_ents = ent2cans[top_ent][hop - 1]  # size: (num_ents_in_subg,)
            glo2loc = {}
            loc_ent = 0
            for ent in glo_ents:
                glo2loc[int(ent)] = loc_ent
                loc_ent += 1

            subg = ent2subg[top_ent][hop - 1]  # torch_geometric.data.Data
            edge_attr = subg.edge_attr
            edge_type = subg.edge_type
            glo_sour_ents = subg.edge_index[0, :]
            glo_targ_ents = subg.edge_index[1, :]
            loc_sour_ents = torch.LongTensor([glo2loc[int(_)] for _ in glo_sour_ents])
            loc_targ_ents = torch.LongTensor([glo2loc[int(_)] for _ in glo_targ_ents])
            loc_edge_index = torch.stack([loc_sour_ents, loc_targ_ents], dim=0)

            # loc_edge_index = torch.cat([loc_edge_index, torch.stack([loc_targ_ents, loc_sour_ents], dim=0)], dim=1)
            # edge_attr = torch.cat([edge_attr, edge_attr + num_rels], dim=0)
            # edge_type = torch.cat([edge_type, torch.full_like(input=edge_type, fill_value=1)], dim=0)
            # loc_edge_index = torch.cat([loc_edge_index, torch.stack([torch.arange(loc_ent), torch.arange(loc_ent)], dim=0)], dim=1)
            # edge_attr = torch.cat([edge_attr, torch.full(size=[loc_ent], dtype=torch.long, fill_value=2 * num_rels)], dim=0)
            # edge_type = torch.cat([edge_type, torch.full(size=[loc_ent], dtype=torch.long, fill_value=2)], dim=0)

            glo_ans = qid2ans[qid]  # size: (num_ans,)
            loc_ans_dis = torch.full_like(input=glo_ents, fill_value=1, dtype=torch.float)  # size: (num_ents_in_subg,)
            for glo_an in glo_ans:
                if int(glo_an) not in glo2loc:
                    print("\t\tanswer {} is not in the subgraph of question {}".format(glo_an, qid))
                else:
                    loc_ans_dis[glo2loc[int(glo_an)]] = 0.

            loc_top = glo2loc[top_ent]

            qid2batch[qid] = [que_embed, glo_ents, loc_edge_index, edge_attr, edge_type, loc_ans_dis, loc_top]
        write_obj(obj=qid2batch, obj_name="{}-hop {} qa batch".format(hop, mode), file_path=os.path.join(args.out_path, "{}H-{}_qid2batch.pickle".format(hop, mode)))


def qa_shortest_path_prep() -> None:
    print("\n* preparing correct shortest paths between topic entities and candidate answers...")
    kg = read_obj(obj_name="{}-hop knowledge graph".format(hop), file_path=os.path.join(args.out_path, "{}H-kg.pickle".format(hop)))
    poss_rel_ids = []
    rel_id2str = {}
    rel2id = kg["rel2id"]
    for rel, idx in rel2id.items():
        rel = rel.split('__')[-1]
        poss_rel_ids.append(idx)
        fil_rel_lab = []
        for word in rel.split("_"):
            if word not in stop_words:
                fil_rel_lab.append(word)
        if len(fil_rel_lab) == 0:
            fil_rel_lab = rel.split(" ")
        rel_id2str[idx] = " ".join(fil_rel_lab)
    rel_id2str[-1] = "self-loop"

    id_paths = [(-1,)]
    for tmp_hop in range(hop):
        id_paths += list(product(*[poss_rel_ids for _ in range(tmp_hop + 1)]))

    str_paths = [[rel_id2str[idx] for idx in id_path] for id_path in id_paths]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(args.device)
    model.eval()
    str_path_embeds = []
    with torch.no_grad():
        for str_path in tqdm(str_paths):
            if str_path[0] == "self-loop":
                tmp_embed = torch.zeros(hop, 768)  # 768 is the out size of the used bert
            else:
                tmp_embed = bert_encode(tokenizer=tokenizer, model=model, batch_str=str_path, device=args.device).cpu()
                tmp_embed = torch.nn.functional.pad(tmp_embed, pad=(0, 0, 0, hop - tmp_embed.size(0)))  # size: (hop, bert_out_size)
            str_path_embeds.append(tmp_embed)
    str_path_embeds = torch.stack(str_path_embeds, dim=0)  # size: (num_poss_paths, hop, bert_out_size)
    write_obj(obj=[id_paths, str_paths, str_path_embeds], obj_name="{}-hop ids, strings, and embeddings of possible shortest paths".format(hop), file_path=os.path.join(args.out_path, "{}H-poss_paths.pickle".format(hop)))

    gt_graph = graph_tool.Graph(directed=True)
    eprop = gt_graph.new_edge_property('int')
    trp2id = kg["trp2id"]
    for trp in trp2id.keys():
        h, r, t = trp
        eprop[gt_graph.add_edge(h, t)] = r
    gt_graph.edge_properties['edge_attr'] = eprop
    write_obj(obj=gt_graph, obj_name="{}-hop graph_tool graph".format(hop), file_path=os.path.join(args.out_path, "{}H-nx_kg.pickle".format(hop)))

    for mode in ["train", "valid", "test"]:
        ques = read_obj(obj_name="preprocessed {}-hop {} questions".format(hop, mode), file_path=os.path.join(args.out_path, "{}H-{}_ques.pickle".format(hop, mode)))
        qid2tops = ques[1]
        qid2ans = ques[3]
        qid2path_dis = {}
        for qid, ans in tqdm(qid2ans.items()):
            top = qid2tops[qid]
            # path2occur = defaultdict(int)
            # for an in ans:
            #     if an == top:
            #         path2occur[(-1,)] += 1
            #     else:
            #         for id_path in find_shortest_paths(nx_kg=nx_kg, source=int(an), target=top, hop=hop):
            #             if hop == 2:
            #                 path2occur[id_path] += 1
            #             else:
            #                 # if id_path[1:] not in [(1, 1), (8, 8), (9, 9), (10, 10), (12, 12)] and id_path[:2] not in [(1, 1), (8, 8), (9, 9), (10, 10), (12, 12)]:
            #                 #     path2occur[id_path] += 1
            #                 path2occur[id_path] += 1
            path2occur = defaultdict(int)
            for an in ans:
                if an == top:
                    path2occur[(-1,)] += 1
                else:
                    for id_path in extract_paths(top=top, ans=an, gt_graph=gt_graph, cutoff=hop):
                        path2occur[id_path] += 1
            highest_occur = 0
            for _, occur in path2occur.items():
                highest_occur = occur if highest_occur < occur else highest_occur
            corr_paths = []
            for id_path, occur in path2occur.items():
                if occur == highest_occur:
                    corr_paths.append(id_path)
            if len(corr_paths) == 0:
                print("\t\t* found no shortest paths for question {}".format(qid))
            path_dis = torch.FloatTensor([0 if id_path in corr_paths else 1 for id_path in id_paths])
            qid2path_dis[qid] = path_dis
        write_obj(obj=qid2path_dis, obj_name="{}-hop question id to shortest path distance".format(hop), file_path=os.path.join(args.out_path, "{}H-{}_qid2path_dis.pickle".format(hop, mode)))


def extract_paths(top: int, ans: int, gt_graph: graph_tool.Graph, cutoff: int):
    all_rel_paths = []  # all unique reasoning paths for a topic entity

    for edge_path in topology.all_paths(g=gt_graph, source=gt_graph.vertex(top), target=gt_graph.vertex(ans), cutoff=cutoff, edges=True):
            tmp_rel_path = []  # one reasoning path between a topic entity and the given answer

            for edge in edge_path:
                tmp_rel_path.append(gt_graph.edge_properties['edge_attr'][edge])

            tmp_rel_path = tuple(tmp_rel_path)  # each candidate path is a tuple of relations + topic entity

            if tmp_rel_path not in all_rel_paths:
                all_rel_paths.append(tmp_rel_path)

    return all_rel_paths


def qa_path_prep() -> None:
    print("\n* preparing correct shortest paths between topic entities and candidate answers...")
    kg = read_obj(obj_name="{}-hop knowledge graph".format(hop), file_path=args.out_path + "/{}H-kg.pickle".format(hop))

    poss_rel_ids = []
    rel_id2str = {}
    rel2id = kg["rel2id"]
    for rel, idx in rel2id.items():
        rel = rel.split('__')[-1]
        poss_rel_ids.append(idx)
        fil_rel_lab = []
        for word in rel.split("_"):
            if word not in stop_words:
                fil_rel_lab.append(word)
        if len(fil_rel_lab) == 0:
            fil_rel_lab = rel.split(" ")
        rel_id2str[idx] = " ".join(fil_rel_lab)
    rel_id2str[-1] = "self-loop"

    id_paths = [(-1,)]
    for tmp_hop in range(hop):
        id_paths += list(product(*[poss_rel_ids for _ in range(tmp_hop + 1)]))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(args.device)
    model.eval()

    ent2subgsz, ent2subg, ent2cans = read_obj(obj_name="{}-hop subgraphs and candidate answers".format(hop),
                                              file_path=os.path.join(args.out_path, "{}H-kg-subg-cans.pickle".format(hop)))

    for mode in ["train", "valid", "test"]:

        qid2batch = read_obj(obj_name="{}-hop {} qa batch".format(hop, mode),
                             file_path=os.path.join(args.out_path, "{}H-{}_qid2batch.pickle".format(hop, mode)))

        ques = read_obj(obj_name="preprocessed {}-hop {} questions".format(hop, mode), file_path=os.path.join(args.out_path, "{}H-{}_ques.pickle".format(hop, mode)))
        qid2tops = ques[1]
        qid2ans = ques[3]

        qid2path_dis = {}
        for qid, ans in tqdm(qid2ans.items()):

            rand_ids = torch.randperm(len(id_paths))[:args.num_poss_paths]
            q_id_paths = [id_paths[int(_)] for _ in rand_ids]

            top = qid2tops[qid]

            subg = ent2subg[top][hop - 1]  # torch_geometric.data.Data
            edge_attr = subg.edge_attr
            glo_sour_ents = subg.edge_index[0, :]
            glo_targ_ents = subg.edge_index[1, :]

            gt_graph = graph_tool.Graph(directed=True)
            eprop = gt_graph.new_edge_property('int')
            for h, r, t in zip(glo_sour_ents, edge_attr, glo_targ_ents):
                eprop[gt_graph.add_edge(int(h), int(t))] = int(r)
            gt_graph.edge_properties['edge_attr'] = eprop

            # path2occur = defaultdict(int)
            # for an in ans:
            #     if an == top:
            #         path2occur[(-1,)] += 1
            #     else:
            #         for id_path in find_shortest_paths(nx_kg=nx_kg, source=int(an), target=top, hop=hop):
            #             if hop == 2:
            #                 path2occur[id_path] += 1
            #             else:
            #                 # if id_path[1:] not in [(1, 1), (8, 8), (9, 9), (10, 10), (12, 12)] and id_path[:2] not in [(1, 1), (8, 8), (9, 9), (10, 10), (12, 12)]:
            #                 #     path2occur[id_path] += 1
            #                 path2occur[id_path] += 1
            path2occur = defaultdict(int)
            for an in ans:
                if an == top:
                    path2occur[(-1,)] += 1
                else:
                    for id_path in extract_paths(top=top, ans=an, gt_graph=gt_graph, cutoff=hop):
                        path2occur[id_path] += 1
            for path in path2occur:
                if path not in q_id_paths:
                    q_id_paths.append(path)
            highest_occur = 0
            for _, occur in path2occur.items():
                highest_occur = occur if highest_occur < occur else highest_occur
            corr_paths = []
            for id_path, occur in path2occur.items():
                if occur == highest_occur:
                    corr_paths.append(id_path)
            if len(corr_paths) == 0:
                print("\t\t* found no paths for question {}".format(qid))

            str_paths = [[rel_id2str[idx] for idx in id_path] for id_path in q_id_paths]

            str_path_embeds = []
            with torch.no_grad():
                for str_path in str_paths:
                    if str_path[0] == "self-loop":
                        tmp_embed = torch.zeros(hop, 768)  # 768 is the out size of the used bert
                    else:
                        tmp_embed = bert_encode(tokenizer=tokenizer, model=model, batch_str=str_path,
                                                device=args.device).cpu()
                        tmp_embed = torch.nn.functional.pad(tmp_embed, pad=(
                            0, 0, 0, hop - tmp_embed.size(0)))  # size: (hop, bert_out_size)
                    str_path_embeds.append(tmp_embed)
            str_path_embeds = torch.stack(str_path_embeds, dim=0)  # size: (num_poss_paths, hop, bert_out_size)

            path_dis = torch.FloatTensor([0 if id_path in corr_paths else 1 for id_path in q_id_paths])

            qid2path_dis[qid] = [id_paths, str_paths, str_path_embeds, path_dis]

        write_obj(obj=qid2path_dis, obj_name="{}-hop question id to shortest path distance".format(hop),
                  file_path=os.path.join(args.out_path, "{}H-{}_qid2path_dis.pickle".format(hop, mode)))


if __name__ == "__main__":

    args = ArgumentParser()
    args.add_argument("--qa_type", type=str, default="2-hop", choices={'2-hop', '3-hop'})
    args.add_argument('--in_path', type=str, default='../data/pql-{}hop/input/', help='input data path')
    args.add_argument('--out_path', type=str, default='../data/pql-{}hop/output/', help='output data path')
    args.add_argument('--device', type=int, default=0)
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--split', type=float, default=[0.8, 0.1, 0.1], nargs='+')
    args.add_argument('--num_poss_paths', type=int, default=512)
    args = args.parse_args()

    args.in_path = args.in_path.format(args.qa_type[0])
    args.out_path = args.out_path.format(args.qa_type[0])
    hop = int(args.qa_type[0])

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    read_qa()
    prep_que_embeds()
    qa_batch_prep()
    if args.qa_type == '2-hop':
        qa_shortest_path_prep()
    else:
        qa_path_prep()
