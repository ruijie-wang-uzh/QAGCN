import os
import re
import torch
from tqdm import tqdm
import networkx as nx
from itertools import product
from collections import defaultdict
from transformers import BertTokenizer, BertModel
from utils import write_obj, read_obj, stop_words, bert_encode
from argparse import ArgumentParser


def qa_prep() -> None:
    print("\n* preparing {} questions...".format(args.qa_type))
    kg = read_obj(obj_name="the knowledge graph", file_path=args.out_path + "kg.pickle")
    ent2id = kg["ent2id"]
    for mode in ["train", "dev", "test"]:
        ques = read_qa(mode=mode, in_path=args.in_path + args.qa_type + "/qa_{}.txt".format(mode), ent2id=ent2id)
        if mode == "dev":
            mode = "valid"
        if not os.path.exists(args.out_path + args.qa_type):
            os.makedirs(args.out_path + args.qa_type)
        write_obj(obj=ques, obj_name="preprocessed {} questions".format(mode), file_path=args.out_path + args.qa_type + "/{}_ques.pickle".format(mode))


def read_qa(mode: str, in_path: str, ent2id: dict) -> list:
    num_ques = 0
    que2id = {}  # question strings to ids (int)
    qid2tops = {}  # question ids to topic entity ids (int)
    qid2erps = {}  # question ids to entity/relation phrases (list of strings)
    qid2ans = {}  # question ids to answers (LongTensor)
    with open(file=in_path, mode="r") as f:
        line = f.readline()
        while line:

            elems = line.rstrip().split("\t")
            question = elems[0].replace("[", "").replace("]", "")
            if question not in que2id:
                que2id[question] = num_ques
                num_ques += 1
            qid = que2id[question]

            pattern = r"\[.*\]"
            it = re.finditer(pattern, elems[0])
            for match in it:
                topic = match.group()[1:-1]
                assert topic in ent2id, "found an unindexed topic entity!"
                qid2tops[qid] = ent2id[topic]  # there is only one topic entity for each question

            qid2erps[qid] = []
            for word in elems[0].split(" "):
                add_flag = True
                if word not in stop_words:
                    if len(qid2erps[qid]) > 0:
                        if qid2erps[qid][-1][0] == "[":
                            add_flag = False
                    if add_flag:
                        qid2erps[qid].append(word)
                    else:
                        qid2erps[qid][-1] += (" " + word)
                    if word[-1] == "]":
                        qid2erps[qid][-1] = qid2erps[qid][-1][1:-1]

            tmp_ans = qid2ans[qid].tolist() if qid in qid2ans else []
            all_answers = elems[1].split("|")
            for answer in all_answers:
                assert answer in ent2id, "found an unseen entity {} from question {} in {} set!".format(answer, question, mode)
                tmp_ans.append(ent2id[answer])
            qid2ans[qid] = torch.LongTensor(tmp_ans)

            line = f.readline()
    return [que2id, qid2tops, qid2erps, qid2ans]


def prep_que_embeds() -> None:
    print("\n* preparing question embeddings...")
    for mode in ["train", "valid", "test"]:
        qid2erps = read_obj(obj_name="preprocessed {} questions".format(mode), file_path=args.out_path + args.qa_type + "/{}_ques.pickle".format(mode))[2]

        qid2embeds = {}
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(args.device)
        model.eval()
        with torch.no_grad():
            for qid, erps in tqdm(qid2erps.items()):
                qid2embeds[qid] = bert_encode(tokenizer=tokenizer, model=model, batch_str=erps, device=args.device).cpu()
        write_obj(obj=qid2embeds, obj_name="{} question embeddings".format(mode), file_path=args.out_path + args.qa_type + "/{}_qid2embeds.pickle".format(mode))


def qa_batch_prep() -> None:
    print("\n* preparing question batch data...")
    hop = int(args.qa_type[0]) - 1
    num_rels = read_obj(obj_name="the knowledge graph", file_path=args.out_path + "kg.pickle")["num_rels"]
    ent2subgsz, ent2subg, ent2cans = read_obj(obj_name="entity subgraphs and candidate answers", file_path=args.out_path + "kg-subg-cans.pickle")
    for mode in ["train", "valid", "test"]:
        qid2batch = {}  # qid to batch_data
        que2id, qid2tops, qid2erps, qid2ans = read_obj(obj_name="preprocessed {} questions".format(mode), file_path=args.out_path + args.qa_type + "/{}_ques.pickle".format(mode))
        qid2embeds = read_obj(obj_name="{} question embeddings".format(mode), file_path=args.out_path + args.qa_type + "/{}_qid2embeds.pickle".format(mode))
        for qid in tqdm(range(len(qid2tops))):

            que_embed = qid2embeds[qid]  # size: (num_er_phrases, bert_out_size)

            top_ent = qid2tops[qid]

            glo_ents = ent2cans[top_ent][hop]  # size: (num_ents_in_subg,)
            glo2loc = {}
            loc_ent = 0
            for ent in glo_ents:
                glo2loc[int(ent)] = loc_ent
                loc_ent += 1

            subg = ent2subg[top_ent][hop]  # torch_geometric.data.Data
            edge_attr = subg.edge_attr
            edge_type = subg.edge_type
            glo_sour_ents = subg.edge_index[0, :]
            glo_targ_ents = subg.edge_index[1, :]
            loc_sour_ents = torch.LongTensor([glo2loc[int(_)] for _ in glo_sour_ents])
            loc_targ_ents = torch.LongTensor([glo2loc[int(_)] for _ in glo_targ_ents])
            loc_edge_index = torch.stack([loc_sour_ents, loc_targ_ents], dim=0)

            loc_edge_index = torch.cat([loc_edge_index, torch.stack([loc_targ_ents, loc_sour_ents], dim=0)], dim=1)
            edge_attr = torch.cat([edge_attr, edge_attr + num_rels], dim=0)
            edge_type = torch.cat([edge_type, torch.full_like(input=edge_type, fill_value=1)], dim=0)
            loc_edge_index = torch.cat([loc_edge_index, torch.stack([torch.arange(loc_ent), torch.arange(loc_ent)], dim=0)], dim=1)
            edge_attr = torch.cat([edge_attr, torch.full(size=[loc_ent], dtype=torch.long, fill_value=2 * num_rels)], dim=0)
            edge_type = torch.cat([edge_type, torch.full(size=[loc_ent], dtype=torch.long, fill_value=2)], dim=0)

            glo_ans = qid2ans[qid]  # size: (num_ans,)
            loc_ans_dis = torch.full_like(input=glo_ents, fill_value=1, dtype=torch.float)  # size: (num_ents_in_subg,)
            for glo_an in glo_ans:
                if int(glo_an) not in glo2loc:
                    print("\t\tanswer {} is not in the subgraph of question {}".format(glo_an, qid))
                else:
                    loc_ans_dis[glo2loc[int(glo_an)]] = 0.

            qid2batch[qid] = [que_embed, glo_ents, loc_edge_index, edge_attr, edge_type, loc_ans_dis]
        write_obj(obj=qid2batch, obj_name="{} qa batch".format(mode), file_path=args.out_path + args.qa_type + "/{}_qid2{}batch.pickle".format(mode, 'real_' if args.qa_type[0] == '3' else ''))


def qa_batch_large_prep() -> None:
    print("\n* preparing question batch data (using the complete KG)...")
    num_ents = read_obj(obj_name="the knowledge graph", file_path=args.out_path + "kg.pickle")["num_ents"]
    for mode in ["train", "valid", "test"]:
        _, _, _, qid2ans = read_obj(obj_name="preprocessed {} questions".format(mode), file_path=args.out_path + args.qa_type + "/{}_ques.pickle".format(mode))
        qid2embeds = read_obj(obj_name="{} question embeddings".format(mode), file_path=args.out_path + args.qa_type + "/{}_qid2embeds.pickle".format(mode))
        qid2batch = {}  # qid to batch_data
        for qid in range(len(qid2ans)):
            que_embed = qid2embeds[qid]  # size: (num_er_phrases, bert_out_size)

            glo_ents = torch.arange(num_ents)  # size: (num_ents_in_kg,)

            glo_ans = qid2ans[qid]  # size: (num_ans,)
            loc_ans_dis = torch.full_like(input=glo_ents, fill_value=1, dtype=torch.float)  # size: (num_ents_in_subg,)
            for glo_an in glo_ans:
                loc_ans_dis[int(glo_an)] = 0.

            qid2batch[qid] = [que_embed, glo_ents, None, None, None, loc_ans_dis]
        write_obj(obj=qid2batch, obj_name="{} qa batch".format(mode), file_path=args.out_path + args.qa_type + "/{}_qid2batch.pickle".format(mode))


def qa_shortest_path_prep() -> None:
    print("\n* preparing correct shortest paths between topic entities and candidate answers...")
    hop = int(args.qa_type[0])
    kg = read_obj(obj_name="the knowledge graph", file_path=args.out_path + "kg.pickle")
    poss_rel_ids = []
    rel_id2str = {}
    rel2id = kg["rel2id"]
    for rel, idx in rel2id.items():
        poss_rel_ids.append(idx)
        fil_rel_lab = []
        for word in rel.split("_"):
            if word not in stop_words:
                fil_rel_lab.append(word)
        if len(fil_rel_lab) == 0:
            fil_rel_lab = rel.split(" ")
        rel_id2str[idx] = " ".join(fil_rel_lab)
    id_paths = list(product(*[poss_rel_ids for _ in range(hop)]))
    str_paths = [[rel_id2str[idx] for idx in id_path] for id_path in id_paths]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(args.device)
    model.eval()
    str_path_embeds = []
    with torch.no_grad():
        for str_path in str_paths:
            str_path_embeds.append(bert_encode(tokenizer=tokenizer, model=model, batch_str=str_path, device=args.device).cpu())  # size: (hop, bert_out_size)
    str_path_embeds = torch.stack(str_path_embeds, dim=0)  # size: (num_poss_paths, hop, bert_out_size)
    write_obj(obj=[id_paths, str_paths, str_path_embeds], obj_name="ids, strings, and embeddings of possible shortest paths", file_path=args.out_path + args.qa_type + "/poss_paths.pickle")

    ent2id = kg["ent2id"]
    id2ent = {}
    for k, v in ent2id.items():
        id2ent[v] = k
    nx_kg = nx.MultiGraph()
    nx_kg.add_nodes_from(id2ent.keys())
    trp2id = kg["trp2id"]
    for trp, _ in trp2id.items():
        nx_kg.add_edge(trp[0], trp[2], rel=trp[1])
    write_obj(obj=nx_kg, obj_name="the networkx graph", file_path=args.out_path + "nx_kg.pickle")

    for mode in ["train", "valid", "test"]:
        ques = read_obj(obj_name="preprocessed {} questions".format(mode), file_path=args.out_path + args.qa_type + "/{}_ques.pickle".format(mode))
        qid2tops = ques[1]
        qid2ans = ques[3]
        qid2path_dis = {}
        for qid, ans in tqdm(qid2ans.items()):

            top = qid2tops[qid]
            path2occur = defaultdict(int)
            for an in ans:
                for id_path in find_shortest_paths(nx_kg=nx_kg, source=int(an), target=top, hop=hop):
                    path2occur[id_path] += 1
            highest_occur = 0
            for _, occur in path2occur.items():
                if highest_occur < occur:
                    highest_occur = occur
            corr_paths = []
            for id_path, occur in path2occur.items():
                if occur == highest_occur:
                    corr_paths.append(id_path)
            if len(corr_paths) == 0:
                print("\t\t* found no shortest paths for question {}".format(qid))
            path_dis = torch.FloatTensor([0 if id_path in corr_paths else 1 for id_path in id_paths])
            qid2path_dis[qid] = path_dis
        write_obj(obj=qid2path_dis, obj_name="question id to shortest path distance", file_path=args.out_path + args.qa_type + "/{}_qid2path_dis.pickle".format(mode))


def find_shortest_paths(nx_kg: nx.MultiGraph, source: int, target: int, hop: int) -> list:
    all_paths = nx.all_simple_paths(G=nx_kg, source=source, target=target, cutoff=hop)
    all_paths = [p for p in all_paths]
    all_id_paths = set()
    for path in all_paths:
        if len(path) == hop + 1:
            id_paths = []
            for step in range(hop):
                poss_rel_ids = []
                for _, attr in dict(nx_kg[path[step]][path[step + 1]]).items():
                    poss_rel = attr["rel"]
                    poss_rel_ids.append(poss_rel)
                id_paths.append(poss_rel_ids)
            id_paths = list(product(*id_paths))
            for id_path in id_paths:
                all_id_paths.add(id_path)
    return all_id_paths


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--qa_type", type=str, default="1-hop", choices={'1-hop', '2-hop', '3-hop'})
    args.add_argument("--in_path", type=str, default="../data/metaqa/input/")
    args.add_argument("--out_path", type=str, default="../data/metaqa/output/")
    args.add_argument('--device', type=int, default=0)
    args = args.parse_args()

    qa_prep()
    prep_que_embeds()
    qa_batch_prep()
    qa_shortest_path_prep()
