import torch
from argparse import ArgumentParser
import networkx as nx
from itertools import product
from collections import defaultdict
from torch.utils.data import random_split
from transformers import BertTokenizer, BertModel
from utils import write_obj, read_obj, stop_words, bert_encode


def split(seed) -> None:
    print("\n\n* randomly split the data into train, valid, and test subsets...")
    print("\t* seed: {}".format(seed))
    with open(args.in_path + args.qa_type + "/PQ-{}H.txt".format(hop), "r") as f:
        all_ques = f.readlines()
    num_ques = len(all_ques)
    num_train = int(num_ques * 0.8)
    num_valid = int(num_ques * 0.1)
    num_test = num_ques - num_train - num_valid
    train_set, valid_set, test_set = random_split(dataset=all_ques, lengths=[num_train, num_valid, num_test], generator=torch.Generator().manual_seed(seed))
    with open(args.out_path + args.qa_type + "/{}H-{}.txt".format(hop, "train"), "w") as f:
        f.writelines(list(train_set))
    with open(args.out_path + args.qa_type + "/{}H-{}.txt".format(hop, "valid"), "w") as f:
        f.writelines(list(valid_set))
    with open(args.out_path + args.qa_type + "/{}H-{}.txt".format(hop, "test"), "w") as f:
        f.writelines(list(test_set))


def qa_prep() -> None:
    print("\n* preparing {}-hop questions...".format(hop))
    kg = read_obj(obj_name="{}-hop knowledge graph".format(hop), file_path=args.out_path + args.qa_type + "/{}H-kg.pickle".format(hop))
    ent2id = kg["ent2id"]
    for mode in ["train", "valid", "test"]:
        ques = read_qa(mode=mode, in_path=args.out_path + args.qa_type + "/{}H-{}.txt".format(hop, mode), ent2id=ent2id)
        print("\t* number of {} questions: {}".format(mode, len(ques[0])))
        write_obj(obj=ques, obj_name="preprocessed {}-hop {} questions".format(hop, mode), file_path=args.out_path + args.qa_type + "/{}H-{}_ques.pickle".format(hop, mode))


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
            question = elems[0]
            if question not in que2id:
                que2id[question] = num_ques
                num_ques += 1
            qid = que2id[question]

            top = elems[2].split("#")[0]
            assert top in ent2id, "found an unindexed topic entity!"
            qid2tops[qid] = ent2id[top]

            qid2erps[qid] = []
            top_flag = False
            for word in elems[0].split(" "):
                if word not in stop_words:
                    if word == top:
                        top_flag = True
                        word = word.replace("_", " ")
                    qid2erps[qid].append(word)
            assert top_flag, "found no topic entity for question {}".format(qid)

            tmp_ans = qid2ans[qid].tolist() if qid in qid2ans else []
            all_answers = set()
            all_answers.add(elems[1].split("(")[0])
            for answer in elems[1].split("(")[1].split("/"):
                if answer != ")":
                    all_answers.add(answer)
            for answer in all_answers:
                assert answer in ent2id, "found an unseen entity {} from question {} in {} set!".format(answer, question, mode)
                tmp_ans.append(ent2id[answer])
            qid2ans[qid] = torch.LongTensor(tmp_ans)

            line = f.readline()
    return [que2id, qid2tops, qid2erps, qid2ans]


def prep_que_embeds() -> None:
    print("\n* computing question embeddings...")
    for mode in ["train", "valid", "test"]:
        qid2erps = read_obj(obj_name="preprocessed {}-hop {} questions".format(hop, mode), file_path=args.out_path + args.qa_type + "/{}H-{}_ques.pickle".format(hop, mode))[2]

        qid2embeds = {}
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(args.device)
        model.eval()
        with torch.no_grad():
            for qid, erps in qid2erps.items():
                qid2embeds[qid] = bert_encode(tokenizer=tokenizer, model=model, batch_str=erps, device=args.device).cpu()
        write_obj(obj=qid2embeds, obj_name="{}-hop {} question embeddings".format(hop, mode), file_path=args.out_path + args.qa_type + "/{}H-{}_qid2embeds.pickle".format(hop, mode))


def qa_batch_prep():
    print("\n* preparing question batch data...")
    num_rels = read_obj(obj_name="{}-hop knowledge graph".format(hop), file_path=args.out_path + args.qa_type + "/{}H-kg.pickle".format(hop))["num_rels"]
    ent2subgsz, ent2subg, ent2cans = read_obj(obj_name="{}-hop subgraphs and candidate answers".format(hop), file_path=args.out_path + args.qa_type + "/{}H-kg-subg-cans.pickle".format(hop))
    for mode in ["train", "valid", "test"]:
        qid2batch = {}  # qid to batch_data
        que2id, qid2tops, qid2erps, qid2ans = read_obj(obj_name="preprocessed {}-hop {} questions".format(hop, mode), file_path=args.out_path + args.qa_type + "/{}H-{}_ques.pickle".format(hop, mode))
        qid2embeds = read_obj(obj_name="{}-hop {} question embeddings".format(hop, mode), file_path=args.out_path + args.qa_type + "/{}H-{}_qid2embeds.pickle".format(hop, mode))
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
        write_obj(obj=qid2batch, obj_name="{}-hop {} qa batch".format(hop, mode), file_path=args.out_path + args.qa_type + "/{}H-{}_qid2batch.pickle".format(hop, mode))


def qa_shortest_path_prep() -> None:
    print("\n* preparing correct shortest paths between topic entities and candidate answers...")
    kg = read_obj(obj_name="{}-hop knowledge graph".format(hop), file_path=args.out_path + args.qa_type + "/{}H-kg.pickle".format(hop))
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
        for str_path in str_paths:
            if str_path[0] == "self-loop":
                tmp_embed = torch.zeros(hop, 768)  # 768 is the out size of the used bert
            else:
                tmp_embed = bert_encode(tokenizer=tokenizer, model=model, batch_str=str_path, device=args.device).cpu()
                tmp_embed = torch.nn.functional.pad(tmp_embed, pad=(0, 0, 0, hop - tmp_embed.size(0)))  # size: (hop, bert_out_size)
            str_path_embeds.append(tmp_embed)
    str_path_embeds = torch.stack(str_path_embeds, dim=0)  # size: (num_poss_paths, hop, bert_out_size)
    write_obj(obj=[id_paths, str_paths, str_path_embeds], obj_name="{}-hop ids, strings, and embeddings of possible shortest paths".format(hop), file_path=args.out_path + args.qa_type + "/{}H-poss_paths.pickle".format(hop))

    ent2id = kg["ent2id"]
    id2ent = {}
    for k, v in ent2id.items():
        id2ent[v] = k
    nx_kg = nx.MultiGraph()
    nx_kg.add_nodes_from(id2ent.keys())
    trp2id = kg["trp2id"]
    for trp, _ in trp2id.items():
        nx_kg.add_edge(trp[0], trp[2], rel=trp[1])
    write_obj(obj=nx_kg, obj_name="{}-hop networkx graph".format(hop), file_path=args.out_path + args.qa_type + "/{}H-nx_kg.pickle".format(hop))

    for mode in ["train", "valid", "test"]:
        ques = read_obj(obj_name="preprocessed {}-hop {} questions".format(hop, mode), file_path=args.out_path + args.qa_type + "/{}H-{}_ques.pickle".format(hop, mode))
        qid2tops = ques[1]
        qid2ans = ques[3]
        qid2path_dis = {}
        for qid, ans in qid2ans.items():
            top = qid2tops[qid]
            path2occur = defaultdict(int)
            for an in ans:
                if an == top:
                    path2occur[(-1,)] += 1
                else:
                    for id_path in find_shortest_paths(nx_kg=nx_kg, source=int(an), target=top, hop=hop):
                        if hop == 2:
                            path2occur[id_path] += 1
                        else:
                            if id_path[1:] not in [(1, 1), (8, 8), (9, 9), (10, 10), (12, 12)] and id_path[:2] not in [(1, 1), (8, 8), (9, 9), (10, 10), (12, 12)]:
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
        write_obj(obj=qid2path_dis, obj_name="{}-hop question id to shortest path distance".format(hop), file_path=args.out_path + args.qa_type + "/{}H-{}_qid2path_dis.pickle".format(hop, mode))


def find_shortest_paths(nx_kg: nx.MultiGraph, source: int, target: int, hop: int) -> list:
    all_paths = nx.all_simple_paths(G=nx_kg, source=source, target=target, cutoff=hop)
    all_paths = [p for p in all_paths]
    all_id_paths = set()
    if hop == 2:
        min_len = 999
        for path in all_paths:
            min_len = len(path) if min_len > len(path) else min_len
        for path in all_paths:
            if len(path) == min_len:
                id_paths = []
                for step in range(min_len - 1):
                    poss_rel_ids = []
                    for _, attr in dict(nx_kg[path[step]][path[step + 1]]).items():
                        poss_rel = attr["rel"]
                        poss_rel_ids.append(poss_rel)
                    id_paths.append(poss_rel_ids)
                id_paths = list(product(*id_paths))
                for id_path in id_paths:
                    all_id_paths.add(id_path)
    else:
        for path in all_paths:
            if len(path) == 2 or len(path) == 4:
                id_paths = []
                for step in range(len(path) - 1):
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
    args.add_argument("--qa_type", type=str, default="2-hop", choices={'2-hop', '3-hop'})
    args.add_argument("--in_path", type=str, default="../data/pathquestion/input/")
    args.add_argument("--out_path", type=str, default="../data/pathquestion/output/")
    args.add_argument('--device', type=int, default=1)
    args.add_argument('--seed', type=int, default=123)
    args = args.parse_args()

    hop = int(args.qa_type[0])
    split(seed=args.seed)
    qa_prep()
    prep_que_embeds()
    qa_batch_prep()
    qa_shortest_path_prep()
