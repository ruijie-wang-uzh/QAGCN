from tqdm import tqdm
from argparse import ArgumentParser
import math
import torch
import graph_tool
from multiprocessing import Pool
from torch_scatter import scatter
from transformers import BertTokenizer, BertModel
from models import PathAlign, AttenModel
from utils import new_find_batch_paths, get_time, read_obj, new_mq_que_proc, subg_extract, bert_encode


class NewInfer:
    def __init__(self):
        print('\n## Inference - {}'.format(get_time()))
        for key, value in vars(args).items():
            print("* {}: {}".format(key, value))

        print('* loading data and models')

        self.atten_model_path = args.out_path + 'models/{}-atten_model({}).pt'.format(args.qa_type, args.timestamp)
        self.path_align_model_path = args.out_path + 'models/{}-path_align_model({}).pt'.format(args.qa_type, args.path_align_timestamp)

        self.ent_embeds = read_obj(obj_name='entity label embeddings', file_path=args.out_path + 'ent_label_embeds.pickle')
        self.rel_embeds = read_obj(obj_name='relation label embeddings', file_path=args.out_path + 'rel_label_embeds.pickle')

        print('\t* loading the pre-trained alignment model from {}'.format(self.atten_model_path))
        self.atten_model = AttenModel(device=args.device, norm=args.norm, ent_init_embeds=self.ent_embeds, rel_init_embeds=self.rel_embeds, in_dims=args.in_dims, out_dims=args.out_dims,
                                      dropouts=args.dropouts, inv_rel=args.inv_rel_embeds)
        self.atten_model.load_state_dict(torch.load(self.atten_model_path, map_location=torch.device(args.device)))
        self.atten_model.eval()

        print('\t* loading the pre-trained path ranking model from {}'.format(self.path_align_model_path))
        self.id_paths, _, self.poss_path_embeds = read_obj(obj_name='preprocessed path data', file_path=args.out_path + args.qa_type + '/poss_paths.pickle')
        self.id_path2idx = {}
        for idx, id_path in enumerate(self.id_paths):
            self.id_path2idx[id_path] = idx
        self.path_align_model = PathAlign(poss_path_embeds=self.poss_path_embeds, in_dim=args.in_dims[0], out_dim=args.out_dims[-1], norm=args.norm, device=args.device)
        self.path_align_model.load_state_dict(torch.load(self.path_align_model_path, map_location=torch.device(args.device)))
        self.path_align_model.eval()

        self.kg_graph = read_obj(obj_name='kg geo data', file_path=args.out_path + "kg-geo_x.pickle")
        self.kg = read_obj(obj_name='kg data', file_path=args.out_path + 'kg.pickle')
        self.ent2id = self.kg['ent2id']
        self.id2ent = {}
        for k, v in self.ent2id.items():
            self.id2ent[v] = k
        self.id2rel = {}
        for k, v in self.kg['rel2id'].items():
            self.id2rel[v] = k

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(args.device)
        self.bert_model.eval()

        self.path_find_pool = Pool(processes=args.num_procs)

        gt_graph = graph_tool.Graph(directed=False)
        gt_graph.add_vertex(len(self.ent2id))
        eprop = gt_graph.new_edge_property('int')
        trp2id = self.kg["trp2id"]
        for trp, _ in trp2id.items():
            head = trp[0]
            tail = trp[2]
            rel = trp[1]
            eprop[gt_graph.add_edge(gt_graph.vertex(head), gt_graph.vertex(tail))] = rel
        gt_graph.edge_properties['edge_attr'] = eprop

        self.gt_graphs = [graph_tool.Graph(gt_graph) for _ in range(args.num_procs)]

        dict_path = args.out_path + "dict.pickle"
        classifier_path = args.out_path + "models/mq_hop_prep.pt"

    def infer(self):
        que_path = args.in_path + args.qa_type + "/qa_test.txt"
        with open(file=que_path, mode='r') as f:
            lines = f.readlines()
            num_ques = len(lines)
            embed_corr = 0
            rerank_corr = 0

            print('* total number of test questions: {}'.format(num_ques))
            for line in tqdm(lines):

                # ======================== question processing
                elems = line.rstrip().split('\t')
                que, top, erps, ans = new_mq_que_proc(elems=elems, ent2id=self.ent2id)

                # ======================== subgraph processing
                subg, cans = subg_extract(hop=int(args.qa_type[0]), ent_idx=top,
                                          edge_index=self.kg_graph.edge_index.to(args.device),
                                          edge_attr=self.kg_graph.edge_attr.to(args.device),
                                          edge_type=self.kg_graph.edge_type.to(args.device))

                glo_ents = cans  # size: (num_ents_in_subg,)
                loc_ids = torch.arange(glo_ents.size()[0], device=args.device)
                mapping = scatter(src=loc_ids, index=glo_ents, dim=0)

                edge_attr = subg.edge_attr
                edge_type = subg.edge_type
                glo_src_ents = subg.edge_index[0, :]
                glo_tar_ents = subg.edge_index[1, :]
                loc_src_ents = mapping[glo_src_ents]
                loc_tar_ents = mapping[glo_tar_ents]
                loc_edge_index = torch.stack([loc_src_ents, loc_tar_ents], dim=0)

                loc_edge_index = torch.cat([loc_edge_index, torch.stack([loc_tar_ents, loc_src_ents], dim=0)], dim=1)
                edge_attr = torch.cat([edge_attr, edge_attr + self.kg['num_rels']], dim=0)
                edge_type = torch.cat([edge_type, torch.full_like(input=edge_type, fill_value=1)], dim=0)
                loc_edge_index = torch.cat([loc_edge_index, torch.stack([loc_ids, loc_ids], dim=0)], dim=1)
                edge_attr = torch.cat([edge_attr, torch.full(size=[loc_ids.size()[0]], dtype=torch.long, fill_value=2 * self.kg['num_rels'])], dim=0)
                edge_type = torch.cat([edge_type, torch.full(size=[loc_ids.size()[0]], dtype=torch.long, fill_value=2)], dim=0)

                # ======================== question encoding
                que_embed = bert_encode(tokenizer=self.tokenizer, model=self.bert_model, batch_str=erps, device=args.device)
                que_contexts = self.atten_model.encode_que(que_embeds=que_embed)

                # ======================== graph encoding
                x, x_list = self.atten_model.encode_kg(que_context=que_contexts, x=glo_ents, loc_edge_index=loc_edge_index, edge_attr=edge_attr, edge_type=edge_type)
                dis = torch.norm(que_contexts[-1].unsqueeze(0) - x, dim=1, p=args.norm)  # size: (num_ents_in_subg,)

                # ======================== answer search
                knn = []
                for _ in range(args.rerank_top):
                    if len(knn) == dis.size()[0]:
                        break
                    tmp_dis, loc_an = torch.min(input=dis, dim=0)
                    knn.append(int(loc_an))
                    dis[loc_an] = 9999.

                # ======================== answer reranking
                if int(glo_ents[knn[0]]) in ans:
                    embed_corr += 1
                tmp_args = [(int(glo_ents[loc_an]), top) for loc_an in knn]

                arg_batches = []
                batch_size = math.ceil(len(tmp_args) / args.num_procs)
                for i in range(0, len(tmp_args), batch_size):
                    arg_batches.append(
                        [
                            self.gt_graphs[len(arg_batches)],
                            [tmp_args[_][0] for _ in range(i, i + batch_size if i + batch_size < len(tmp_args) else len(tmp_args))],
                            [tmp_args[_][1] for _ in range(i, i + batch_size if i + batch_size < len(tmp_args) else len(tmp_args))],
                            int(args.qa_type[0])
                        ]
                    )
                all_paths_batches = self.path_find_pool.starmap(new_find_batch_paths, arg_batches)
                all_paths = []
                for _ in all_paths_batches:
                    all_paths += _

                can_ids = []
                path_ids = []
                real_paths = []
                for idx, tmp_paths in enumerate(all_paths):
                    for id_path in tmp_paths:
                        can_ids.append(idx)
                        path_ids.append(self.id_path2idx[id_path])
                        real_paths.append(id_path)
                can_ids = torch.LongTensor(can_ids)
                path_ids = torch.LongTensor(path_ids)  # size: (num_paths_for_all_candidates)

                que_embed = self.path_align_model.que_enc(que_embed=que_embed)  # size: (1, out_dim)
                path_embeds = self.path_align_model.path_enc(path_ids=path_ids)  # size: (num_poss_paths, out_dim)
                dis = torch.norm(que_embed - path_embeds, dim=1, p=args.norm)  # size: (num_poss_paths)
                min_paths = torch.min(input=dis, dim=0)
                final_an = can_ids[min_paths[1]]
                if int(glo_ents[knn[final_an]]) in ans:
                    rerank_corr += 1
                # final_rel = [self.id2rel[_] if _ != -1 else 'self-loop' for _ in real_paths[min_paths[1]]]  # interpretable paths for final answers

        print('* num_ques: {}, embed_hits@1: {}, post-hits@1: {}'.format(num_ques, embed_corr/num_ques, rerank_corr/num_ques))


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--device', type=int, default=0)
    args.add_argument("--qa_type", type=str, default="3-hop")
    args.add_argument("--in_dims", type=int, nargs='+', default=[768, 512, 256])
    args.add_argument("--out_dims", type=int, nargs='+', default=[512, 256, 128])
    args.add_argument("--in_path", type=str, default="../data/metaqa/input/")
    args.add_argument("--out_path", type=str, default="../data/metaqa/output/")
    args.add_argument('--path_align_timestamp', type=str, default="")
    args.add_argument("--norm", type=int, default=2)
    args.add_argument("--dropouts", type=float, nargs='+', default=[0.1, 0.1, 0.])
    args.add_argument("--timestamp", type=str, default='')
    args.add_argument("--inv_rel_embeds", type=float, default=-1.)
    args.add_argument("--rerank_top", type=int, default=8000)
    args.add_argument('--num_procs', type=int, default=60)
    args = args.parse_args()


    new_infer = NewInfer()
    with torch.no_grad():
        new_infer.infer()
