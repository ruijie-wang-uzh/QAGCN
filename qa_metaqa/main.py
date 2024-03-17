import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import AttenModel, PathAlign
from que_prep import find_shortest_paths
from utils import read_obj, get_time, QASet, collate_fn, compute_hits, find_top, write_obj
from argparse import ArgumentParser


class Main:
    def __init__(self):
        print("\n## Run Start")
        print("* started at {}".format(get_time()))
        for key, value in vars(args).items():
            print("* {}: {}".format(key, value))

        self.model_path = args.out_path + "models/{}-atten_model({}).pt".format(args.qa_type, datetime.now().strftime("%Y.%m.%d.%H.%M"))
        self.path_align_model_path = args.out_path + "models/{}-path_align_model({}).pt".format(args.qa_type, args.path_align_timestamp)

        print("* loading data...")
        self.ent_embeds = read_obj(obj_name="entity embeddings", file_path=args.out_path + "ent_label_embeds.pickle")
        self.rel_embeds = read_obj(obj_name="relation embeddings", file_path=args.out_path + "rel_label_embeds.pickle")

        if args.num_epochs > 0:
            self.train_qid2batch = read_obj(obj_name="train qa batch", file_path=args.out_path + args.qa_type + "/train_qid2{}batch.pickle".format('real_' if args.qa_type[0] == '3' else ''))
            self.valid_qid2batch = read_obj(obj_name="valid qa batch", file_path=args.out_path + args.qa_type + "/valid_qid2{}batch.pickle".format('real_' if args.qa_type[0] == '3' else ''))
        self.test_qid2batch = read_obj(obj_name="test qa batch", file_path=args.out_path + args.qa_type + "/test_qid2{}batch.pickle".format('real_' if args.qa_type[0] == '3' else ''))

        if args.num_epochs > 0:
            self.train_qid2embeds = read_obj(obj_name="train question embeddings", file_path=args.out_path + args.qa_type + "/train_qid2embeds.pickle")
            self.valid_qid2embeds = read_obj(obj_name="valid question embeddings", file_path=args.out_path + args.qa_type + "/valid_qid2embeds.pickle")
        self.test_qid2embeds = read_obj(obj_name="test question embeddings", file_path=args.out_path + args.qa_type + "/test_qid2embeds.pickle")

        if args.num_epochs > 0:
            self.valid_qid2tops = read_obj(obj_name="preprocessed valid questions", file_path=args.out_path + args.qa_type + "/valid_ques.pickle")[1]
        self.test_qid2tops = read_obj(obj_name="preprocessed test questions", file_path=args.out_path + args.qa_type + "/test_ques.pickle")[1]

        self.nx_kg = read_obj(obj_name="the networkx graph", file_path=args.out_path + "nx_kg.pickle")
        self.id_paths, _, self.poss_path_embeds = read_obj(obj_name="ids, strings, and embeddings of possible shortest paths", file_path=args.out_path + args.qa_type + "/poss_paths.pickle")
        self.id_path2idx = {}
        for id_path_id in range(len(self.id_paths)):
            self.id_path2idx[self.id_paths[id_path_id]] = id_path_id

        self.path_align_model = PathAlign(poss_path_embeds=self.poss_path_embeds, in_dim=args.in_dims[0], out_dim=args.out_dims[-1], norm=args.norm, device=args.device)
        print("\t* loading the pre-trained path ranking model from {}".format(self.path_align_model_path))
        self.path_align_model.load_state_dict(torch.load(self.path_align_model_path, map_location=torch.device(args.device)))
        self.path_align_model.eval()

        self.path_cache = {}

    def train(self) -> None:
        print("### Training")
        atten_model = AttenModel(device=args.device, norm=args.norm, ent_init_embeds=self.ent_embeds, rel_init_embeds=self.rel_embeds, in_dims=args.in_dims,
                                 out_dims=args.out_dims, dropouts=args.dropouts, inv_rel=args.inv_rel_embeds)
        if args.continue_train:
            print("* loading the pre-trained model from {}".format(args.out_path + "models/{}-atten_model({}).pt".format(args.qa_type, args.timestamp)))
            atten_model.load_state_dict(torch.load(args.out_path + "models/{}-atten_model({}).pt".format(args.qa_type, args.timestamp)))
        optimizer = optim.Adam(params=atten_model.parameters(), lr=args.lr)
        criterion = nn.BCELoss(reduction="mean")

        train_qa_data = QASet(qid2batch=self.train_qid2batch, qid2embeds=self.train_qid2embeds)
        train_qa_loader = DataLoader(dataset=train_qa_data, collate_fn=collate_fn, batch_size=1, shuffle=True)

        highest_acc = 0.
        for epoch in range(args.num_epochs):
            print("* epoch {} - {}".format(epoch, get_time()))
            atten_model.train()
            epoch_loss = 0.
            for batch_id, batch_data in enumerate(train_qa_loader):
                optimizer.zero_grad()
                [_, glo_ents, loc_edge_index, edge_attr, edge_type, loc_ans_dis], que_embed, _ = batch_data
                all_dis = atten_model(que_embeds=que_embed.to(args.device), x=glo_ents.to(args.device),
                                      loc_edge_index=loc_edge_index.to(args.device), edge_attr=edge_attr.to(args.device), edge_type=edge_type.to(args.device))
                batch_loss = criterion(all_dis, loc_ans_dis.to(args.device))
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
            print("\t* loss: {}, time: {}".format(epoch_loss, get_time()))

            acc = self.eval(mode="valid", model=atten_model)
            if acc > highest_acc:
                highest_acc = acc
                torch.save(atten_model.state_dict(), self.model_path)
                print("\t* model saved to {}".format(self.model_path))

    def eval(self, mode: str, model: AttenModel) -> float:
        assert mode in ["valid", "test"], "wrong mode!"
        model.eval()
        with torch.no_grad():
            if mode == "valid":
                eval_qid2batch = self.valid_qid2batch
                eval_qid2embeds = self.valid_qid2embeds
                eval_qid2tops = self.valid_qid2tops
            else:
                eval_qid2batch = self.test_qid2batch
                eval_qid2embeds = self.test_qid2embeds
                eval_qid2tops = self.test_qid2tops

            eval_qa_data = QASet(qid2batch=eval_qid2batch, qid2embeds=eval_qid2embeds)
            eval_qa_loader = DataLoader(dataset=eval_qa_data, collate_fn=collate_fn, batch_size=1, shuffle=True)

            all_embed_ranks = []
            all_path_ranks = []
            for batch_id, batch_data in tqdm(enumerate(eval_qa_loader), total=len(eval_qa_loader)):
                [_, glo_ents, loc_edge_index, edge_attr, edge_type, loc_ans_dis], que_embed, qid = batch_data
                all_dis = model(que_embeds=que_embed.to(args.device), x=glo_ents.to(args.device),
                                loc_edge_index=loc_edge_index.to(args.device), edge_attr=edge_attr.to(args.device), edge_type=edge_type.to(args.device))

                neg_masks = loc_ans_dis.to(args.device) * 9999
                corr_embed_dis = all_dis + neg_masks  # distances of answers remain unchanged; others' are added with 9999., size: (num_ents_in_subg,)
                corr_embed_dis = torch.min(input=corr_embed_dis)  # sizes: (1,);

                embed_rank = torch.nonzero(all_dis <= corr_embed_dis, as_tuple=True)[0].size()[0]
                path_rank = embed_rank

                if embed_rank <= args.rerank_top:  # no need to do re-ranking if the target answer is not in top-x
                    path_rank = args.rerank_top
                    top = eval_qid2tops[qid]
                    loc_an2embed_dis = []
                    for loc_an in range(all_dis.size(0)):
                        loc_an2embed_dis.append((loc_an, all_dis[loc_an]))
                    loc_an2embed_dis = find_top(tup_list=loc_an2embed_dis, top=args.rerank_top)
                    loc_an2path_dis = []
                    for loc_an, _ in loc_an2embed_dis:
                        source_ent = glo_ents[loc_an].item()
                        target_ent = top
                        if (source_ent, target_ent) not in self.path_cache:
                            self.path_cache[(source_ent, target_ent)] = find_shortest_paths(nx_kg=self.nx_kg, source=source_ent, target=target_ent, hop=int(args.qa_type[0]))
                        shortest_paths = self.path_cache[(source_ent, target_ent)]
                        lowest_dis = 9999
                        if len(shortest_paths) != 0:
                            for id_path in shortest_paths:
                                tmp_dis = self.path_align_model.infer(que_embed=que_embed, path_id=self.id_path2idx[id_path])
                                if lowest_dis > tmp_dis:
                                    lowest_dis = tmp_dis
                        loc_an2path_dis.append((loc_an, lowest_dis))
                    loc_an2path_dis = sorted(loc_an2path_dis, key=lambda tup: tup[1])

                    corr_ans = list(torch.nonzero(loc_ans_dis == 0., as_tuple=True)[0])  # list of correct answers
                    for tmp_id in range(len(loc_an2path_dis)):
                        if loc_an2path_dis[tmp_id][0] in corr_ans:
                            path_rank = tmp_id + 1
                            break

                all_embed_ranks.append(embed_rank)
                all_path_ranks.append(path_rank)

            all_embed_ranks = torch.FloatTensor(all_embed_ranks)  # (num_eval_ques)
            all_path_ranks = torch.FloatTensor(all_path_ranks)

            mr = torch.mean(all_embed_ranks.float())
            acc = compute_hits(all_embed_ranks, 1)
            hits3 = compute_hits(all_embed_ranks, 3)
            hits10 = compute_hits(all_embed_ranks, 10)
            hits20 = compute_hits(all_embed_ranks, 20)
            hits50 = compute_hits(all_embed_ranks, 50)
            hits100 = compute_hits(all_embed_ranks, 100)

            p_mr = torch.mean(all_path_ranks.float())
            p_acc = compute_hits(all_path_ranks, 1)
            p_hits3 = compute_hits(all_path_ranks, 3)
            p_hits10 = compute_hits(all_path_ranks, 10)
            p_hits20 = compute_hits(all_path_ranks, 20)
            p_hits50 = compute_hits(all_path_ranks, 50)
            p_hits100 = compute_hits(all_path_ranks, 100)
            print("\t* mean rank: {}, hits@1: {}, hits@3: {}, hits@10: {}, hits@20: {}, hits@50: {}, hits@100: {}".format(mr, acc, hits3, hits10, hits20, hits50, hits100))
            print("\t* post-mean rank: {}, post-hits@1: {}, post-hits@3: {}, post-hits@10: {}, post-hits@20: {}, post-hits@50: {}, post-hits@100: {}".format(p_mr, p_acc, p_hits3, p_hits10, p_hits20, p_hits50, p_hits100))

            write_obj(obj=self.path_cache, obj_name="{} path cache".format(args.qa_type), file_path=args.out_path + args.qa_type + "/path_cache_{}.pickle".format(args.rerank_top))

            return acc

    def test(self) -> None:
        print("### Testing")
        atten_model = AttenModel(device=args.device, norm=args.norm, ent_init_embeds=self.ent_embeds, rel_init_embeds=self.rel_embeds, in_dims=args.in_dims,
                                 out_dims=args.out_dims, dropouts=args.dropouts, inv_rel=args.inv_rel_embeds)
        if args.num_epochs == 0:
            assert args.timestamp != "", "no timestamp given!"
            self.model_path = args.out_path + "models/{}-atten_model({}).pt".format(args.qa_type, args.timestamp)
        print("* loading the pre-trained model from {}".format(self.model_path))
        atten_model.load_state_dict(torch.load(self.model_path, map_location=torch.device(args.device)))
        self.eval(mode="test", model=atten_model)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--device', type=int, default=0)
    args.add_argument("--qa_type", type=str, default="1-hop", choices={'1-hop', '2-hop'})
    args.add_argument("--in_dims", type=int, nargs='+', default=[768])
    args.add_argument("--out_dims", type=int, nargs='+', default=[256])
    args.add_argument("--out_path", type=str, default="../data/metaqa/output/")
    args.add_argument('--path_align_timestamp', type=str, default="")
    args.add_argument("--norm", type=int, default=2)
    args.add_argument("--dropouts", type=float, nargs='+', default=[0.])
    args.add_argument("--timestamp", type=str, default='')
    args.add_argument("--inv_rel_embeds", type=float, default=-1.)
    args.add_argument("--lr", type=float, default=0.0002)
    args.add_argument("--num_epochs", type=int, default=20)
    args.add_argument("--continue_train", default=False, action='store_true')
    args.add_argument("--rerank_top", type=int, default=20)
    args = args.parse_args()

    main = Main()
    if args.num_epochs > 0:
        main.train()
    main.test()
