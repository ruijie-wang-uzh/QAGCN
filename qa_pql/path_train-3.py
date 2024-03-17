import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from models import PathAlign3
from datetime import datetime
from torch.utils.data import DataLoader
from utils import get_time, read_obj, QAPathSet, collate_fn, compute_hits


class PathTrain:
    def __init__(self):
        print("\n## Run Start - Path Ranking")
        print("* started at {}".format(get_time()))
        for key, value in vars(args).items():
            print("* {}: {}".format(key, value))

        self.model_path = os.path.join(args.out_path, "models", "path_align_model({}).pt".format(datetime.now().strftime("%Y.%m.%d.%H.%M")))

        self.qid2embeds = read_obj(obj_name="train question embeddings", file_path=os.path.join(args.out_path, "{}H-train_qid2embeds.pickle".format(hop)))
        self.valid_qid2embeds = read_obj(obj_name="valid question embeddings", file_path=os.path.join(args.out_path, "{}H-valid_qid2embeds.pickle".format(hop)))
        self.test_qid2embeds = read_obj(obj_name="test question embeddings", file_path=os.path.join(args.out_path, "{}H-test_qid2embeds.pickle".format(hop)))

        self.qid2path_dis = read_obj(obj_name="target train shortest path distances", file_path=os.path.join(args.out_path, "{}H-train_qid2path_dis.pickle".format(hop)))
        self.valid_qid2path_dis = read_obj(obj_name="target valid shortest path distances", file_path=os.path.join(args.out_path, "{}H-valid_qid2path_dis.pickle".format(hop)))
        self.test_qid2path_dis = read_obj(obj_name="target test shortest path distances", file_path=os.path.join(args.out_path, "{}H-test_qid2path_dis.pickle".format(hop)))

    def train(self) -> None:
        print("### Training")
        path_align_model = PathAlign3(in_dim=args.in_dim, out_dim=args.out_dim, norm=args.norm, device=args.device)
        optimizer = optim.Adam(params=path_align_model.parameters(), lr=args.lr)
        criterion = nn.BCELoss(reduction="mean")

        qa_path_set = QAPathSet(qid2embeds=self.qid2embeds, qid2path_dis=self.qid2path_dis)
        train_qa_loader = DataLoader(dataset=qa_path_set, collate_fn=collate_fn, batch_size=1, shuffle=True)

        highest_acc = 0.
        for epoch in range(args.num_epochs):
            print("* epoch {} - {}".format(epoch, get_time()))
            path_align_model.train()
            epoch_loss = 0.
            invalid = 0
            for batch_id, batch_data in enumerate(tqdm(train_qa_loader)):
                optimizer.zero_grad()

                que_embed, [id_paths, str_paths, str_path_embeds, path_dis] = batch_data
                if torch.sum(path_dis == 0):
                    all_dis = path_align_model(que_embed=que_embed.cuda(), path_embeds=torch.transpose(str_path_embeds, 0, 1).cuda())
                    batch_loss = criterion(all_dis, path_dis.to(args.device))

                    batch_loss.backward()
                    optimizer.step()
                    epoch_loss += batch_loss.item()
                else:
                    invalid += 1
            print("\t* loss: {}, time: {}, invalid: {}".format(epoch_loss, get_time(), invalid, invalid/len(train_qa_loader)))

            acc = self.eval(mode="valid", model=path_align_model)
            if acc > highest_acc:
                highest_acc = acc
                torch.save(path_align_model.state_dict(), self.model_path)
                print("\t* model saved to {}".format(self.model_path))

    def eval(self, mode: str, model: PathAlign3) -> float:
        assert mode in ["valid", "test"], "wrong mode!"
        model.eval()
        with torch.no_grad():
            if mode == "valid":
                eval_que2embeds = self.valid_qid2embeds
                eval_que2path_dis = self.valid_qid2path_dis
            else:
                eval_que2embeds = self.test_qid2embeds
                eval_que2path_dis = self.test_qid2path_dis

            eval_qa_path_set = QAPathSet(qid2embeds=eval_que2embeds, qid2path_dis=eval_que2path_dis)
            eval_qa_loader = DataLoader(dataset=eval_qa_path_set, collate_fn=collate_fn, batch_size=1, shuffle=False)

            all_ranks = []
            all_ranks_inc_equ = []
            for batch_id, batch_data in enumerate(tqdm(eval_qa_loader)):
                que_embed, [id_paths, str_paths, str_path_embeds, path_dis] = batch_data
                all_dis = model(que_embed=que_embed.cuda(), path_embeds=torch.transpose(str_path_embeds, 0, 1).cuda())

                if torch.sum(path_dis == 0):
                    neg_masks = path_dis.to(args.device) * 9999
                    corr_embed_dis = all_dis + neg_masks  # distances of answers remain unchanged; others' are added with 9999., size: (num_ents_in_subg,)
                    corr_embed_dis = torch.min(input=corr_embed_dis)  # size: (1,)

                    corr_ranks = torch.nonzero(all_dis < corr_embed_dis, as_tuple=True)[0].size()[0] + 1
                    corr_ranks_inc_equ = torch.nonzero(all_dis <= corr_embed_dis, as_tuple=True)[0].size()[0]
                else:
                    corr_ranks = all_dis.size(0)
                    corr_ranks_inc_equ = all_dis.size(0)
                all_ranks.append(corr_ranks)
                all_ranks_inc_equ.append(corr_ranks_inc_equ)
            all_ranks = torch.FloatTensor(all_ranks)  # (num_eval_ques)
            all_ranks_inc_equ = torch.FloatTensor(all_ranks_inc_equ)  # (num_eval_ques)

            mr = torch.mean(all_ranks_inc_equ.float())
            acc = compute_hits(all_ranks_inc_equ, 1)
            acc_exc_equ = compute_hits(all_ranks, 1)
            hits3 = compute_hits(all_ranks_inc_equ, 3)
            hits10 = compute_hits(all_ranks_inc_equ, 10)
            hits20 = compute_hits(all_ranks_inc_equ, 20)
            hits50 = compute_hits(all_ranks_inc_equ, 50)
            hits100 = compute_hits(all_ranks_inc_equ, 100)
            print("\t* mean rank: {}, hits@1: {}, hits@1(exc_equ): {}, hits@3: {}, hits@10: {}, hits@20: {}, hits@50: {}, hits@100: {}".format(mr, acc, acc_exc_equ, hits3, hits10, hits20, hits50, hits100))

            return acc

    def test(self) -> None:
        print("### Testing")
        path_align_model = PathAlign3(in_dim=args.in_dim, out_dim=args.out_dim, norm=args.norm, device=args.device)

        if args.num_epochs == 0:
            assert args.path_align_timestamp != "", "no timestamp given!"
            self.model_path = os.path.join(args.out_path, "models", "path_align_model({}).pt".format(args.path_align_timestamp))

        print("* loading the pre-trained model from {}".format(self.model_path))
        path_align_model.load_state_dict(torch.load(self.model_path))
        self.eval(mode="test", model=path_align_model)


if __name__ == "__main__":

    args = ArgumentParser()
    args.add_argument("--qa_type", type=str, default="3-hop")
    args.add_argument("--out_path", type=str, default="../data/pql-3hop/output")
    args.add_argument("--in_dim", type=int, default=768)
    args.add_argument("--out_dim", type=int, default=128)
    args.add_argument("--lr", type=float, default=5e-4)
    args.add_argument("--norm", type=int, default=2)
    args.add_argument("--num_epochs", type=int, default=20)
    args.add_argument('--path_align_timestamp', type=str, default="")
    args.add_argument('--device', type=int, default=0)
    args = args.parse_args()

    hop = int(args.qa_type[0])

    path_train = PathTrain()
    path_train.train()
    path_train.test()
