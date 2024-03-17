import torch
import torch.nn as nn
from torch_scatter import scatter
from torch import LongTensor, FloatTensor


class AttenModel(nn.Module):
    def __init__(self, device: torch.device, norm: int, ent_init_embeds: FloatTensor, rel_init_embeds: FloatTensor,
                 in_dims: list, out_dims: list, dropouts: list, inv_rel: float):
        super(AttenModel, self).__init__()
        self.device = device
        self.norm = norm
        self.num_layers = len(in_dims)

        self.ent_init_embeds = ent_init_embeds.to(self.device)  # size: (num_ents, bert_output_size)
        self.rel_init_embeds = rel_init_embeds.to(self.device)  # size: (num_rels, bert_output_size)
        self.rel_init_embeds = torch.cat([self.rel_init_embeds, self.rel_init_embeds * inv_rel], dim=0)  # size: (2 * num_rels, bert_output_size)
        self.rel_init_embeds = torch.cat([self.rel_init_embeds, torch.full(size=[1, self.rel_init_embeds.size()[1]], fill_value=0., dtype=torch.float).to(self.device)], dim=0)  # size: (2 * num_rels + 1, bert_output_size)

        self.lstms = nn.ModuleList()
        for l_id in range(self.num_layers):
            self.lstms.append(nn.LSTM(input_size=in_dims[l_id], hidden_size=out_dims[l_id]).to(self.device))

        self.gcn_layers = nn.ModuleList()
        for l_id in range(self.num_layers):
            self.gcn_layers.append(GCNLayer(in_dim=in_dims[l_id], out_dim=out_dims[l_id], dropout=dropouts[l_id]).to(self.device))

    def forward(self, que_embeds: FloatTensor, x: LongTensor, loc_edge_index: LongTensor, edge_attr: LongTensor, edge_type: LongTensor) -> FloatTensor:
        que_states = [que_embeds.unsqueeze(1)]
        que_context = []
        for l_id in range(self.num_layers):
            state, (_, context) = self.lstms[l_id](que_states[l_id])
            que_states.append(state)  # size: (que_length, 1, out_dim)
            que_context.append(context.view(-1))  # size: (out_dim,)

        x = torch.index_select(input=self.ent_init_embeds, index=x, dim=0)  # size: (num_ents_in_subg, bert_output_size)
        r = self.rel_init_embeds  # size: (num_rels, bert_output_size)
        for l_id in range(self.num_layers):
            x, r = self.gcn_layers[l_id](x=x, r=r,
                                         que_context=que_context[l_id], edge_index=loc_edge_index,
                                         edge_attr=edge_attr, edge_type=edge_type)
        all_dis = torch.norm(que_context[-1].unsqueeze(0) - x, dim=1, p=self.norm)  # size: (num_ents_in_subg,)
        return torch.sigmoid(all_dis)

    def encode_que(self, que_embeds: FloatTensor):
        que_states = [que_embeds.unsqueeze(1)]
        que_context = []
        for l_id in range(self.num_layers):
            state, (_, context) = self.lstms[l_id](que_states[l_id].to(self.device))
            que_states.append(state)  # size: (que_length, 1, out_dim)
            que_context.append(context.view(-1))  # size: (out_dim,)
        return que_context

    def encode_kg(self, que_context: list, x: LongTensor, loc_edge_index: LongTensor, edge_attr: LongTensor, edge_type: LongTensor) -> list:
        x = torch.index_select(input=self.ent_init_embeds, index=x.to(self.device), dim=0)  # size: (num_ents_in_subg, bert_output_size)
        r = self.rel_init_embeds  # size: (num_rels, bert_output_size)

        x_list = []
        for l_id in range(self.num_layers):
            x, r = self.gcn_layers[l_id](x=x.to(self.device), r=r.to(self.device), que_context=que_context[l_id].to(self.device),
                                         edge_index=loc_edge_index.to(self.device), edge_attr=edge_attr.to(self.device), edge_type=edge_type.to(self.device))
            x_list.append(x)
        return x, x_list


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super(GCNLayer, self).__init__()
        self.mess_trans = nn.Linear(2 * in_dim, out_dim)

        self.atten_weight = nn.Parameter(torch.FloatTensor(1, 2 * out_dim))
        nn.init.xavier_normal_(self.atten_weight)

        # relation transformation
        self.rel_trans = nn.Linear(in_dim, out_dim)

        # batch normalization
        self.e_bn = nn.BatchNorm1d(out_dim)
        self.r_bn = nn.BatchNorm1d(out_dim)

        # dropout layer
        self.dp = nn.Dropout(dropout)

    def forward(self, x: FloatTensor, r: FloatTensor, que_context: FloatTensor, edge_index: LongTensor, edge_attr: LongTensor, edge_type: LongTensor) -> [FloatTensor, FloatTensor]:
        # message computation and transformation
        source_ents = edge_index[0, :]  # size: (num_edges,)
        source_embeds = torch.index_select(input=x, index=source_ents, dim=0)  # size: (num_edges, in_dim)
        edge_rel_embeds = torch.index_select(input=r, index=edge_attr, dim=0)  # size: (num_edges, in_dim)
        messages = torch.cat([source_embeds, edge_rel_embeds], dim=1)  # size: (num_edges, 2 * in_dim)
        messages = self.mess_trans(messages)  # size: (num_edges, out_dim)

        target_ents = edge_index[1, :]  # size: (num_ents_in_subg,)

        # attentional message aggregation
        que_context = que_context.unsqueeze(0).repeat(messages.size(0), 1)  # size: (num_edges, out_dims)
        atten_coeffs = torch.cat([messages, que_context], dim=1).unsqueeze(2)  # size: (num_edges, 2 * out_dim, 1)
        atten_coeffs = torch.matmul(self.atten_weight, atten_coeffs).view(-1)  # size: (num_edges,)
        atten_coeffs = torch.exp(torch.tanh(atten_coeffs))  # size: (num_edges,)

        coeffs_sum = scatter(src=atten_coeffs, index=target_ents, dim=0, reduce="sum")  # size: (num_ents_in_subg,)
        coeffs_sum = torch.index_select(input=coeffs_sum, index=target_ents, dim=0)  # size: (num_edges,)
        weights = torch.div(atten_coeffs, coeffs_sum)  # size: (num_edges,)
        messages = torch.mul(messages, weights.unsqueeze(1).repeat(1, messages.size(1)))  # size: (num_edges, out_dim)

        x = scatter(src=messages, index=target_ents, dim=0, reduce="sum")  # size: (num_ents_in_subg, out_dim)
        x = self.e_bn(x)
        x = torch.tanh(x)
        x = self.dp(x)

        # relation embedding transformation
        r = self.rel_trans(r)  # size: (num_rels, out_dim)
        r = self.r_bn(r)
        r = torch.tanh(r)
        r = self.dp(r)

        return x, r


class PathAlign(nn.Module):
    def __init__(self, poss_path_embeds: FloatTensor, in_dim: int, out_dim: int, norm: int, device: torch.device):
        super(PathAlign, self).__init__()
        self.norm = norm
        self.device = device
        self.poss_path_embeds = torch.transpose(poss_path_embeds.to(self.device), 0, 1)  # size: (hop, num_poss_paths, bert_out_size)

        self.que_lstm = nn.LSTM(input_size=in_dim, hidden_size=out_dim).to(self.device)
        self.path_lstm = nn.LSTM(input_size=in_dim, hidden_size=out_dim).to(self.device)

    def forward(self, que_embed: FloatTensor) -> FloatTensor:
        que_embed = que_embed.unsqueeze(1).to(self.device)  # size: (que_length, 1, bert_out_size)
        _, (_, que_embed) = self.que_lstm(que_embed)  # size: (1, 1, out_dim)
        _, (_, poss_path_embeds) = self.path_lstm(self.poss_path_embeds)  # size: (1, num_poss_paths, out_dim)
        all_dis = torch.norm(que_embed.squeeze(0) - poss_path_embeds.squeeze(0), dim=1, p=self.norm)  # size: (num_poss_paths,)
        return torch.sigmoid(all_dis)

    def infer(self, que_embed: FloatTensor, path_id: int) -> float:
        que_embed = que_embed.unsqueeze(1).to(self.device)  # size: (que_length, 1, bert_out_size)
        _, (_, que_embed) = self.que_lstm(que_embed)  # size: (1, 1, out_dim)
        path_embed = self.poss_path_embeds[:, path_id, :].unsqueeze(1)  # size: (hop, 1, bert_out_size)
        _, (_, path_embed) = self.path_lstm(path_embed)  # size: (1, 1, out_dim)
        dis = torch.norm(que_embed.view(-1) - path_embed.view(-1), dim=0, p=self.norm)  # size: (1,)
        return torch.sigmoid(dis)

    def que_enc(self, que_embed: FloatTensor) -> FloatTensor:
        que_embed = que_embed.unsqueeze(1).to(self.device)  # size: (que_length, 1, bert_out_size)
        _, (_, que_embed) = self.que_lstm(que_embed)  # size: (1, 1, out_dim)
        return que_embed.squeeze(0)  # size: (1, out_dim)

    def path_enc(self, path_ids: LongTensor) -> FloatTensor:
        path_embeds = torch.index_select(input=self.poss_path_embeds, index=path_ids.to(self.device), dim=1)  # size: (hop, num_poss_paths, bert_out_size)
        _, (_, path_embeds) = self.path_lstm(path_embeds)  # size: (1, num_poss_paths, out_dim)
        return path_embeds.squeeze(0)  # size: (num_poss_paths, out_dim)

