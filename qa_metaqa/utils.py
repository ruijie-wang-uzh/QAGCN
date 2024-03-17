import re
import torch
import graph_tool
from graph_tool import topology
import pickle
from models import AttenModel
from datetime import datetime
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
from transformers import BertTokenizer, BertModel


stop_words = ["a", "as", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards",
              "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although",
              "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything",
              "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "arent", "around",
              "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "be", "became", "because",
              "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside",
              "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "cmon", "cs", "came", "can", "cant",
              "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes",
              "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding",
              "could", "couldn't", "course", "currently", "definitely", "described", "despite", "did", "didnt", "different",
              "do", "does", "doesnt", "doing", "dont", "done", "down", "downwards", "during", "each", "edu", "eg", "eight",
              "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody",
              "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "far", "few", "ff", "fifth", "first",
              "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further",
              "furthermore", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten",
              "greetings", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "hes",
              "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
              "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "id", "ill", "im", "ive",
              "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner",
              "insofar", "instead", "into", "inward", "is", "isn't", "it", "itd", "it'll", "its", "its", "itself", "just", "keep",
              "keeps", "kept", "know", "knows", "known", "last", "lately", "later", "latter", "latterly", "least", "less", "lest",
              "let", "lets", "like", "liked", "list", "likely", "little", "look", "looking", "looks", "ltd", "mainly", "many", "may",
              "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much", "must", "my",
              "myself", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless",
              "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now",
              "nowhere", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only",
              "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over",
              "overall", "own", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably",
              "probably", "provides", "que", "quite", "qv", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless",
              "regards", "relatively", "respectively", "right", "said", "saw", "say", "saying", "says", "second", "secondly",
              "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious",
              "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six", "so", "some", "somebody",
              "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified",
              "specify", "specifying", "still", "sub", "such", "sup", "sure", "ts", "take", "taken", "tell", "tends", "th",
              "than", "thank", "thanks", "thanx", "that", "thats", "that's", "the", "their", "theirs", "them", "themselves",
              "then", "thence", "there", "theres", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon",
              "these", "they", "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly",
              "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward",
              "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "un", "under", "unfortunately", "unless",
              "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "value",
              "various", "very", "via", "viz", "vs", "want", "wants", "was", "wasn't", "way", "we", "wed", "well", "were",
              "we've", "welcome", "well", "went", "were", "weren't", "what", "whats", "whatever", "when", "whence", "whenever",
              "where", "wheres", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which",
              "while", "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with",
              "within", "without", "wont", "wonder", "would", "would", "wouldn't", "yes", "yet", "you", "you'd", "you'll", "you're",
              "you've", "your", "yours", "yourself", "yourselves", "zero", "whose", "which", "is", ",", "\\\\", "?", "\\"]


def check_model(atten_model: AttenModel) -> None:
    print('* the attention model:\n```')
    print(atten_model)
    print('```')

    param_names = []
    for name, param in atten_model.named_parameters():
        if param.requires_grad:
            param_names.append(name)
    print('* model parameters: \n```\n{}\n```'.format(param_names))


def bert_encode(tokenizer: BertTokenizer, model: BertModel, batch_str: list, device: torch.device) -> FloatTensor:
    tokenized_batch = tokenizer(batch_str, padding=True, truncation=True, return_tensors='pt')
    output = model(input_ids=tokenized_batch['input_ids'].to(device), attention_mask=tokenized_batch['attention_mask'].to(device),
                   token_type_ids=tokenized_batch['token_type_ids'].to(device), output_hidden_states=True)
    hidden_states = output.hidden_states[-2]  # the second last layer's output, size: (batch_size, num_tokens_after_padding, state_dim)
    state_dim = hidden_states.size(2)
    atten_mask = tokenized_batch.attention_mask.to(device).unsqueeze(2).repeat(1, 1, state_dim)
    hidden_states = torch.sum(torch.mul(hidden_states, atten_mask), dim=1)  # size: (batch_size, state_dim)
    atten_mask = torch.sum(tokenized_batch.attention_mask.to(device), dim=1).unsqueeze(1).repeat(1, state_dim)  # size: (batch_size, state_dim)
    hidden_states = torch.div(hidden_states, atten_mask)  # size: (batch_size, state_dim)
    return hidden_states


def get_time() -> str:
    return datetime.now().strftime('%H:%M:%S %Y-%m-%d')


def write_obj(obj: object, obj_name: str, file_path: str) -> None:
    print('\t* dumping {} to {} at {} ...'.format(obj_name, file_path, get_time()))
    with open(file=file_path, mode='wb') as f:
        pickle.dump(obj=obj, file=f, protocol=4)


def read_obj(obj_name: str, file_path: str) -> dict:
    print('\t* loading {} from {} at {} ...'.format(obj_name, file_path, get_time()))
    with open(file=file_path, mode='rb') as f:
        obj = pickle.load(file=f)
    return obj


def add_elems(elem: str, data_dict: dict, num: int) -> int:
    if elem not in data_dict:
        data_dict[elem] = num
        num += 1
    return num


class QASet(Dataset):
    def __init__(self, qid2batch: dict, qid2embeds: dict):
        super(QASet, self).__init__()
        self.qid2batch = qid2batch
        self.qid2embeds = qid2embeds

    def __len__(self) -> int:
        return len(self.qid2batch)

    def __getitem__(self, item) -> list:
        return [self.qid2batch[item], self.qid2embeds[item], item]


def collate_fn(batch_data) -> tuple:
    return batch_data[0]


def compute_hits(ranks: LongTensor, hit: int) -> float:
    return torch.nonzero(ranks <= hit, as_tuple=True)[0].size()[0] / ranks.size()[0]


class QAPathSet(Dataset):
    def __init__(self, qid2embeds: FloatTensor, qid2path_dis: FloatTensor):
        super(QAPathSet, self).__init__()
        self.qid2embeds = qid2embeds
        self.qid2path_dis = qid2path_dis

    def __len__(self) -> int:
        return len(self.qid2embeds)

    def __getitem__(self, item) -> list:
        return [self.qid2embeds[item], self.qid2path_dis[item]]


def find_top(tup_list: list, top: int) -> list:
    top_list = []
    added_ids = []
    for _ in range(top):
        min_tup = (None, 9999)
        for tup in tup_list:
            if tup[0] not in added_ids and tup[1] < min_tup[1]:
                min_tup = tup
        if min_tup[0] is not None:
            top_list.append(min_tup)
            added_ids.append(min_tup[0])
    return top_list


def new_find_batch_paths(graph: graph_tool.Graph, sources: list, targets: list, hop: int) -> list:
    all_id_paths_batch = []
    for tmp_idx in range(len(sources)):
        all_id_paths_batch.append(new_find_paths(graph=graph, source=sources[tmp_idx], target=targets[tmp_idx], hop=hop))
    return all_id_paths_batch

def new_find_paths(graph: graph_tool.Graph, source: int, target: int, hop: int) -> list:
    all_id_paths = set()
    if source != target:
        all_paths = topology.all_paths(g=graph, source=source, target=target, cutoff=hop, edges=True)
        for path in all_paths:
            if len(path) == hop:
                id_path = []
                for edge in path:
                    id_path.append(graph.edge_properties['edge_attr'][edge])
                all_id_paths.add(tuple(id_path))
    return all_id_paths


def new_mq_que_proc(elems: list, ent2id: dict) -> list:
    que = elems[0].replace('[', '').replace(']', '')

    top = None
    pattern = r'\[.*\]'
    it = re.finditer(pattern, elems[0])
    for match in it:
        top = match.group()[1:-1]
    assert top in ent2id, "found no topic entity!"

    erps = []
    for word in elems[0].split(" "):
        add_flag = True
        if word not in stop_words:
            if len(erps) > 0:
                if erps[-1][0] == '[':
                    add_flag = False
            if add_flag:
                erps.append(word)
            else:
                erps[-1] += (" " + word)
            if word[-1] == "]":
                erps[-1] = erps[-1][1:-1]

    ans = []
    for an in elems[1].split("|"):
        assert an in ent2id, "found an unseen entity {}!".format(an)
        ans.append(ent2id[an])
    return que, ent2id[top], erps, ans

def subg_extract(hop: int, ent_idx: int, edge_index: LongTensor, edge_attr: LongTensor, edge_type: LongTensor) -> list:
    # 1hop_triple_ids and 1hop_neighbors
    fhop_edge_ids, fhop_neis = find_neighbors(cen_ent=ent_idx, edge_index=edge_index)
    subg = subg_con(nei_edge_ids=fhop_edge_ids, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type, num_ents=fhop_neis.size(0))
    cans = fhop_neis
    if hop == 1:
        return subg, cans

    # 2hop_triple_ids and 2hop_neighbors
    shop_edge_ids = [fhop_edge_ids]
    shop_neis = [fhop_neis]
    for ent in fhop_neis:
        tmp_edge_ids, tmp_neis = find_neighbors(cen_ent=ent, edge_index=edge_index)
        shop_edge_ids.append(tmp_edge_ids)
        shop_neis.append(tmp_neis)
    shop_edge_ids = torch.unique(torch.cat(shop_edge_ids, dim=0))
    shop_neis = torch.unique(torch.cat(shop_neis, dim=0))
    subg = subg_con(nei_edge_ids=shop_edge_ids, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type, num_ents=shop_neis.size(0))
    cans = shop_neis
    if hop == 2:
        return subg, cans

    # 3hop_triple_ids and 3hop_neighbors
    thop_edge_ids = [shop_edge_ids]
    thop_neis = [shop_neis]
    for ent in shop_neis:
        tmp_edge_ids, tmp_neis = find_neighbors(cen_ent=ent, edge_index=edge_index)
        thop_edge_ids.append(tmp_edge_ids)
        thop_neis.append(tmp_neis)
    thop_edge_ids = torch.unique(torch.cat(thop_edge_ids, dim=0))
    thop_neis = torch.unique(torch.cat(thop_neis, dim=0))
    subg = subg_con(nei_edge_ids=thop_edge_ids, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type, num_ents=thop_neis.size(0))
    cans = thop_neis
    if hop == 3:
        return subg, cans

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



