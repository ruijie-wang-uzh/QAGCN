import torch
import pickle
from models import AttenModel
from datetime import datetime
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
              "cannot", "cant", "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes",
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
              "you've", "your", "yours", "yourself", "yourselves", "zero", "whose", "which", "is", ",", "\\\\", "?", "\\",
              "'s"]


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
