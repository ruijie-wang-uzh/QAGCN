import torch
import pickle
from datetime import datetime
from models import BoWClassifier
from torch.utils.data import Dataset, DataLoader


class QueSet(Dataset):
    def __init__(self):
        super(QueSet, self).__init__()
        self.num_ques = 0
        self.que2id = {}

        self.qid2hop = {}
        self.qid2vec = {}

    def add(self, que: str, hop: int, bias: int) -> None:
        if que not in self.que2id:
            self.que2id[que] = self.num_ques
            self.num_ques += 1
        if self.que2id[que] in self.qid2hop.keys():
            if self.qid2hop[self.que2id[que]] != hop - bias:
                print('que {} from {} to {}'.format(que, self.qid2hop[self.que2id[que]], hop - bias))
        self.qid2hop[self.que2id[que]] = hop - bias

    def encode(self, word2idx: dict, device: torch.device) -> None:
        for que, qid in self.que2id.items():
            self.qid2vec[qid] = bow_vec_convert(que=que, word2idx=word2idx, device=device)

    def __len__(self) -> int:
        return self.num_ques

    def __getitem__(self, item) -> int:
        return self.qid2vec[item], self.qid2hop[item], item


def collate_fn(batch_data: list):
    return [_[0] for _ in batch_data], [_[1] for _ in batch_data], [_[2] for _ in batch_data]


def read_ques(path: str, hop: int, dataset: str, name: str, queset: QueSet) -> None:
    name = 'dev' if dataset == 'm' and name == 'valid' else name
    path = path.replace('X', str(hop)).replace('Y', str(name))
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            que = line.rstrip().split('\t')[0]

            tmp_que = []
            if dataset == 'm':
                flag = True
                for word in que.split(' '):
                    if word not in stop_words:
                        if word[0] == '[':
                            flag = False
                        if flag:
                            tmp_que.append(word)
                        if word[-1] == ']':
                            flag = True
            if dataset == 'p':
                top = line.rstrip().split('\t')[2].split("#")[0]
                for word in que.split(' '):
                    if word not in stop_words and word != top:
                        tmp_que.append(word)

            que = ' '.join(tmp_que)
            queset.add(que=que, hop=hop, bias=1 if dataset == 'm' else 2)


def bow_vec_convert(que: str, word2idx: dict, device: torch.device):
    vec = torch.zeros(len(word2idx)).to(device)
    for word in que.split(' '):
        if word in word2idx:
            vec[word2idx[word]] += 1
        else:
            vec[1] += 1
        vec[0] += 1
    return vec


def evaluate(model: BoWClassifier, loader: DataLoader, device: torch.device):
    model.eval()
    with torch.no_grad():
        num_corr = 0
        for _, batch_data in enumerate(loader):
            que_vecs = torch.stack([_ for _ in batch_data[0]], dim=0).to(device)
            hop_target = torch.LongTensor(batch_data[1]).to(device)
            hop_pred = model(que_vecs=que_vecs)
            num_corr += torch.nonzero(hop_target == torch.argmax(hop_pred, dim=1)).size()[0]
    return num_corr

def get_time() -> str:
    return datetime.now().strftime('%H:%M:%S %Y-%m-%d')


def write_obj(obj: object, file_path: str) -> None:
    print('\t* dumping {} at {} ...'.format(file_path, get_time()))
    with open(file=file_path, mode='wb') as f:
        pickle.dump(obj=obj, file=f, protocol=4)


def read_obj(file_path: str) -> dict:
    print('\t* loading {} at {} ...'.format(file_path, get_time()))
    with open(file=file_path, mode='rb') as f:
        obj = pickle.load(file=f)
    return obj


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
              "nowhere", "obviously", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only",
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
              "those", "though", "three", "through", "throughout", "thru", "thus", "to", "too", "took", "toward",
              "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "un", "under", "unfortunately", "unless",
              "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "value",
              "various", "very", "via", "viz", "vs", "want", "wants", "was", "wasn't", "way", "we", "wed", "well", "were",
              "we've", "welcome", "well", "went", "were", "weren't", "what", "whats", "whatever", "when", "whence", "whenever",
              "where", "wheres", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which",
              "while", "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with",
              "within", "without", "wont", "wonder", "would", "would", "wouldn't", "yes", "yet", "you", "you'd", "you'll", "you're",
              "you've", "your", "yours", "yourself", "yourselves", "zero", "whose", "which", "is", ",", "\\\\", "?", "\\",
              "'s", ""]
