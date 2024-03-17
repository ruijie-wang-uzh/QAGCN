import torch
import torch.nn as nn
import torch.optim as optim
from models import BoWClassifier
from torch.utils.data import DataLoader
from utils import QueSet, read_ques, collate_fn, evaluate, read_obj, write_obj

dataset = 'p'
hops = [2, 3]
in_path = "../data/pathquestion/output/X-hop/XH-Y.txt"
pre_data_path = "../data/pathquestion/output/hop_prep_data.pickle"
model_path = "../data/pathquestion/output/models/pq_hop_prep.pt"
dict_path = "../data/pathquestion/output/dict.pickle"
discard = 15
lr = 0.0005
batch_size = 16
num_epochs = 50
device = torch.device('cuda:0')
encode_ques = True

if encode_ques:
    num_words = 2
    train_set = QueSet()
    valid_set = QueSet()
    test_set = QueSet()

    for hop in hops:
        read_ques(path=in_path, hop=hop, dataset=dataset, name='train', queset=train_set)
        read_ques(path=in_path, hop=hop, dataset=dataset, name='valid', queset=valid_set)
        read_ques(path=in_path, hop=hop, dataset=dataset, name='test', queset=test_set)

    seen_words = {}
    for que in train_set.que2id.keys():
        for word in que.split(' '):
            if word not in seen_words:
                seen_words[word] = 0
            else:
                seen_words[word] += 1

    sort_words = [(k, v) for k, v in seen_words.items()]
    sort_words = sorted(sort_words, key=lambda tup: tup[1])
    sort_words = [_[0] for _ in sort_words[:discard]]

    word2idx = {'num_words': 0, 'out_of_voc': 1}
    for word in seen_words.keys():
        if word not in sort_words:
            word2idx[word] = num_words
            num_words += 1
    print('discarded words: {}'.format(sort_words))
    print('* number of words: {}, vocabulary size: {}'.format(len(seen_words), num_words))

    train_set.encode(word2idx=word2idx, device=device)
    valid_set.encode(word2idx=word2idx, device=device)
    test_set.encode(word2idx=word2idx, device=device)

    write_obj(obj=[train_set, valid_set, test_set, num_words], file_path=pre_data_path)
    write_obj(obj=word2idx, file_path=dict_path)
else:
    train_set, valid_set, test_set, num_words = read_obj(file_path=pre_data_path)

classifier = BoWClassifier(vocab_size=num_words, num_hops=len(hops))
classifier.to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(classifier.parameters(), lr=lr)

train_loader = DataLoader(dataset=train_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

acc = 0.
for epoch in range(num_epochs):
    print('* epoch - {}'.format(epoch))
    epoch_loss = 0.
    classifier.train()
    for batch_id, batch_data in enumerate(train_loader):
        optimizer.zero_grad()

        que_vecs = torch.stack([_ for _ in batch_data[0]], dim=0).to(device)
        hop_target = torch.LongTensor(batch_data[1]).to(device)
        hop_pred = classifier(que_vecs=que_vecs)
        batch_loss = criterion(hop_pred, hop_target)
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    v_acc = 100 * evaluate(model=classifier, loader=valid_loader, device=device) / valid_set.num_ques
    if acc <= v_acc:
        acc = v_acc
        torch.save(classifier.state_dict(), model_path)
        print('\t model saved to {}'.format(model_path))
    print('\t * loss: {}, valid acc: {}%'.format(epoch_loss, acc))

test_loader = DataLoader(dataset=test_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
test_classifier = BoWClassifier(vocab_size=num_words, num_hops=len(hops))
test_classifier.to(device)
test_classifier.load_state_dict(torch.load(model_path, map_location=device))
print('* test acc: {}%'.format(100 * evaluate(model=test_classifier, loader=test_loader, device=device)/test_set.num_ques))
