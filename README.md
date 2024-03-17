# QAGCN: Answering Multi-Relation Questions via Single-Step Implicit Reasoning over Knowledge Graphs

Ruijie Wang, Luca Rossetto, Michael Cochez, and Abraham Bernstein

To appear at the [21st European Semantic Web Conference (ESWC 2024)](https://2024.eswc-conferences.org/).

Arxiv preprint: [QAGCN: Answering Multi-Relation Questions via Single-Step Implicit Reasoning over Knowledge Graphs](https://arxiv.org/abs/2206.01818).

<img src="https://github.com/ruijie-wang-uzh/QAGCN/blob/main/system.png" width=70%>

----

## Environment Setup

Please set up a [Python](https://www.python.org/) environment with
[Pytorch](https://pytorch.org/), 
[Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/),
[Pytorch Scatter](https://pytorch-scatter.readthedocs.io/en/latest/),
[Transformers](https://huggingface.co/docs/transformers/index),
[NetworkX](https://networkx.org/),
and [Graph-tool](https://graph-tool.skewed.de/) installed.

---

## Data and Models

Please download the prepared data and our pretrained models for [MetaQA](https://github.com/yuyuz/MetaQA), [PathQuestion (PQ)](https://github.com/zmtkeke/IRN), and [PathQuestion-Large (PQL)](https://github.com/zmtkeke/IRN) from [OSF](https://osf.io/9gnsr/?view_only=331484b71b144a168c31e11b0193381f). (Unzip `data.zip` and move it to the root directory.)

---

## Inference using Pre-trained Models

The following commands can be used to load and evaluate the pre-trained models.

### MetaQA
```shell
cd qa_metaqa
```

```shell
# evaluate the pre-trained model on MetaQA-1hop
touch MetaQA_1hop_eval.log
python -u main.py --num_epochs 0 --path_align_timestamp 2022.02.12.20.00 --timestamp 2022.02.13.15.09 >> MetaQA_1hop_eval.log
```

```shell
# evaluate the pre-trained model on MetaQA-2hop
touch MetaQA_2hop_eval.log
python -u main.py --qa_type 2-hop --in_dims 768 512 --out_dims 512 256 --dropouts 0.1 0. --rerank_top 200 --num_epochs 0 --path_align_timestamp 2022.02.12.20.18 --timestamp 2022.02.15.21.17 >> MetaQA_2hop_eval.log
```

```shell
# evaluate the pre-trained model on MetaQA-3hop
touch MetaQA_3hop_eval.log
python -u infer.py --path_align_timestamp 2022.02.15.19.30 --timestamp 2022.02.19.12.28 >> MetaQA_3hop_eval.log
```

### PathQuestion
```shell
cd qa_pq
```

```shell
# evaluate the pre-trained model on PQ-2hop
touch PathQuestion_2hop_eval.log
python -u main.py --num_epochs 0 --path_align_timestamp 2022.02.20.20.06 --timestamp 2022.02.20.20.55 >> PathQuestion_2hop_eval.log
```

```shell
# evaluate the pre-trained model on PQ-3hop
touch PathQuestion_3hop_eval.log
python -u main.py --qa_type 3-hop --num_epochs 0 --in_dims 768 512 256 --out_dims 512 256 128 --dropouts 0.1 0.1 0. --rerank_top 50 --path_align_timestamp 2022.03.13.18.11 --timestamp 2022.02.19.23.45 >> PathQuestion_3hop_eval.log
```


### PathQuestion-Large

```shell
cd qa_pql
```

```shell
# evaluate the pre-trained model on PQL-2hop
touch PathQuestionLarge_2hop_eval.log
python -u main.py --num_epochs 0 --path_align_timestamp 2023.12.08.02.04 --timestamp 2023.12.08.03.10 >> PathQuestionLarge_2hop_eval.log
```

```shell
# evaluate the pre-trained model on PQL-3hop
touch PathQuestionLarge_3hop_eval.log
python -u main3.py --num_epochs 0 --path_align_timestamp 2023.12.08.06.07 --timestamp 2023.12.08.06.45 >> PathQuestionLarge_3hop_eval.log
```

---

## Training New Models

The following commands can be used to train new models.

Please take a note of the timestamp of `path_train.py` and fill it in the placeholder `[timestamp]` below for running `main.py`.

### MetaQA
```shell
cd qa_metaqa
```

```shell
# train the model on MetaQA-1hop (including data preprocessing and training)
touch MetaQA_1hop_train.log
python -u kg_prep.py >> MetaQA_1hop_train.log
python -u que_prep.py --qa_type 1-hop >> MetaQA_1hop_train.log
python -u path_train.py >> MetaQA_1hop_train.log
python -u main.py --path_align_timestamp [timestamp] >> MetaQA_1hop_train.log
```

```shell
# train the model on MetaQA-2hop (including data preprocessing and training)
touch MetaQA_2hop_train.log
python -u que_prep.py --qa_type 2-hop >> MetaQA_2hop_train.log
python -u path_train.py --qa_type 2-hop --lr 5e-4 >> MetaQA_2hop_train.log
python -u main.py --qa_type 2-hop --path_align_timestamp [timestamp] --in_dims 768 512 --out_dims 512 256 --dropouts 0.1 0. --rerank_top 200 --lr 5e-4 >> MetaQA_2hop_train.log
```

```shell
# train the model on MetaQA-3hop (including data preprocessing and training)
touch MetaQA_3hop_train.log
# please note that the path extraction for 3-hop questions in que_prep.py is memory-consuming. it is recommended to run it on a machine with at least 500GB RAM. 
# (due to the memory swapping mechanism of linux, running on a machine with insufficient RAM may cause the machine to be unresponsive!!!)
python -u que_prep.py --qa_type 3-hop >> MetaQA_3hop_train.log
python -u path_train.py --qa_type 3-hop --out_dim 128 --lr 1e-3 >> MetaQA_3hop_train.log
python -u main.py --qa_type 3-hop --path_align_timestamp [timestamp] --in_dims 768 512 256 --out_dims 512 256 128 --dropouts 0.1 0.1 0. --rerank_top 8000 --lr 1e-3 >> MetaQA_3hop_train.log
```


### PathQuestion
```shell
cd qa_pq
```

```shell
# train the model on PQ-2hop (including data preprocessing and training)
touch PathQuestion_2hop_train.log
python -u kg_prep.py >> PathQuestion_2hop_train.log
python -u que_prep.py >> PathQuestion_2hop_train.log
python -u path_train.py >> PathQuestion_2hop_train.log
python -u main.py --path_align_timestamp [timestamp] >> PathQuestion_2hop_train.log
```

```shell
# train the model on PQ-3hop (including data preprocessing and training)
touch PathQuestion_3hop_train.log
python -u kg_prep.py --qa_type 3-hop >> PathQuestion_3hop_train.log
python -u que_prep.py --qa_type 3-hop >> PathQuestion_3hop_train.log
python -u path_train.py --qa_type 3-hop --in_dim 768 --out_dim 128 --lr 1e-3 >> PathQuestion_3hop_train.log
python -u main.py --qa_type 3-hop --path_align_timestamp [timestamp] --in_dims 768 512 256 --out_dims 512 256 128 --dropouts 0.1 0.1 0. --lr 1e-3 --rerank_top 50 >> PathQuestion_3hop_train.log
```

### PathQuestion-Large

```shell
cd qa_pql
```

```shell
# train the model on PQL-2hop (including data preprocessing and training)
touch PathQuestionLarge_2hop_train.log
python -u kg_prep.py >> PathQuestionLarge_2hop_train.log
python -u que_prep.py >> PathQuestionLarge_2hop_train.log
python -u path_train.py >> PathQuestionLarge_2hop_train.log
python -u main.py --path_align_timestamp [timestamp] >> PathQuestionLarge_2hop_train.log
```

```shell
# re-train the model on PQL-3hop (including data preprocessing and training)
touch PathQuestionLarge_3hop_train.log
python -u kg_prep.py --qa_type 3-hop >> PathQuestionLarge_3hop_train.log
python -u que_prep.py --qa_type 3-hop >> PathQuestionLarge_3hop_train.log
python -u path_train.py --qa_type 3-hop --out_path ../data/pql-3hop/output --out_dim 128 --lr 5e-4 >> PathQuestionLarge_3hop_train.log
python -u main3.py --path_align_timestamp [timestamp] >> PathQuestionLarge_3hop_train.log
```

## Question Classification

The following commands can be used to train the question classifier that predicts the complexity of questions (i.e., 1, 2, or 3 hops).

```shell
cd question_classifier
```


```shell
# train and evaluate the classifier on MetaQA
touch MetaQA_classifier_train.log
python hop_pred_m.py >> MetaQA_classifier_train.log

# train and evaluate the classifier on PathQuestion
touch PathQuestion_classifier_train.log
python hop_pred_pq.py >> PathQuestion_classifier_train.log

# train and evaluate the classifier on PathQuestionLarge
touch PathQuestionLarge_classifier_train.log
python hop_pred_pql.py >> PathQuestionLarge_classifier_train.log
```

---

### Citation
```
@misc{wang2023qagcn,
      title={QAGCN: Answering Multi-Relation Questions via Single-Step Implicit Reasoning over Knowledge Graphs}, 
      author={Ruijie Wang and Luca Rossetto and Michael Cochez and Abraham Bernstein},
      year={2023},
      eprint={2206.01818},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```



