# Sentence-State LSTM in Pytorch 

This implementation only focus on the classification task in the [paper](https://arxiv.org/pdf/1805.02474.pdf).


## Getting started
Get text classification dataset mentioned in paper from [HERE](http://nlp.fudan.edu.cn/data/). 
Place the dataset files at folder ```mtl_data``` and the Glove vectors files at folder ```embedding```.

1. Preprocess data 

```
python mlt_preprocessing.py
```


2. Train and evaluate the model with

```
python train.py --gpu device_id
```

example:

```
python train.py --gpu 0
```

To change hyperparameters of model, please modify config.py