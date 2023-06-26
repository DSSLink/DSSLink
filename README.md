# DSSLink
Deep semi-supervised learning for recovering traceability links between issues and commits

![DSSLink.png](DSSLink.png)

### Installation
Run the following command to install software dependencies (Python >= 3.8) on a Linux platform with an Nvidia RTX 2080Ti GPU.
- Pytorch == 1.5.0
- CUDA == 10.2
- cudNN == 8.5.0
```commandline
pip install -r requirements.txt
```

### Task module introduction
This repository contains three folders, corresponding to three tasks, namely DSSLink (self-training), DSSLink (label propagation) and TraceFUN.
```
self_training
label_propagation
tracefun
```
In each task, the functions of different files are introduced as following:
```ruby
| File             | Description                   |
| ---------------- |:-----------------------------:|
| run.py           | The entry file                |
| train_stage1.py  | The original TraceBERT        |
| train_deepssl.py | Deep semi-supervised learning |
| eval_stage.py    | Used to evaluating results    |
```
#### 0. Initialization data
First unzip **DSSLink-Datasets.zip** to a suitable location. And, set the location of the datasets and the location of the outputs in **run.py**.
```python
datasets_dir = Your Datasets location
outputs_dir = Your Outputs location
```
Our dataset is divided into 5 groups by 5-fold stratified cross-validation, and each group splits the labeled data and unlabeled data in different proportions. In **run.py**, determine the data to be executed by setting the dataset name, group and proportion of labeled data. Examples are as following:
```python
dname = 'flask'
group = '1'
lab_frac = 10
```
#### 1. T-BERT
There is **train_stage1.py** in each task to execute the original T-BERT, you can choose a task at random, and in its **run.py**, set the variable **stage** to **ori**, so that the original T-BERT can be executed. Furthermore, set the variable **epochs** to determine the number of epochs to execute. In our experiments, the epochs of the original T-BERT is 200.
```python
stage = 'ori'
epochs = 200
```
#### 2. DSSLink
There is **train_deepssl.py** in each task to execute the DSSLink. In **run.py**, set the variable **stage** to **ssl**, so that the DSSLink can be executed.
- In our experiments, DSSLink first uses the labeled data to train an original T-BERT.
```python
stage = 'ori'
epochs = 100
```
- Then use the labeled and unlabeled data to train a deep semi-supervised learning model.
```python
stage = 'ssl'
epochs = 100
```
- If you need to modify the training parameters of network, such as learning rate and batch size, etc. It can be set specifically through the **train_stage1.py**, **train_deepssl.py** and **eval_stage.py** files.
- After all settings are complete, run **run.py** to start executing tasks.
#### 2. TraceFUN
- TraceFUN includes a **labeling** folder for creating new links with unlabeled data, run **run.py** in the **labeling** folder to start labeling.
> In **run.py**, the variable **frac** represents the proportion of the amount of labeled data, and the number of new links to be created is determined by setting the variable **frac**. The default values of the variable **frac** for the Flask, Pgcli and Keras projects are 1.1, 1.1 and 0.05 respectively.
- After creating new links, run **run.py** in the **TraceFUN** folder to start the TraceFUN task.

### TraceBERT
> The TLR methods (TraceBERT) evaluated in this paper can be found at [TraceBERT](https://github.com/jinfenglin/TraceBERT).
