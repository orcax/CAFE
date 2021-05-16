# [CIKM 2020] Cafe: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation

This repository contains the source code of the CIKM 2020 paper "Cafe: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation" [3].

## Data
Two Amazon datasets (Amazon_Beauty, Amazon_Cellphones) are available in the "data/" directory and the split is consistent with [2]. The pretrained KG embeddings based on [1] are also provided.

## How to Run
1. Data preprocessing.
```python
python preprocess.py --dataset <dataset_name>
```

2. Train neural-symbolic model.
```python
python train_neural_symbol.py --dataset <dataset_name> --name <model_name>
```
The model checkpoint can be located in the directory "tmp/<dataset_name>/<model_name>/symbolic_model_epoch*.ckpt".

3(a) Do path inference by the trained neural-symbolic model.
```python
python execute_neural_symbol.py --dataset <dataset_name> --name <model_name> --do_infer true
```
3(b) Execute neural program (tree layout given by user profile) for profile-guided path reasoning.
```python
python execute_neural_symbol.py --dataset <dataset_name> --name <model_name> --do_execute true
```

## References
[1] Qingyao Ai, et al. "Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation." In *Algorithms*. 2018.  
[2] Yikun Xian, et al. "Reinforcement Knowledge Graph Reasoning for Explainable Recommendation." In *SIGIR*. 2019.  
[3] Yikun Xian, et al. "Cafe: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation." In *CIKM*. 2020.  
