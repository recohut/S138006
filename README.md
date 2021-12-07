## Sign-Aware Recommendation Systems with Graph Neural Networks (SiReN)

### This is PyTorch implementation code for paper:

> C. Seo et al., SiReN: Sign-Aware Recommendation Using Graph Neural Networks, 
>
> [Paper in arXiv](https://arxiv.org/abs/2108.08735)


### Example : ML-1M dataset

```python
python main.py --dataset ML-1M --version 1 --reg 0.1
```

### Example : Amazon-Book dataset

```python
python main.py --dataset amazon --version 1 --reg 0.05
```

### Example : Yelp dataset

```python
python main.py --dataset yelp --version 1 --reg 0.05
```

### Tree
```
.
├── [1.2K]  convols.py
├── [2.4K]  data_loader.py
├── [5.5K]  evaluator.py
├── [342K]  images
│   ├── [ 47K]  aggregation.png
│   ├── [123K]  algorithm.png
│   ├── [ 29K]  lightgcn.png
│   ├── [ 32K]  lrgccf.png
│   └── [106K]  model.png
├── [5.7K]  main.py
├── [392K]  nbs
│   ├── [165K]  T087038_Sign_Aware_Recommendation_Using_Graph_Neural_Networks_on_ML_1m_Dataset_in_PyTorch.ipynb
│   ├── [138K]  T158246_Sign_Aware_Recommendation_Using_Graph_Neural_Networks_on_Yelp_Dataset_in_PyTorch.ipynb
│   └── [ 85K]  T890631_Sign_Aware_Recommendation_Using_Graph_Neural_Networks_on_Amazon_Books_Dataset_in_PyTorch.ipynb
├── [ 869]  README.txt
├── [3.4K]  siren.py
└── [3.4K]  util.py

 760K used in 2 directories, 15 files
```

### Links
- https://github.com/woni-seo/siren-reco
