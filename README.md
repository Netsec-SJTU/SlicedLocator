# SlicedLocator
Source code of our paper accepted by Computers & Security: [SlicedLocator: Code Vulnerability Locator Based on Sliced Dependence Graph](https://doi.org/10.1016/j.cose.2023.103469).


# Requirements
```
pytorch                   1.11.0
torch-geometric           2.0.5
tree-sitter               0.20.0
pytorch-lightning         1.6.4 
torchmetrics              0.9.0
scikit-learn              1.1.1
```

# Dataset
- Baidu Netdisk
[https://pan.baidu.com/s/16znz2dxF5MxbcQj8EmyyBg ](https://pan.baidu.com/s/16znz2dxF5MxbcQj8EmyyBg) 提取码: gupr

# Citation

```
@article{WU2023103469,
title = {SlicedLocator: Code Vulnerability Locator Based on Sliced Dependence Graph},
journal = {Computers & Security},
pages = {103469},
year = {2023},
issn = {0167-4048},
doi = {https://doi.org/10.1016/j.cose.2023.103469},
url = {https://www.sciencedirect.com/science/article/pii/S0167404823003796},
author = {Bolun Wu and Futai Zou and Ping Yi and Yue Wu and Liang Zhang},
keywords = {vulnerability detection, localization, program analysis, program representation, deep learning},
abstract = {Machine learning-based fine-grained vulnerability detection is an important technique for locating vulnerable statements, which assists engineers in efficiently analyzing and fixing the vulnerabilities. However, due to insufficient code representations, code embeddings, and neural network design, current methods suffer low vulnerability localization performance. In this paper, we propose to address these shortcomings by presenting SlicedLocator, a novel fine-grained code vulnerability detection model that is trained in a dual-grained manner and can predict both program-level and statement-level vulnerabilities. We design the sliced dependence graph, a new code representation that not only preserves rich interprocedural relations but also eliminates vulnerability-irrelevant statements. We create attention-based code embedding networks that are trained with the entire model to extract vulnerability-aware code features. In addition, we present a new LSTM-GNN model as a fusion of semantic modeling and structural modeling. Experiment results on a large-scale C/C++ vulnerability dataset reveal that SlicedLocator outperforms state-of-the-art machine learning-based vulnerability detectors, especially in terms of localization metrics.}
}
```
