# HyperGraph Contrastive Learning for Recommendation
This is our official implementation for the paper:

Yuma Dose, Shuichiro Haruta, Yihong Zhang, and Takahiro Hara\\
"Hypergraph Contrastive Learning with Graph Structure Learning for Recommendation"\\
Proceedings of IEEE International Conference on Machine Learning and Applications (ICMLA), pages 416-423, 2024.

## Environment
The codes of HGCL are implemented and tested under the following development environment:

```
pyTorch:
* python=3.10.4
* torch=1.11.0
* numpy=1.22.3
* scipy=1.7.3

```

### For pyTorch
For the dataset used in this paper, we recommend the following configurations:

* Yelp
```
python Main.py --data yelp 
```
* gowalla
```
python Main.py --data gowalla
```
* Amazon
```
python Main.py --data amazon
```


## Acknowledgements
This research is supported by the Graduate School of Information Science and Technology, the University of Osaka.

