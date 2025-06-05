
# HyperGraph Contrastive Learning for Recommendation

```

## Environment
The codes of HGCL are implemented and tested under the following development environment:

TensorFlow:
* python=3.6.12
* tensorflow=1.14.0
* numpy=1.16.0
* scipy=1.5.2

pyTorch:
* python=3.10.4
* torch=1.11.0
* numpy=1.22.3
* scipy=1.7.3

```

### For pyTorch
Switch your working directory to ```torchVersion/```, run ```python Main.py```. The implementation has been improved in the torch code. You may need to adjust the hyperparameter settings. If you want to run HCCF on other datasets, we suggest you consider using a simplified version `torchVersion/Model_sparse.py` if your dataset is sparse. To do so, you should change the imported module in `torchVersion/Main.py` from `Model` to `Model_sparse`. For the dataset used in this paper, we recommend the following configurations:

* Yelp
```
python Main.py --data yelp 

* Gowalla
```
python Main.py --data gowalla

* Amazon
```
python Main.py --data amazon
```


## Acknowledgements
This research is supported by the research grants from the Department of Computer Science & Musketeers Foundation Institute of Data Science at the University of Hong Kong, the Natural Sciences & Engineering Research Council (NSERC) of Canada.
