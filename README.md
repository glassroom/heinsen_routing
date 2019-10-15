# heinsen_routing

Official implementation of "Routing Capsules by Their Net Cost to Use or Ignore, with Sample Applications in Vision and Language" (Heinsen, 2019). Please note: the paper is still a draft subject to revision.

## Why?

Initial evaluations show that our routing algorithm, _without change_, achieves state-of-the-art results in two domains, vision and language. In our experience, this is unusual, and therefore worthy of attention and further research.

Our routing algorithm is a new variant of "EM routing" ([Hinton et al., 2018](https://openreview.net/pdf?id=HJWLfGWRb)), a form of "routing by agreement" which uses expectation-maximization (EM) to cluster similar votes from input capsules to output capsules in a layer of a neural network. A capsule is a group of neurons whose outputs represent different properties of the same entity in different contexts. Routing by agreement is an iterative form of clustering in which each output capsule detects an entity by looking for agreement among votes from input capsules that have already detected parts of the entity in a previous layer.

Recent research has shown that capsule networks with routing by agreement can be more effective than convolutional neural networks for segmenting highly overlapping images ([Sabour et al., 2017](https://arxiv.org/pdf/1710.09829.pdf)) and for generalizing to different poses of objects embedded in images and resisting white-box adversarial image attacks ([Hinton et al., 2018](https://openreview.net/pdf?id=HJWLfGWRb)).

We show that capsule networks with our routing algorithm can be more effective than other models in two domains, vision and language. Please see our paper for details.

## Sample usage

Our routing algorithm is implemented as a self-contained (one file) composable PyTorch module:

```python
from heinsen_routing import Routing

# 100 input capsules of shape 4 x 4
a_inp = torch.randn(100)                    # 100 input scores
mu_inp = torch.randn(100, 4, 4)             # 100 capsules of shape 4 x 4

# Instantiate routing module.
m = Routing(d_spc=4, d_out=4, n_out=10, d_inp=4, n_inp=100)

# Route to 10 capsules of shape 4 x 4
a_out, mu_out, sig2_out = m(a_inp, mu_inp)
print(mu_out)                               # 10 capsules of shape 4 x 4
```

## Installation and Replication of Results

If you wish to replicate our results, we recommend recreating our setup in a virtual environment, with the same versions of all libraries and dependencies. Runing the code requires _at least one_ Nvidia GPU with 11GB+ RAM, along with a working installation of CUDA 10 or newer. The code is meant to be easily modifiable to work with greater numbers of GPUs, or with TPUs. The code is also meant to be easily modifiable to work with frameworks other than PyTorch (as long as they support Einsten summation notation for describing multilinear operations), such as TensorFlow.

To replicate our environment and results, follow these steps:

1. Change to the directory in which you cloned this repository:

```
cd /home/<my_name>/<my_directory>
```

2. Create a new Python 3 virtual environment:

```
virtualenv --python=python3 python
```

3. Activate the virtual environment:

```
source ./python/bin/activate
```

4. Install required Python libraries in environment:

```
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

5. Install other dependencies:

```
mkdir deps
git clone https://github.com/glassroom/torch_train_test_loop.git deps/torch_train_test_loop
git clone https://github.com/ndrplz/small_norb.git deps/small_norb
```

6. Download and decompress smallNORB files:

```
mkdir .data
mkdir .data/smallnorb
cd .data/smallnorb
wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
for FILE in *.gz; do gunzip -k $FILE; done
cd ../..
```

7. Launch Jupyter notebook:

`jupyter notebook`

You should see two notebooks that replicate the results in our paper. Open and run them.

Please note: tested only on Ubuntu Linux 18.04 with Python 3.6+.

## Citing

If our work is helpful to your research, please cite it:

```
@misc{HeinsenRouting2019,
    title	= {Routing Capsules by Their Net Cost to Use or Ignore, with Sample Applications in Vision and Language},
    author	= {Franz A. Heinsen},
    year	= {2019},
}
```

## How is this used at GlassRoom?

We conceived and implemented this routing algorithm to be a component (i.e., a layer) of larger models that are in turn part of our AI software, nicknamed Graham. Our code is designed to be plugged into or tacked unto any model.

Most of the work we do at GlassRoom tends to be either proprietary in nature or tightly coupled to internal code, so we cannot share it with others. This is the first time we have something which (a) we believe is valuable to others, (b) we can release as stand-alone open-source software without disclosing any valuable intellectual property, and (c) is new, original AI research. We hope others find it useful.
