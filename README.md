# heinsen_routing

Official implementation of "[An Algorithm for Routing Capsules in All Domains](https://content.glassroom.com/An_Algorithm_for_Routing_Capsules_in_All_Domains.pdf)" (Heinsen, 2019).

Please note: the paper is still a draft, subject to revision. Feedback and suggestions are welcome!

## Why?

Initial evaluations show that our routing algorithm, _without change_, achieves state-of-the-art results in two domains, vision and language. In our experience, this is unusual, and therefore worthy of attention and further research:

> ![Figs. 1 and 2 from paper](assets/draft_paper_fig1_and_fig2.png)

Moreover, we find evidence that our routing algorithm, as we apply it to the smallNORB visual recognition task, _learns to perform a form of "reverse graphics."_ The following visualization, from the paper, shows a two-dimensional approximation of the trajectories of the pose vectors of an activated class capsule as we change viewpoint elevation of the same toy from one image to the next:

> ![Fig. 4 from paper](assets/draft_paper_fig4.png)

Our routing algorithm is a new variant of "EM routing" ([Hinton et al., 2018](https://openreview.net/pdf?id=HJWLfGWRb)), a form of "routing by agreement" which uses expectation-maximization (EM) to cluster similar votes from input capsules to output capsules in a layer of a neural network. A capsule is a group of neurons whose outputs represent different properties of the same entity in different contexts. Routing by agreement is an iterative form of clustering in which each output capsule detects an entity by looking for agreement among votes from input capsules that have already detected parts of the entity in a previous layer.

Recent research has shown that capsule networks with routing by agreement can be more effective than convolutional neural networks for segmenting highly overlapping images ([Sabour et al., 2017](https://arxiv.org/pdf/1710.09829.pdf)) and for generalizing to different poses of objects embedded in images and resisting white-box adversarial image attacks ([Hinton et al., 2018](https://openreview.net/pdf?id=HJWLfGWRb)).

We show that capsule networks with our routing algorithm can be more effective than other models in two domains, vision and language. Our routing algorithm is readily usable in other domains too. Please see our paper for details.

## Sample usage

We have implemented our routing algorithm as a self-contained PyTorch module in a [single file](heinsen_routing.py):

```python
from heinsen_routing import Routing

# 100 input capsules of shape 4 x 4
a_inp = torch.randn(100)         # input scores
mu_inp = torch.randn(100, 4, 4)  # input capsules

# Instantiate routing module.
m = Routing(d_cov=4, d_out=4, n_out=10, d_inp=4, n_inp=100)

# Route to 10 output capsules of shape 4 x 4
a_out, mu_out, sig2_out = m(a_inp, mu_inp)
print(mu_out)  # shape is 10 x 4 x 4
```

## Replication of Results

If you wish to replicate our results, we recommend recreating our setup in a virtual Python environment, with the same versions of all libraries and dependencies. Runing the code requires at least one Nvidia GPU with 11GB+ RAM, along with a working installation of CUDA 10 or newer. The code is meant to be easily modifiable to work with greater numbers of GPUs, or with TPUs. It is also meant to be easily modifiable to work with frameworks other than PyTorch (as long as they support Einsten summation notation for describing multilinear operations), such as TensorFlow.

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

```
jupyter notebook
```

Make sure the virtual environment is activated before you do this.

You should see two notebooks that replicate the results in our paper. Open and run them.

## Pretrained weights

We have made pretrained weights available for the smallNORB and SST models:

```python
import torch
from models import SmallNORBClassifier, SSTClassifier

# Load pretrained smallNORM model.
model = SmallNORBClassifier(n_objs=5, n_parts=64, d_chns=64)
model.load_state_dict(torch.load('smallNORB_pretrained_model_state_dict.pt'))

# Load SST model pretrained on binary labels
# (make sure order of label ids match those we used in training).
model = SSTClassifier(d_depth=37, d_emb=1280, d_cap=2, n_parts=64, n_classes=2)
model.load_state_dict(torch.load('SST2R_pretrained_model_state_dict.pt'))

# Load SST model pretrained on fine-grained labels
# (make sure order of label ids match those we used in training).
model = SSTClassifier(d_depth=37, d_emb=1280, d_cap=2, n_parts=64, n_classes=5)
model.load_state_dict(torch.load('SST5R_pretrained_model_state_dict.pt'))
```

## Notes

We have tried to optimize our code for clarity and brevity, so we have abstained from adding many nice-to-have features that would have increased the cognitive effort required to understand our routing algorithm and the models that use it.

Our draft paper is typeset with the ACL conference's LaTeX template, for no other reason than we find its two-column format, with fewer words per line, easier to read.

We briefly considered submitting this work to an academic conference, but by the time we had finished running experiments on academic datasets, documenting and writing up the results in a paper, and removing all traces of internal code, the deadline for NIPS had already passed and we didn't want to wait until ICML. We decided to post everything here and let the work speak for itself.

Finally, we have tested our code only on Ubuntu Linux 18.04 with Python 3.6+.

## Citing

If our work is helpful to your research, please cite it:

```
@article{HeinsenRouting2019,
    title	= {An Algorithm for Routing Capsules in All Domains},
    author	= {Franz A. Heinsen},
    year	= {2019},
}
```

## How is this used at GlassRoom?

We conceived and implemented this routing algorithm to be a component (i.e., a layer) of larger models that are in turn part of our AI software, nicknamed Graham. Our implementation of the algorithm is designed to be plugged into or tacked onto existing PyTorch models with minimal hassle.

Most of the original work we do at GlassRoom tends to be either proprietary in nature or tightly coupled to internal code, so we cannot share it with outsiders. In this case, however, we were able to isolate our code and release it as stand-alone open-source software without having to disclose any key intellectual property.

We hope others find our work and our code useful.
