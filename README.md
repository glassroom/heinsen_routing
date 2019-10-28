# heinsen_routing

Official implementation of "[An Algorithm for Routing Capsules in All Domains](https://content.glassroom.com/An_Algorithm_for_Routing_Capsules_in_All_Domains.pdf)" (Heinsen, 2019).

## Examples

Detect objects from parts and their poses in images:

```python
import torch
from heinsen_routing import Routing

part_scores = torch.randn(100)       # 100 scores, one per detected part
part_poses = torch.randn(100, 4, 4)  # 100 capsules, each a 4 x 4 pose matrix

detect_objs = Routing(d_cov=4, d_inp=4, d_out=4, n_inp=100, n_out=10)
obj_scores, obj_poses, obj_poses_sig2 = detect_objs(part_scores, part_poses)

print(obj_scores)                    # 10 scores, one per detected object
print(obj_poses)                     # 10 capsules, each a 4 x 4 pose matrix
```

Classify sequences of token embeddings:

```python
tok_scores = torch.randn(n)          # token scores, n is variable
tok_embs = torch.randn(n, 1024)      # token embeddings, n is variable
tok_embs = tok_embs.unsqueeze(1)     # reshape to n x 1 x 1024 (n matrices)

classify = Routing(d_cov=1, d_inp=1024, d_out=8, n_out=2)  # variable n_inp
class_scores, class_embs, class_embs_sig2 = classify(tok_scores, tok_embs)

print(class_scores)                  # 2 scores, one per class
print(class_embs)                    # 2 capsules, each a 1 x 8 matrix
```

Try it on your data. You will be delightfully surprised at how well it works!

## Installation and use

1. Download one file: [heinsen_routing.py](heinsen_routing.py).
2. Import the module: `from heinsen_routing import Routing`.
3. Use it as shown above.

## Why?

Initial evaluations show that our routing algorithm, _without change_, achieves state-of-the-art results in two domains, vision and language. In our experience, this is unusual, and therefore worthy of attention and further research:

> ![Figs. 1 and 2 from paper](assets/draft_paper_fig1_and_fig2.png)

We find evidence that our routing algorithm, when we apply it to a visual recognition task, _learns to perform a form of "reverse graphics."_ The following visualization, from our [paper](https://content.glassroom.com/An_Algorithm_for_Routing_Capsules_in_All_Domains.pdf), shows a two-dimensional approximation of the trajectories of the pose vectors of an activated class capsule as we change viewpoint elevation of the same object from one image to the next:

> ![Fig. 4 from paper](assets/draft_paper_fig4.png)

Our routing algorithm is a new variant of "EM routing" ([Hinton et al., 2018](https://openreview.net/pdf?id=HJWLfGWRb)), a form of "routing by agreement" which uses expectation-maximization (EM) to cluster similar votes from input capsules to output capsules in a layer of a neural network. A capsule is a group of neurons whose outputs represent different properties of the same entity in different contexts. Routing by agreement is an iterative form of clustering in which each output capsule detects an entity by looking for agreement among votes from input capsules that have already detected parts of the entity in a previous layer.

Recent research has shown that capsule networks with routing by agreement can be more effective than convolutional neural networks for segmenting highly overlapping images ([Sabour et al., 2017](https://arxiv.org/pdf/1710.09829.pdf)) and for generalizing to different poses of objects embedded in images and resisting white-box adversarial image attacks ([Hinton et al., 2018](https://openreview.net/pdf?id=HJWLfGWRb)).

We show that capsule networks with our routing algorithm can be more effective than other models in two domains, vision and language. Our routing algorithm is readily usable in other domains too. Please see our paper for details.

## Replication of results in paper

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

# Load SST model pretrained on binary labels (needs GPT-2-large model
# too; also, make sure order of label ids matches those used in training).
model = SSTClassifier(d_depth=37, d_emb=1280, d_cap=2, n_parts=64, n_classes=2)
model.load_state_dict(torch.load('SST2R_pretrained_model_state_dict.pt'))

# Load SST model pretrained on fine-grained labels (needs GPT-2-large model
# too; also, make sure order of label ids matches those used in training).
model = SSTClassifier(d_depth=37, d_emb=1280, d_cap=2, n_parts=64, n_classes=5)
model.load_state_dict(torch.load('SST5R_pretrained_model_state_dict.pt'))
```

## Notes

Our paper is still a draft, subject to revision. We will upload it to Arxiv after receiving a bit more feedback. Comments and suggestions are welcome!  We typeset the paper with the ACL conference's LaTeX template for no other reason than we find its two-column format, with fewer words per line, easier to read. We briefly considered submitting our work to an academic conference, but by the time we had finished running experiments on academic datasets, documenting and writing up the results, and removing all traces of internal code, the deadline for NIPS had already passed and we didn't want to wait until ICML. We decided to post everything here and let the work speak for itself.

We have tested our code only on Ubuntu Linux 18.04 with Python 3.6+.

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
