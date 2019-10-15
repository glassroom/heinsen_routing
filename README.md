# heinsen_routing

Official implementation of "Routing Capsules by Their Net Cost to Use or Ignore, with Sample Applications in Vision and Language" (Heinsen, 2019). The code in this repository uses our routing algorithm, _without change_, in two different domains, vision and language, and achieves state-of-the-art results in both. Please see the paper for details.

The algorithm is implemented as a composable PyTorch module.

## Why?

Most of the research we do at GlassRoom tends to be proprietary in nature or tightly coupled to internal code, so we cannot share it with others. This is the first time we have worked on something that (a) we believe is valuable to others, (b) we can release as stand-alone open-source software, and (c) is new, original AI research.

Over the years, we have benefited immensely from, and feel incredibly thankful for, the work of a large number of AI researchers and practitioners who have made their work public or released it as open-source software.

We hope others find our work useful.

## How do you use this routing algorithm at GlassRoom?

We conceived and implemented this routing algorithm to be a component (i.e., a layer) of larger models that are in turn part of our AI software, Graham. The algorithm is designed to be plugged into or tacked unto any model. That said, we can neither confirm nor deny that we are using this exact same algorithm in any of Graham's models.

## Installation

If you wish to replicate our results, we recommend recreating our setup in a virtual environment, with the same versions of all libraries and dependencies. The code requires _at least one_ Nvidia GPU with 11GB+ RAM with a working installation of CUDA 10 or newer. The code is meant to be easily modifiable to work with greater numbers of GPUs, or with TPUs. It is also meant to be easily modifiable to work with frameworks other than PyTorch, e.g., TensorFlow. Please note: Tested only on Ubuntu Linux 18.04 with Python 3.6+.

To replicate our environment, follow these steps:

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


## Sample usage

```python
from heinsen_routing import Routing

# 100 input capsules of shape 4 x 4
x = torch.rand(100, 4, 4)

# Route to 10 capsules of shape 4 x 4.
m = Routing(d_spc=4, d_out=4, n_out=10, d_inp=4, n_inp=100)
h = m(x)
```


## Citing

If our work is helpful to your research, please cite it:

```
@misc{HeinsenRouting2019,
    title	= {Routing Capsules by Their Net Cost to Use or Ignore, with Sample Applications in Vision and Language},
    author	= {Franz A. Heinsen},
    year	= {2019},
}
```
