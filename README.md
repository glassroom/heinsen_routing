# heinsen_routing

Official implementation of "Routing Capsules by Their Net Cost to Use or Ignore, with Sample Applications in Vision and Language" (Heinsen, 2019). The code in this repository uses our routing algorithm, _without change_, in two different domains, vision and language, and achieves state-of-the-art results in both. Please see the paper for details.

The algorithm is implemented as a composable PyTorch module.

## Why?

This is the first time we have produced original research that:

* we believe could be of value to the wider AI research community, to whom we are incredibly grateful for all the work they have done and made public and/or released as open-source software over the years, and

* we could clean up of internal code, apply to public academic datasets, and release as open-source code, without having to disclose any proprietary intellectual property we consider valuable.

We hope others find our work useful.

## How do you use this routing algorithm at GlassRoom?

We conceived and implemented this routing algorithm to be a component (i.e., a layer) of larger models that are in turn part of our AI software, Graham. The algorithm is designed to be plugged into or tacked unto any model. That said, we can neither confirm nor deny that we are using this routing algorithm as part of any of Graham's models.

We regularly replicate new AI research that we find interesting or promising. Soon after we came across the routing-by-agreement algorithm Geoff Hinton et al proposed in their recent paper, "Matrix Routing with EM Capsules," we decided to replicate it, and then, as always, started thinking of ways to improve and extend it. The result is the routing algorithm in this repository. Please see the paper for details.

## Installation

If you wish to replicate our results, we recommend recreating our setup in a virtual environment, with the same versions of all libraries and dependencies. Note: the code in this repository requires at least one higher-end (at least 11GB RAM) Nvidia GPU with a working installation of CUDA 10 or later. Tested only on Ubuntu Linux 18.04 with Python 3.6+.

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
source ./python/bin/activate`
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

6. Download and gunzip smallNORB files:

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


