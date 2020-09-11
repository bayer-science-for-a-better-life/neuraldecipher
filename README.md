# Neuraldecipher
Implementation of the Paper "Neuraldecipher - Reverse-engineering extended-connectivity fingerprints (ECFPs) to their molecular structures" by Tuan Le, Robin Winter, Frank Noé and Djork-Arné Clevert.<sup>1</sup>

![workflow](https://github.com/bayer-science-for-a-better-life/neuraldecipher/blob/master/figures/workflow.png "Workflow")

## Installation
### Prerequisites: python==3.6.10
```
rdkit==2020.03.2
numpy==1.18.1
tqdm==4.46.1
h5py==2.10.0
jupyter==1.0.0
```

### Conda
Create a new enviorment:

```
git clone URL
cd neuraldecipher
conda env create -f environment.yml
conda activate neuraldecipher
```
### `pytorch==1.4.0` (GPU with cuda10 or CPU)
```
conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch # GPU
# conda install pytorch==1.4.0 torchvision cpuonly -c pytorch # CPU
```

### Dependency for encoding and decoding SMILES representations
* [cddd](https://github.com/jrwnter/cddd "Continuous and Data-Driven Descriptors (CDDD)")  
To complete the reverse-engineering workflow, the decoder network from Winter et al. (see Workflow) is needed in the
final evaluation.  Note, it suffices to clone the `cddd` repository and start from the installation of `tensorflow-gpu==1.10.0` without creating the environment. It is important to have the `cddd` module installed within the `neuraldecipher` environment for latter inference.
To use tensordboard with pytorch, remove the `tensorboard==1.10.0` from the cddd dependency   
```pip uninstall tensorboard```  
```pip install tensorboard==1.14.0```  
We included this workaround to still be able to use the CDDD inference server and tensorboard to log the training of the Neuraldecipher.  
The CDDD server is also needed to compute the CDDD vector representation from the SMILES to train the Neuraldecipher.  
We provided a Jupyter Notebook in `source/get_cddd.ipynb` to compute the CDDD representations from the ChEMBL25 dataset.

## Repository structure
The repository consists of several subdirectories:
- `data` consists of the training and test data.
- `logs` consists of the tensorboard log files for each training run
- `params` consists of the json parameter files for each run. See [example](https://github.com/tuanle618/neuraldecipher/blob/master/params/1024_config_bit_gpu.json "example param file for ECFP6_1024 Bits training").
- `models` consists of the saved models. In case the Neuraldecipher was trained on bit-ECFPs, the results are saved in `models/bits_results`. Otherwise the models are saved in `models`.
- `source` consists of all necessary python scripts for execution.
  
The provided data consists of:
* `data/smiles.npy`: List of SMILES from the filtered ChEMBL25 database saved as numpy array.
* `data/smiles_temporal.npy`: List of temporal SMILES from the filtered ChEMBL26 database saved as numpy array.
* `data/cluster.npy`: List of cluster assignment from the `smiles.npy` array. This array is needed to create train and validation datasets.

## Getting started

#### Computing several extended-connectivity fingerprints (ECFPs) depending on length *k* and bond diameter *d*
The python script in `source/get_ecfp.py` computes the extended-connectivity fingerprints.  
The options for the script are the following:  

--all: Boolean flag whether or not all ECFP configurations as described in the paper<sup>1</sup> should be computed. Defaults to False. In this case on the ECFP with bond-diameter *d=6* and fingerprint size *k=1024* are computed for the binary and count representations. 

--nworkers: Integer of number of parallel cpu-workers to use in order to compute the ECFP representations. Defaults to 1. In order to speed up the computation, it is recommended to use more workers.

Execution:
````
python source/get_ecfp.py -h # in order to see the information for the arguments
python source/get_ecfp.py --all False --nworkers 10 # only compute one ECFP setting and use 10 cpus for multiprocessing
````

#### Computing CDDD representations
The Jupyter Notebook in `source/get_cddd.ipynb` shows how to generate CDDD representations from the `data/smiles.npy` array.

#### Training the Neuraldecipher model
The python script in `source/main.py` excutes the training for the Neuraldecipher.  
The options for the script are the following:  

--config: String to the params.json file that consists the information for Neuraldecipher network architecture and training settings. Defaults to `params/1024_config_bit_gpu.json` 

--split: String to select if the `cluster` or `random` split should be used (see reference <sup>1</sup>) for details.  
Defaults to `cluster`.

--workers: Integer of number of parallel cpu-workers for the dataloader. Defaults to `5`

--cosineloss: Boolean flag whether or not the cosineloss should be used within the training. Defaults to `False`. This flag can be set to `True` to additionally add the cosine similarity loss next to the difference loss (e.g. L2, or logcosh).

Execution:
````
python source/main.py -h # in order to see the information for the arguments
python source/main.py --config params/1024_config_bit_gpu.json --split cluster --workers 5 --cosineloss False
````

###### Monitoring the training 
Since `tensorboard-gpu==1.10.0` is installed within the `neuraldecipher` environment, we cannot run `tensorboard==1.14.0` within the `neuraldecipher`environment. We merely included `tensorboard==1.14.0` to the `neuraldecipher` environment to log the training of our Neuraldecipher. 
To monitor the training, please create a new environment `tb` and install `tensorflow==1.14.0` (CPU version) which also includes `tensorboard==1.14.0` in its installation.
````
conda create -n tb python=3.6.10 tensorflow==1.14.0
conda activate tb
````
Run tensorboard command in a new shell (here to localhost:8888):
```
tensorboard --logdir logs/ --port 8888 --host localhost
```

#### Evaluationg the trained model
We provide the model weights for the trained model on ECFP6 representations of length 1024 trained on the cluster split and show the performance on the 
cluster validation dataset and temporal dataset in  in the Notebook `source/evaluation.ipynb`.

## References
[1] T. Le, R. Winter, F. Noe and D. Clevert, Chem. Sci., 2020, [DOI: 10.1039/D0SC03115A](https://doi.org/10.1039/D0SC03115A)

