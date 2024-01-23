This is a PyTorch-based open-source deep learning framework for recommender systems enhanced by self-supervised learning techniques. 
It's user-friendly and contains commonly-used datasets, code scripts for data processing, training, testing, evaluation, and state-of-the-art research models. 

### General Collaborative Filtering
[Automated Self-Supervised Learning for Recommendation](https://arxiv.org/pdf/2303.07797.pdf) | [Disentangled Contrastive Collaborative Filtering](https://arxiv.org/pdf/2305.02759.pdf)|


### Sequential Recommendation
| Model | Conference | Paper |
| ----- | ---------- | ----- |
|BERT4Rec|CIKM'19|[BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer](https://arxiv.org/pdf/1904.06690.pdf)|
|CL4SRec|ICDE'22|[Contrastive learning for sequential recommendation](https://arxiv.org/pdf/2010.14395.pdf)|
|DuoRec|WSDM'22|[Contrastive learning for representation degeneration problem in sequential recommendation](https://arxiv.org/pdf/2110.05730.pdf)|
|ICLRec|WWW'22|[Intent contrastive learning for sequential recommendation](https://arxiv.org/pdf/2202.02519.pdf)|
|DCRec|WWW'23|[Debiased Contrastive Learning for Sequential Recommendation](https://arxiv.org/pdf/2303.11780.pdf)|

## Get Started

This is implemented under the following development environment:

+ python==3.10.4
+ numpy==1.22.3
+ torch==1.11.0
+ scipy==1.7.3
+ dgl==1.1.1

You can easily train CL4SRec using our framework by running the following script:
```
python main.py --model CL4SRec
```
This script will run the CL4SRec model on the ml datasets. 

The training configuration for CL4SRec is saved in [cl4srec.yml](https://github.com/sreeragvn/carsii/blob/main/config/modelconf/cl4srec.yml). You can modify the values in this file to achieve different training effects. Furthermore, if you're interested in trying out other implemented models, you can find a list of them under [Models](./docs/Models.md), and easily replace CL4SRec with your model of choice.

For users who wish to gain a deeper understanding, we recommend reading our [User Guide](https://github.com/sreeragvn/carsii/blob/main/docs/User%20Guide.md). This guide provides comprehensive explanations of SSLRec's concepts and usage, including:
+ SSLRec framework architecture design
+ Implementing your own model in SSLRec
+ Deploying your own datasets in SSLRec
+ Implementing your own training process in SSLRec
+ Automatic hyperparameter tuning in SSLRec

and so on.

# User Guide😉
The user guide contains the following content, you can quickly jump to the corresponding section.

+ Architecture Design of SSLRec
+ Create My Own Model
+ Create My Own DataHandler and Dataset
+ Create My Own Trainer
+ Create My Own Configuration
+ Tune My Model

## Architecture Design of SSLRec
SSLRec is a unified self-supervised recommendation algorithm framework, 
which includes the following 5 main parts.
### DataHandler
**DataHandler** is used to read the raw data, perform data preprocessing (such as converting to a sparse matrix format), and finally organize the data into a DataLoader for training and evaluation.
In our design, it contains two important functions:
+ ```__init__()```: It stores the original path of the corresponding dataset according to the configuration provided by the user.
+ ```load_data()```: It reads the raw data, performs necessary data preprocessing and finally instantiates ```train_dataloader``` and ```test_dataloader```

We designed different DataHandlers for four scenario (i.e., General Collaborative Filtering, Sequential Recommendation, Multi-behavior Recommendation, 
Social Recommendation) respectively. You can get a more detailed understanding by reading the source code of [DataHandlerGeneralCF](https://github.com/HKUDS/SSLRec/blob/main/data_utils/data_handler_general_cf.py).

### Dataset
**Dataset** inherits the ```torch.data.Dataset``` class for instantiating ```data_loader```. 
Generally, ```train_dataloader``` and ```test_dataloader``` require different Dataset classes. 
For example, in General Collaborative Filtering, we provide [PairwiseTrnData](https://github.com/HKUDS/SSLRec/blob/main/data_utils/datasets_general_cf.py) for ```train_dataloader``` to achieve negative sampling during training, 
and provide [AllRankTstData](https://github.com/HKUDS/SSLRec/blob/main/data_utils/datasets_general_cf.py) for ```test_dataloader``` to achieve all-rank evaluation.

### Model
**Model** inherits the [BasicModel](https://github.com/HKUDS/SSLRec/blob/main/models/base_model.py) class to implement various self-supervised recommendation algorithms in different scenarios.
It has four necessary functions:
+ ```__init__()```: It stores the hyper-parameter settings from user configuration as the attribute of the model, and initializes trainable parameters (e.g., user embeddings).
+ ```forward()```: It performs the model-specific forward process, such as message passing and aggregation in graph-based methods.
+ ```cal_loss(batch_data)```: The input ```batch_data (tuple)``` is a batch of training samples provided by ```train_loader```. 
  This function calculates the loss function defined by the model and has two return values: (1) ```loss (0-d torch.Tensor)``` : the overall weighted loss, (2) ```losses (dict)``` dict for specific terms of losses for printing.
+ ```full_predict(batch_data)```: The input ```batch_data (tuple)``` is the data in a test batch (e.g., ```batch_users``` (the tested users in this batch) and ```train_mask``` (training items of those users)). 
  This function return a prediction tensor ```full_pred (torch.Tensor)``` for all-rank evaluation.

You can get a more detailed understanding by reading the source code of [LightGCN](https://github.com/HKUDS/SSLRec/blob/main/models/general_cf/lightgcn.py).

### Trainer
**Trainer** provides a unified process of training, evaluation and storing model parameters. 
Using a unified trainer for different models can ensure the fairness of comparison. Our trainer including the following six functions:
+ ```create_optimizer(model)```: It creates the optimizer (e.g., ```torch.optim.Adam```) according to the configuration.
+ ```train_epoch(model, epoch_idx)```: It performs one epoch training, including calculating loss, optimizing parameters and printing the losses.
+ ```save_model(model)```: It saves the model parameters as a ```pth``` file.
+ ```load_model(model)```: It loads the model parameters from a ```pth``` file.
+ ```evaluate(model)```: It evaluates the model on test/validation set and return the results of selected metrics according to the configuration.
+ ```train(model)```: It conducts the whole training, evaluation and saving process.

Sometimes, some models may use different training process during one epoch. 
We recommend only overwriting the ```train_epoch(model, epoch_idx)``` to ensure a fair comparison.
You can read [Create My Own Trainer]() for more details.

### Configuration
Each model has its own different configuration, we write it in a ```yml``` file (e.g., [lightgcn.yml](https://github.com/HKUDS/SSLRec/blob/main/config/modelconf/lightgcn.yml)).
In a ```yml``` file, the following keys are required:
+ ```optimizer```: It contains necessary information to create an optimizer, such as the name of that optimizer and learning rate.
+ ```train```: It contains the setting of training process, such as the number of epochs, the size of each batch and so on.
+ ```test```: It sets the necessary configuration for evaluation, such as metrics, etc.
+ ```data```: It determines which dataset to use.
+ ```model```: It determines which model to create and the hyper-parameters of that model.

If you create your own model, then you have to create a configuration file for it. We recommend you to read 
[lightgcn.yml](https://github.com/HKUDS/SSLRec/blob/main/config/modelconf/lightgcn.yml) to get a basic impression of how to write configuration files, 
then jump to [Create My Own Configuration](), in which we provided a more detailed description.

## Create My Own Model
You can follow the 5 steps below to create and train your model:

_Here we assume that your model belongs to General Collaborative Filtering, which only affects the location where the model files are placed._

**First**, please create a file named ```{model_name}.py``` under ```models/general_cf/```, where ```{model_name}``` is the name of your model in lowercase.
In this file, you can code your model and implement at least these four functions: (1) ```__init__()```, (2)```forward()```, (3)```cal_loss(batch_data)``` and (4) ```full_predict(batch_data)```.
We recommend that your model class inherit the [BaseModel](https://github.com/HKUDS/SSLRec/blob/main/models/base_model.py) class to ensure the consistency of the interface.

**Second**, please create a configuration file named ```{model_name}.py``` under ```config/modelconf/``` for your model. 
You can refer to [Create My Own Configuration]() for more details.

**Third**, create a trainer in the file ```trainer/trainer.py``` for your model if you need additional operations when training your model (e.g., fix parameters).
You can refer to [Create My Own Trainer]() to see how to create and use it. 
Otherwise, you can skip this step and directly use the default [Trainer](https://github.com/HKUDS/SSLRec/blob/main/trainer/trainer.py).

**Fourth**, import your model in ```models/__init__.py``` and add additional codes in ```models/build_model.py``` like other models.

**Fifth**, train your model by this script: ```python main.py --model {model_name}```.

## Create My Own DataHandler and Dataset

### DataHandler
Currently, SSLRec has four different ```DataHandler``` classes for training and evaluation under different scenario.

We recommend that users can directly modify the existing DataHandler to avoid redundant code writing, 
for example, users can add new raw data path in the ```__init__()``` function of existed ```DataHandlers``` 
or perform different data preprocessing operations in the ```load_data()``` function.

Generally speaking, only different scenarios will use different ```DataHandlers```. 
If users need to create their own ```DataHandler```, they need to implement two functions: ```__init__()``` and ```load_data()```. 
And create two instances of the ```torch.data.DataLoader``` in load_data(), namely ```train_loader``` and ```test_loader```, for training set and test set respectively.

### Dataset

The ```Dataset``` class is used to provide sampled data for training and evaluation. 
If you need different sampling methods, you can code your own ```Dataset``` in ```data_utils/datasets_{scenario}.py```. 
And modify the ```load_data()``` function in ```DataHandler``` to choose your own ```Dataset``` by configuration.

## Create My Own Trainer
SSLRec provides a unified training process for different models in order to compare different models fairly. 
Generally, you only build your own ```Trainer``` when you need to perform some additional operations (e.g., fix parameters) during the training epoch.
You can follow the 3 steps below to create your own ```Trainer```.

**First**, create your own ```Trainer``` class in ```trainer/trainer.py```, which inherit the original ```Trainer``` class.
Then, you need to overwrite the ```train_epoch()``` function to perform your specific training operations in one epoch.

**Second**, in the {model_name}.yml, specify your trainer through the key of ```trainer``` in ```train```, 
and the recommended value is ```{model_name}_trainer```. 
You can refer to [cml.yml](https://github.com/HKUDS/SSLRec/blob/main/config/modelconf/cml.yml), which uses its own trainer.

**Third**, import your ```Trainer``` in ```trainer/__init__.py``` and add additional selection codes in ```trainer/build_trainer.py```.

## Create My Own Configuration
After you have created your own model, you need to create a ```{model_name}.yml``` file for your model in ```config/modelconf```.
The content of the configuration file must follow the following format:
```yaml
optimizer: # to define the optimizer
  name: # name of the optimizer, e.g., adam
  lr: # learning rate
  ... # other parameters, such as weight_decay for adam optimizer

train: # to define the training process
  epoch: # total number of training epochs
  batch_size: # the size of each batch
  loss: # It is used to define the Dataset for training in load_data() function from DataHandler
  test_step: # evaluate per {test_step} epochs
  reproducible: # {true, false}, whether to fix random seed
  seed: # random seed

test:
  metrics:  # list, choose in {ndcg, recall, precision, mrr}
  k: # list, top-k
  batch_size: # How many users per batch during validation

data:
  type: # choose in {general_cf, multi_behavior, sequential, social}
  name: # the name of the raw data, such as yelp


model:
  name: # case-insensitive model name, must be the same as {model_name}
  ... # other model-specific hyper-parameters, such as the number of graph neural layers
```

You can refer to [lightgcn.yml](https://github.com/HKUDS/SSLRec/blob/main/config/modelconf/lightgcn.yml) for more details.

## Tune My Model
You only need to add the following content in the configuration file to search for the optimal hyper-parameters through grid search.

_Here we take LightGCN as an example._

```yaml
tune:
  enable: true # Whether to enable grid search to search for optimal hyper-parameters
  hyperparameters: [layer_num, reg_weight] # The name of the hyper-parameter
  layer_num: [1, 2, 3] # Use a list to store the search range
  reg_weight: [1.0e-1, 1.0e-2, 1.0e-3]
```

After that, use the same script: ```python main.py --model LightGCN``` and the search will start automatically.
Note that the model name ```LightGCN``` can also be typed as ```lightgcn```, because it is case-insensitive.

# Update 2023-06-28

To facilitate the grid searching of optimal hyper-parameters, 
**early stop** is now available in SSLRec for methods that trained by `basic trainer`.

If you need to use early stop, you only need to add the `patience` keyword under the `train` keyword
in the `yaml` file to define the endurance number *n*. 
When the *n* verification results are lower than the existing best results, 
the algorithm will stop training and extract the best model parameters for testing on the test set.

Here is an example:

```yaml
train:
  epoch: 3000
  batch_size: 4096
  save_model: false
  loss: pairwise
  log_loss: false
  test_step: 3
  patience: 3
```

If there is no `patience` keyword, the trainer will train the model with fixed number of epoch as defined.

Also, note that in the ``data_handler`` file, add the correct validation and test sets. 
Otherwise, the algorithm will **loosely** use the test set as the validation set.

Here is an example in ``data_hander_general_cf.py``:
```python
def load_data(self):
    trn_mat = self._load_one_mat(self.trn_file)
    tst_mat = self._load_one_mat(self.tst_file)
    val_mat = self._load_one_mat(self.val_file)

    self.trn_mat = trn_mat
    configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape
    self.torch_adj = self._make_torch_adj(trn_mat)

    if configs['train']['loss'] == 'pairwise':
        trn_data = PairwiseTrnData(trn_mat)
    elif configs['train']['loss'] == 'pairwise_with_epoch_flag':
        trn_data = PairwiseWEpochFlagTrnData(trn_mat)
    val_data = AllRankTstData(val_mat, trn_mat)
    tst_data = AllRankTstData(tst_mat, trn_mat)
    
    self.valid_dataloader = data.DataLoader(val_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
    self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
    self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
```

This Repo used the following codebase as template
[SSLRec](https://github.com/HKUDS/SSLRec/blob/main/docs/User%20Guide.md)