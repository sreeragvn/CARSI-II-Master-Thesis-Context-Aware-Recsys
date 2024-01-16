This is a PyTorch-based open-source deep learning framework for recommender systems enhanced by self-supervised learning techniques. 
It's user-friendly and contains commonly-used datasets, code scripts for data processing, training, testing, evaluation, and state-of-the-art research models. 

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

The training configuration for CL4SRec is saved in [cl4srec.yml](https://github.com/HKUDS/SSLRec/blob/main/config/modelconf/cl4srec.yml). You can modify the values in this file to achieve different training effects. Furthermore, if you're interested in trying out other implemented models, you can find a list of them under [Models](./docs/Models.md), and easily replace CL4SRec with your model of choice.

For users who wish to gain a deeper understanding, we recommend reading our [User Guide](https://github.com/HKUDS/SSLRec/blob/main/docs/User%20Guide.md). This guide provides comprehensive explanations of SSLRec's concepts and usage, including:
+ SSLRec framework architecture design
+ Implementing your own model in SSLRec
+ Deploying your own datasets in SSLRec
+ Implementing your own training process in SSLRec
+ Automatic hyperparameter tuning in SSLRec

and so on.