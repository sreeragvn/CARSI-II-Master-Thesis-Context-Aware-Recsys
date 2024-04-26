# Context-Aware In-Car Recommender System: Enhancing User Interface Interactions Based on Driving Context

To advance the development of automotive UI systems, it's essential to deeply understand user interactions with the display and controls. This is achieved by leveraging data from various sensors on the Controller Area Network (CAN-bus) and external Application Programming Interfaces (APIs), combined with previous UI interaction history. This data integration facilitates the creation of a refined UI equipped with a recommender system, aimed at improving the driving experience. Such systems are vital for helping users filter information and pinpoint their preferences. The core aim here is to provide recommendations that simplify user interactions with the UI, thereby minimizing distractions and cognitive load.

Predictions for subsequent UI interactions need to be informed by the context, incorporating the driver's past UI interactions and current driving conditions such as road type, speed, and traffic.

## Dataset

For this study on context-aware in-car recommender systems, we began a data gathering initiative using a fleet of Porsche Taycan vehicles driven by six participants. The collected data includes vehicle signals from the CAN, and user interface interactions recorded via an event logging system.

## Research Questions

Our research seeks answers to several questions:
1. Which driving context variables significantly influence UI interactions?
2. How accurate are the UI recommendations based on contextual driving conditions and individual usage patterns?
3. How effective is the context-sensitive UI recommender system at reducing driver distraction by optimizing UI interactions?
4. How does this system adapt and evolve over time with individual UI preferences and driving histories, thereby enhancing the accuracy and usefulness of its recommendations?

## Get Started

This is implemented under the following development environment:

- python==3.8.18

You can easily train this framework by running the following script:

```bash
python main.py
```

## Architecture Design of SSLRec
This library encompasses five primary components, each integral to the system's functionality.

### DataHandler
**DataHandler** plays a pivotal role in managing raw data. It executes several critical operations:
+ `__init__()` stores the path of the dataset as per user configuration.
+ `load_data()` reads and preprocesses raw data, then organizes it into `train_dataloader` and `test_dataloader` for effective training and evaluation.

### Dataset
**Dataset** extends the `torch.data.Dataset` class, facilitating the instantiation of `data_loader`. It is tailored to handle distinct classes for `train_dataloader` and `test_dataloader`.

### Model
**Model** is derived from the BasicModel class and is designed to implement diverse self-supervised recommendation algorithms suited to various scenarios. It includes several key methods:
+ `__init__()` initializes the model with user-configured hyper-parameters and trainable parameters such as user embeddings.
+ `forward()` conducts the specific forward operations of the model, like message passing and aggregation in graph-based methods.
+ `cal_loss(batch_data)` calculates losses during training. It takes a tuple of training data as input and returns both the overall weighted loss and specific loss details for monitoring.
+ `full_predict(batch_data)` generates predictions across all ranks using test batch data, returning a prediction tensor.

### Trainer
**Trainer** standardizes the training, evaluation, and parameter storage processes across models to ensure fairness in comparison. It includes:
+ `create_optimizer(model)` configures the optimizer based on predefined settings (e.g., `torch.optim.Adam`).
+ `train_epoch(model, epoch_idx)` handles the operations of a single training epoch, such as loss calculation, parameter optimization, and loss reporting.
+ `save_model(model)` and `load_model(model)` manage the storage and retrieval of model parameters.
+ `evaluate(model)` assesses the model performance on test or validation sets and reports selected metrics.
+ `train(model)` oversees the entire training and evaluation cycle.

### Configuration
Configuration settings for each model are specified in a `yml` file, which includes:
+ `optimizer`: Details necessary for optimizer creation.
+ `train`: Training process settings like epoch count and batch size.
+ `test`: Evaluation configurations.
+ `data`: Dataset specifications.
+ `model`: Model creation parameters and hyper-parameters.

Thanks: [SSLRec User Guide](https://github.com/HKUDS/SSLRec/).