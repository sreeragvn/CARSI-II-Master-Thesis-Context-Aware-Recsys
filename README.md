# Context-Aware In-Car Recommender System: Enhancing User Interface Interactions Based on Driving Context

Modern automotive infotainment systems offer a complex and wide array of controls and features through various interaction methods. However, such complexity can distract the driver from the primary task of driving, increasing response time and posing safety risks to both car occupants and other road users. Additionally, an overwhelming UI can significantly diminish usability and the overall user experience. A streamlined UI enhances user experience, reduces driver distraction, and improves road safety. Adaptive UIs that recommend preferred items to the user represent an intelligent UI, potentially enhancing both user experience and driver safety. This research explores the use of deep learning techniques to develop a recommender system aimed at improving the infotainment system. This is achieved by leveraging data from various sensors on the Controller Area Network (CAN-bus) and external Application Programming Interfaces (APIs), combined with previous UI interaction history. This data integration facilitates the creation of a refined UI equipped with a recommender system, aimed at improving the driving experience. Such systems are vital for helping users filter information and pinpoint their preferences. The core aim here is to provide recommendations that simplify user interactions with the UI, thereby minimizing distractions and cognitive load. Predictions for subsequent UI interactions need to be informed by the context, incorporating the driver's past UI interactions and current driving conditions such as road type, speed, and traffic.

The automotive industry is in the nascent stages of integrating deep learning methods into production cars to enhance existing systems. Previous studies on automated driving and driver identification underscore the importance of driving context in improving these systems and creating personalized experiences. A recommender system can recommend vehicle functionalities relevant to the driver based on the driving context. Deep learning, due to its ability to learn intricate relationships, is well-suited to understand correlations between UI usage and driving context. This research introduces a transformer-based deep learning model architecture to learn user behaviour in selected driving contexts and make relevant recommendations. The model demonstrates promising results in identifying driving contexts and providing contextually appropriate UI item recommendations, even for previously unseen users. This study also investigates self-supervised contrastive learning approaches to assess their impact on model performance. Furthermore, the model's performance is evaluated with fine-tuning to evaluate its ability to make personalised recommendations to new users. An ablation study is conducted to evaluate the relative importance of each model component and explore causal relationships.


## Dataset

For this study on context-aware in-car recommender systems, we began a data gathering initiative using a fleet of Porsche Taycan vehicles driven by six participants. The collected data includes vehicle signals from the CAN, and user interface interactions recorded via an event logging system. The data includes vehicle signals aggregated through the vehicle Controller Area Network (CAN) and infotainment system UI interaction logs captured via an event logging system. The CAN operates as the essential communication backbone among various electronic control units (ECUs) within the vehicle, facilitating efficient data exchange critical for vehicle functionality. Concurrently, the event logging system records user interactions with the vehicle's infotainment system, capturing inputs such as hard key presses and touchscreen interactions.

The vehicle's CAN-bus signals were captured in MDF-4 format, consisting of approximately 10,000 channels. These channels encompass a wide range of frequencies, including high-frequency, low-frequency, and cyclic-frequency sensors, tailored to capture detailed vehicle dynamics and operational parameters. Through a combination of literature review and exploratory data analysis, specific signals are relevant to defining the driving context for the recommender system. MDF-4 files also included records of hard key press interactions, particularly drive mode changes.

User interactions with the vehicle’s infotainment system were logged in JSON format, detailing the types of interactions, timestamps, and associated metadata. This data was subsequently parsed and converted into a flat-file format to streamline analysis. Given the sparse nature of certain interactions and the small size of the dataset, only the top 22 most frequently used features were selected for further study, discarding features used fewer than five times across the dataset.

Preprocessing of the data parquet and pncrec after it is parsed is available in the folder [preprocessing](preprocessing)

## Research Questions

Our research seeks answers to several questions:
1. Does CARSI II outperform state-of-the-art models, including other transformer or RNN-based methods?
2. What is the influence of various components in the CARSI II architecture?
3. How well model identify and predict those events which are rarely used by the user?
4. What is the impact of the loss function on improving model performance?
5. How well does the CARSI II model generalize to unseen users?
6. Is there any improvement in model performance with a self-supervised learning approach over supervised training?

## Model deployment

To make real-time predictions with the model, follow these instructions. Please note that the `inference.py` file is no longer required; use the `main.py` file instead.

### Configuration

1. **Config File**: Update the configuration file [cl4rec.yml](config/modelconf/cl4rec.yml) to change the trained model used during inference. 

2. **Inference Model Path**: Ensure that the `["test"]["inference_model_path"]` in the configuration file points to the model intended for inference. This model should be available in the [cl4rec](checkpoint/cl4rec) directory.

3. **Data Preparation**: The data for making predictions should be placed in the [inference](datasets/sequential/inference) folder. Template files are available in this directory. Replace the data in these templates with your data, ensuring the correct order of features is maintained.

    - **Interaction History**: The [interaction_history](datasets/sequential/inference/seq/test.tsv) should be mapped based on esotrace [label](datasets/sequential/featengg/parameters/label_mapping.pkl) mapping and saved in the specified format.
    - **Preprocessing**: Refer to the [preprocessing](preprocessing) folder for the various preprocessing steps to be conducted before replacing the data in the [inference](datasets/sequential/inference) folder. All preprocessing should be completed before writing to the corresponding model input files in [inference](datasets/sequential/inference).

4. **Mode Setting**: In the [cl4rec.yml](config/modelconf/cl4rec.yml) file, ensure the mode is set to `inference` under `["model"]["mode"]`. Other available modes are `train`, `test`, and `tune`.

### Inference Process Flow

1. **Replace Data**: Place the data you want to use for predictions in the [inference](datasets/sequential/inference) folder. If you want to make multiple predictions for different scenarios, append the data together in the appropriate format.

2. **Run Main Script**: Execute the `main.py` file. Ensure that the mode is set to `inference` in the [cl4rec.yml](config/modelconf/cl4rec.yml) configuration file.

    ```bash
    python main.py
    ```

Following these steps will enable you to make real-time predictions using your trained model.

## Model architecture
The CARSI II model architecture handles various input modalities, including time series data, sequential data, and a mix of categorical and dense scalar inputs. Emphasizing sequential modeling, the architecture aims to improve the accuracy and relevance of recommendations by capturing the dynamic nature of user interactions. This approach helps clarify typical user engagement patterns with system features, such as playing music from a phone after connecting it to the infotainment system, and prevents redundant recommendations, like suggesting an already active drive mode.

This model considers both dynamic and static context components. Static context elements, such as the number of occupants, time of day, vehicle occupancy, and trip distance, remain constant in real-time. In contrast, dynamic context involves real-time sensor signals.

The CARSI II architecture transforms static context features and previously clicked items into low-dimensional vector embeddings. Time-series dynamic context features are processed using a Temporal Convolutional Network (TCN) to capture time-dependent patterns such as periodic changes in driving speed or acceleration, indicating distinct driving styles or road conditions.
    
To enhance the representation of user interaction history, a transformer layer is integrated to extract deeper sequential insights from the data. Each interaction within a user's history is transformed into a dense vector representation via embedding, similar to a bag of words.

For each categorical feature in the static context, an individual embedding network is established to learn discrete embeddings for each category. Cyclical static context features, such as hours, days, months, and seasons, are encoded to reflect their repeating cycles using sine and cosine transformations.

Outputs from both processes are input into a Multi-layer Perceptron (MLP). The softmax function ultimately transforms the vector of real numbers into a probability distribution, indicating the likelihood of each target label.

The various components of the model and the final model architecture is available in [models](models)

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
[param.pkl](datasets/sequential/featengg/parameters/param.pkl) consists of counter weights required for balancing cross entropy loss during the training process. This is no longer required as we are using focal loss

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
