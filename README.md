
# Short Video Recommender System (KuaiRec)

## Introduction

The goal of this project is to develop a video recommender system using different approaches, including **Collaborative Filtering** and **Neural Collaborative Filtering (NCF)**. In this project, I explore several methods to build an efficient and accurate recommendation system that can predict videos a user would likely enjoy based on their previous interactions. The project uses a combination of data preprocessing, feature engineering, and model development to create an optimal solution.

---

## Setup and Installation

### Prerequisites

Create the Conda environment using the `environment.yml` file:

```bash
conda env create -f environment.yml
```

Activate the Conda environment:

```bash
conda activate rema
```

To be able to use kuairec_caption_category.csv:

On Mac
```bash
sed -i '' 's/，/;/g' data/kuairec_caption_category.csv
```

On linux
```bash
sed -i 's/，/;/g' data/kuairec_caption_category.csv
```
---

## Project Structure
├── [EDA.ipynb](EDA.ipynb)                `Exploratory Data Analysis`
├── models/
│   ├── hp_optimizer/        `Hyperparameter tuning trials`
│   ├── trained_models/      `Saved trained models`
│   ├── [ncf.ipynb](models/ncf.ipynb) `NCF model trained and tested on small_matrix`
│   └── [ncf-extended.ipynb](models/ncf-extended.ipynb)      `NCF model trained on big_matrix, tested on small_matrix`
├── [environment.yml](environment.yml)
└── [README.md](README.md)                

---

## Experiments

The initial analysis of the dataset revealed a rich set of user-item interaction logs, while content metadata such as video descriptions was more limited or sparse. Although a good number of tags were available, user behavior showed a strong bias toward a single dominant tag (e.g., Tag 28), which limited the diversity of content signals. For these reasons, I chose to work with a **Collaborative Filtering** approach, which leverages interaction patterns rather than item content .

I structured my experimentation into three key stages:

**1. Collaborative Filtering with ALS**

I first implemented **ALS**, a matrix factorization method optimized for implicit feedback, as introduced in class. To adapt it to our scenario, I followed the methodology seen in lectures: I constructed an implicit user-item matrix by introducing a binary "like" column. A video was marked as "liked" by a user if their watch ratio was above their personal average watch ratio across all videos they had watched.

ALS allowed me to make use of user-item interactions effectively, but one of its key limitations is its inability to incorporate additional  features in a flexible way. This motivated the need to explore more feature-rich architectures later in the project.

**2. Hybrid Model: ALS + TF-IDF with Cosine Similarity**
To enhance the collaborative signal with content information, I tested a hybrid method: I computed TF-IDF vectors based on video tags and metadata, and used cosine similarity to capture item-item semantic relationships. These similarity scores were then combined with ALS predictions.
However, I struggled to obtain good results with this approach.

**3. Neural Collaborative Filtering (NCF)**

Given the limitations of ALS and the hybrid model, I transitioned to a Neural Collaborative Filtering (NCF) model to improve the flexibility and expressiveness of the recommendation system. One key observation was that the ALS model's use of a binary "like" signal (based on a thresholded watch ratio) was not fully capturing the nuances of user preferences, especially for users with less consistent behavior or videos with subtle interactions. To address this, I decided to use ratings instead of binary values. The watch ratio itself was treated as a continuous rating, which allowed for a more refined representation of user-item interactions.

Therefore, I implemented a **Neural Collaborative Filtering (NCF)** model, inspired by the paper [“Neural Collaborative Filtering”](https://arxiv.org/abs/1708.05031).

NCF allowed me to incorporate rich feature sets for both users and items, making it a more expressive model for learning user preferences.

#### Feature Integration

One of the main advantages of NCF is its ability to handle diverse input features. In my implementation, I incorporated:


* User features (from user_features.csv), providing more context about user profiles and behavior patterns.

* Video tags (from item_categories.csv), which helped encode content-based information and thematic relevance.

* A custom popularity score (derived from item_daily_categories.csv), designed to capture the overall attractiveness of a video based on engagement signals such as shows, plays, likes, shares, and comments. The popularity score was learned using a linear regression model trained to predict the average watch ratio of each video. As inputs, I used aggregate metrics such as show counts, play counts, number of likes, shares, and comments. The resulting score helped capture implicit popularity patterns and was used as an additional item-level input to the neural model.

#### NCF Architecture

The architecture consists of two parallel components:

* **Generalized Matrix Factorization (GMF):** Learns latent embeddings for users and items and combines them using element-wise multiplication to model linear interactions.

* **Multi-Layer Perceptron (MLP):** Also starts with user and item embeddings, but concatenates them and feeds them into a feed-forward neural network to model more complex, non-linear interactions.

The outputs of the GMF and MLP branches are then **fused** and passed through a final prediction layer. This hybrid design allows the model to capture both **memorization patterns** (via GMF) and **generalization ability** (via MLP).

**Final Choice** 
Given its flexibility and capacity to integrate multiple feature types, **NCF was chosen as the final model for this project.**

---

## Methodology

### 1. **Data Preprocessing**


Missing values and duplicates were removed from all relevant datasets.
Invalid timestamps were filtered out from the interaction logs.
To avoid overloading the model with uninformative features, I examined the distribution of categorical variables in `user_features.csv`. Features where a single category represented over 90% of the values were removed, as they lacked variability and would not meaningfully contribute to learning user preferences.
Columns that ended in `_range` (e.g., `age_range`) were also dropped to avoid redundancy with the original categorical variables.

All features including tags, user attributes, and popularity scores were merged with the main interaction dataset.

---
### 2. **Feature Engineering**
**Score popularity**
To capture the overall appeal of each video, I engineered a popularity score from `user_item_daily.csv` using a linear regression model. It was trained to predict the average watch_ratio of a video based on aggregated engagement metrics such as:

Number of impressions (show_cnt)

Plays

Likes

Shares

Comments

The resulting score was used as an additional numeric feature for each video, providing a supervised proxy for popularity.

**Video Tag Encoding**
Video metadata included tag IDs ranging from 0 to 30. These were converted into multi-hot vectors using binary encoding, allowing each video to be associated with multiple content categories simultaneously.

**User Feature Transformation**
Categorical features in `user_features.csv`, including user_active_degree, were encoded as integers using label encoding. These features were later used as input IDs to the embedding layers of the model.


**Watch Ratio Rescale**
The watch_ratio was highly skewed, so I applied a logarithmic transformation (log(1 + x)) to reduce the effect of extreme values and improve learning stability during regression.

**Dataset Preparation for Training**
The final dataset was split into training and test sets (80/20 split). The training set was further divided into training and validation subsets. TensorFlow tf.data.Dataset pipelines were used to prepare and batch the data efficiently.

Each sample was represented as:

Categorical IDs: user_id, video_id, and user attributes

Dense features: popularity_score, tag_multi_hot

Target: the log-transformed watch_ratio



---

### 3. **Model Development**

#### NCF Model Architecture

The model consists of the following key components:

1. **Inputs:**

   * **user\_id** and **video\_id** as categorical inputs representing users and items (videos).
   * **tag\_multi\_hot** for encoding video tags into multi-hot vectors.
   * **popularity\_score** to represent the popularity of videos.
   * **user features** to incorporate additional context about the user.

2. **Embeddings:**

   * Embedding layers were used for the user and item (video) IDs to learn low-dimensional representations of users and items.
   * Each user feature is embedded using separate embeddings to allow the model to learn representations for various user attributes.

3. **GMF Branch:**

   * The **GMF** component captures linear interactions between the user and item embeddings by element-wise multiplying them.

4. **MLP Branch:**

   * The **MLP** component captures non-linear interactions by concatenating the user, item, tag, popularity, and user feature embeddings and passing them through several dense layers with dropout for regularization.

5. **Output Layer:**

   * The outputs of the GMF and MLP branches are concatenated and passed through a final dense layer to make the prediction.

6. **Model Compilation:**

   * The model was compiled with the Adam optimizer and mean squared error (MSE) loss, aiming to predict the `watch_ratio` for a user-item pair.

#### Training and Evaluation

I trained the model using the train and validation dataframe, employing early stopping to prevent overfitting. To evaluate the model’s performance, I implemented and used several metrics including NDCG@k, MAE, RMSE, Serendipity, and Popularity

#### Hyperparameter Tuning

Given the computational cost of hyperparameter optimization, I focused on **embedding dimension** and **dropout rate** as the main hyperparameters to tune. The **Keras Tuner** was used to perform the hyperparameter search. The hyperparameters tested were:

* **Embedding dimension:** Ranges from 8 to 32 with a step of 8.
* **Dropout rate:** Ranges from 0.1 to 0.3 to prevent overfitting.

Despite this, the hyperparameter tuning did not result in significant improvements in the model's performance.


#### Baseline model
To evaluate the performance of the NCF model, I implemented a global mean baseline model. This simple model predicts the log-transformed mean of the watch ratio for all users, regardless of the specific user-item interaction. The global mean was computed based on the training data.

The baseline model predicts the same value (the global mean) for every input. This simple approach serves as a reference point to compare the performance of my model. 

---
## 4. **TOP-K Recommendations**

A function to generate top-k recommendations per user using the trained model and the required input features is given.

---

## Results
TODO


---

## Future Work

* **Explore hybrid models further**: Test other hybrid techniques (e.g., weighted ensemble models) to better combine collaborative and content-based features.
TODO

---

*Pierre CHHIENG*