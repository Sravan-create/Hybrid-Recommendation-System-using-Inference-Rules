# Hybrid Recommendation System Using Inference Rules

An individual final-year project exploring movie and TV-show recommendations through association rule mining, graph visualisation, clustering, neural networks, and time-series forecasting.

Built with Python using the Netflix Movies and TV Shows dataset.

## Overview

Finding relevant content involves understanding both the catalogue and the user's preferences. This project explores several approaches to that problem: discovering relationships between content attributes, grouping similar titles, analysing catalogue growth, and matching user-entered genres.

The implementation is organised as an exploratory Jupyter Notebook. It includes separate experiments with Apriori, NetworkX, K-Means, TensorFlow, and Prophet, followed by an interactive genre-based recommendation function.

**Author:** Yerrapragada Ananda Surya Sravan  
**Project type:** Individual final-year project  
**Domain:** Artificial Intelligence and Machine Learning

## Features

- Association rule mining with the Apriori algorithm.
- Experiments with different confidence and lift thresholds.
- Directed graph visualisation of relationships between catalogue attributes.
- K-Means clustering to identify groups of similar titles.
- Cluster analysis using inertia and silhouette scores.
- TensorFlow neural-network training with dropout.
- Training and validation plots, MAE, and MSE calculations.
- Forecasting of monthly catalogue additions using Prophet.
- Approximate genre matching with FuzzyWuzzy.
- Five genre-matched recommendations by default.
- Additional genre suggestions based on the selected results.

## Technology Stack

| Technology | Purpose |
| --- | --- |
| Python | Main programming language |
| Pandas | Data loading and transformation |
| KaggleHub | Dataset download |
| mlxtend | Transaction encoding, Apriori, and association rules |
| NetworkX | Graph construction and relationship visualisation |
| scikit-learn | Encoding, scaling, imputation, clustering, and evaluation |
| TensorFlow / Keras | Neural-network implementation |
| Prophet | Time-series forecasting |
| FuzzyWuzzy | Approximate genre matching |
| Matplotlib | Charts and experiment visualisation |
| Joblib | Model persistence experiment |

## Dataset

The project uses the [Netflix Movies and TV Shows dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows), published on Kaggle by Shivam Bansal.

The dataset includes:

- Title and content type
- Director and cast
- Country
- Date added to Netflix
- Release year
- Content rating
- Duration
- Genres
- Description

> The `rating` column contains content classifications such as `TV-MA` and `PG-13`. It does not contain audience review scores.

## How the Project Works

### 1. Data Preparation

The notebook examines missing values and prepares different data representations for each experiment.

The preprocessing includes:

- Filling missing categorical values.
- Splitting genre attributes for transaction construction.
- Encoding categorical features.
- Extracting numerical duration values.
- Standardising release year and duration.
- Imputing missing numerical values.

### 2. Association Rule Mining

Apriori identifies frequent combinations of catalogue attributes using a minimum support of **0.01**.

Association rules are initially generated with a minimum confidence of **0.5**. The notebook then explores several filtering settings:

| Rule Selection | Confidence | Lift |
| --- | --- | --- |
| Initial filtering | > 0.7 | > 1.5 |
| Stricter selection | > 0.8 | > 2.0 |
| Broader filtering | > 0.5 | > 1.2 |
| Revised strong-rule selection | > 0.6 | > 1.5 |

These are manually specified experimental thresholds. Comparing them helps examine how filtering affects the available rules.

A preference-based function retrieves rules whose antecedents contain the user's chosen term and sorts the resulting suggestions by confidence.

### 3. Graph-Based Relationship Visualisation

Selected association rules are represented as directed graphs using NetworkX.

- Nodes represent rule antecedents and consequents.
- Edges connect associated attribute combinations.
- Lift is used as the edge weight.

The graphs provide a visual way to inspect recurring relationships within the catalogue.

### 4. Content Clustering

K-Means groups titles using encoded features derived from:

- Director
- Cast
- Country
- Release year
- Content rating
- Duration

The notebook examines inertia for cluster counts from **1 to 10** and silhouette scores for counts from **2 to 10**.

It includes a four-cluster experiment and a later five-cluster fit. A title-based function retrieves up to ten other titles belonging to the same cluster.

### 5. Neural-Network Experiment

The notebook implements a TensorFlow/Keras model with the following architecture:

| Layer | Configuration |
| --- | --- |
| Dense | 128 units, ReLU |
| Dropout | 0.3 |
| Dense | 64 units, ReLU |
| Dense | 32 units, ReLU |
| Output | 1 unit, linear |

**Training configuration:**

- Train-test split: 80/20
- Optimiser: Adam
- Loss: Mean Squared Error
- Epochs: 50
- Batch size: 32
- Reported metrics: MAE and MSE

Training and validation curves are plotted to inspect model behaviour.

This experiment predicts the encoded content-rating field. The current inputs also include that field, so its reported errors are not an independent measure of predictive performance.

### 6. Time-Series Forecasting

Prophet is used to analyse monthly catalogue additions.

The notebook:

1. Converts `date_added` into dates.
2. Groups titles by year and month.
3. Counts additions in each period.
4. Fits a Prophet model.
5. Generates a forecast extending twelve monthly periods into the future.

This component explores catalogue growth rather than audience demand or genre popularity.

### 7. Interactive Genre Recommendations

The final interactive function accepts comma-separated genre preferences.

It then:

1. Extracts the available genres.
2. Matches each input to the closest genre using FuzzyWuzzy.
3. Accepts matches with a similarity score **greater than 80**.
4. Filters titles containing any matched genre.
5. Returns five titles by default.
6. Suggests additional genres appearing in those results.

Example input:

```text
Comedies, Horror Movies, Action & Adventure
```

The current function sorts matching titles by the encoded `rating` column. This is a prototype ordering rule, not an audience-quality ranking. Results can include movies and TV shows.

## Project Structure

The main implementation is contained in:

```text
netflix recommendations.ipynb
```

Place the dataset alongside the notebook:

```text
Hybrid-Recommendation-System-using-Inference-Rules/
├── netflix recommendations.ipynb
├── netflix_titles.csv
└── README.md
```

Download `netflix_titles.csv` separately if it is not included in your checkout.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sravan-create/Hybrid-Recommendation-System-using-Inference-Rules.git
cd Hybrid-Recommendation-System-using-Inference-Rules
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install jupyter ipykernel kagglehub pandas numpy mlxtend networkx matplotlib scikit-learn tensorflow prophet fuzzywuzzy python-Levenshtein joblib
```

The notebook was authored in a Python 3.9 environment. Dependency versions are not pinned, so a new installation may require compatibility adjustments.

### 4. Prepare the Dataset

Download the dataset from Kaggle and place `netflix_titles.csv` in the notebook's working directory.

The notebook includes a KaggleHub download call, but subsequent CSV-loading cells use a relative file path. Either copy the downloaded CSV into the working directory or update all CSV-loading cells to use its downloaded location.

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `netflix recommendations.ipynb`, select the appropriate Python kernel, and run the cells in order.

The notebook reloads and transforms the dataset in several places, so later cells depend on the state established by earlier cells.

## Experimental Scope and Limitations

This project explores multiple methods within one notebook. The association-rule, clustering, neural-network, and forecasting experiments are not currently combined into a unified recommendation-ranking pipeline.

Other limitations include:

- **No viewing-history data:** recommendations use catalogue attributes and supplied genres.
- **Categorical encoding:** label encoding imposes numerical order on nominal attributes, which can affect distance-based clustering.
- **Evaluation leakage:** the neural-network target appears in its input features, and preprocessing is fitted before the train-test split.
- **Ranking signal:** encoded content classifications do not measure title quality or user preference.
- **Mixed duration units:** movie minutes and TV-show season counts require separate treatment.
- **Exploratory implementation:** repeated transformations and notebook state make reproducibility more dependent on execution order.

## Future Improvements

- Use categorical representations and text embeddings suited to content similarity.
- Separate movie and TV-show duration features.
- Add explicit filtering by content type.
- Combine candidate-generation methods into a shared ranking pipeline.
- Replace content-rating sorting with a meaningful relevance signal.
- Remove target leakage and fit preprocessing on training data only.
- Evaluate recommendation relevance with Precision@K, Recall@K, and NDCG when suitable interaction data are available.
- Move reusable functions into Python modules.
- Pin tested dependency versions and use native Keras model saving.

## Learning Outcomes

The project provided practical experience in:

- Preparing data for different AI and ML methods.
- Interpreting support, confidence, and lift.
- Exploring the effects of rule-filtering thresholds.
- Visualising relationships through directed graphs.
- Comparing cluster configurations.
- Implementing neural-network training and inspecting its outputs.
- Preparing monthly count data for forecasting.
- Handling approximate user input.
- Recognising the importance of suitable ranking signals and reliable evaluation.

## Author

**Yerrapragada Ananda Surya Sravan**

Individual final-year project in Computer Science and Engineering, specialising in Artificial Intelligence and Machine Learning.

[GitHub Repository](https://github.com/Sravan-create/Hybrid-Recommendation-System-using-Inference-Rules)

## Acknowledgements

Thanks to Shivam Bansal for the Netflix dataset and to the maintainers of the open-source libraries used in this project.

Refer to the original dataset page for its terms of use and to each library for its licence.
