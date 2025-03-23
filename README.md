# 🎬 Hybrid Recommendation System Using Inference Rules

This project is an intelligent, hybrid movie recommendation system built on a Netflix dataset. It combines multiple AI and ML techniques such as **association rule mining**, **knowledge graphs**, **clustering**, **deep learning**, and **time series forecasting** to generate highly relevant recommendations based on user input.

---

## 📌 Features

✅ Association rule mining using the **Apriori algorithm**  
✅ Relationship extraction via **Knowledge Graphs**  
✅ **K-Means clustering** to group similar content  
✅ Training using **TensorFlow** models  
✅ **Prophet** for forecasting trends using movie release dates  
✅ **FuzzyWuzzy** to smartly match user-input genres  
✅ Top 5 recommended movies shown based on highest ratings  
✅ Extra suggestions of genres the user may also like  

---

## 📂 Files Included

| File | Description |
|------|-------------|
| `netflix_titles.csv` | The dataset used (sourced from Kaggle) |
| `netflix recommendations.ipynb` | Main notebook containing the entire recommendation system |
| `README.md` | Project documentation |
| `LICENSE` | License for this repository |

---

## 🛠️ Tech Stack & Libraries

- **Python**
- **Pandas**, **NumPy**
- **mlxtend** (Apriori & Association Rules)
- **NetworkX** (Knowledge Graphs)
- **scikit-learn** (K-Means Clustering)
- **TensorFlow** (Model Training)
- **Prophet** (Time Series Forecasting)
- **FuzzyWuzzy** (Fuzzy Matching)
- **Matplotlib / Seaborn** (Visualizations)

---

## 📈 How It Works

1. Load and preprocess the Netflix dataset.
2. Extract relationships between genres, cast, and other attributes using Apriori and build a **Knowledge Graph**.
3. Apply **K-Means** to cluster similar types of content.
4. Use **TensorFlow** to train models on the clustered and associated data.
5. Feed release dates into **Prophet** to understand and predict trending genres over time.
6. When the user inputs genres (even misspelled or partial), **FuzzyWuzzy** matches them to the closest valid genres.
7. Based on user input, recommend the **Top 5 highest-rated movies**.
8. Also suggest additional genres the user may enjoy based on patterns in the top 5 results.

---

## 📦 Installation

```bash
git clone https://github.com/Sravan-create/Hybrid-Recommendation-System-using-Inference-Rules
cd Hybrid-Recommendation-System-using-Inference-Rules
pip install -r requirements.txt
