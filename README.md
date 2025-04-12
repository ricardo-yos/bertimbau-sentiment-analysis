# Sentiment Analysis and Topic Modeling of Pet Market Reviews in Santo André, Brazil

This project combines natural language processing and machine learning to analyze customer reviews of pet-related businesses in Santo André, SP. It uses BERTimbau and Active Learning to fine-tune a model that predicts star ratings (1–5) based on review texts, while also extracting and clustering thematic patterns through topic modeling. Developed independently using real customer feedback collected via the Google Maps API.

## Table of Contents

- [Overview](#overview)
- [Main Technologies](#main-technologies)
- [Project Structure](#project-structure)
- [Data Collection](#data-collection)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Contact](#contact)

---

## Overview

Customer reviews often provide rich, detailed feedback, but the star ratings can be misleading or overly subjective. This project focuses on:

- **Predicting Star Ratings**: Using BERTimbau to train a deep learning model that predicts star ratings based on the review text.
- **Active Learning**: Implementing an Active Learning approach to iteratively improve the model with a limited set of labeled data.
- **Topic Modeling**: Analyzing reviews to extract and group recurring themes from both positive and negative feedback.
- **Scalable Pipeline**: Building a flexible pipeline that can be adapted to other types of business reviews and can be reused or adapted for other local business domains.

---

## Main Technologies

| Category             | Tools/Libraries                                   |
|:---------------------|:--------------------------------------------------|
| Programming Language | Python                                            |
| Data Manipulation    | `pandas`, `numpy`                                 |
| Visualization        | `matplotlib`, `seaborn`, `plotly`                 |
| Deep Learning        | `PyTorch`, `transformers`, `BERTimbau`            |
| Active Learning      | Custom entropy-based sampling loop               |
| Clustering & Topics  | `UMAP`, `HDBSCAN`, `KeyBERT`, `scikit-learn`      |
| Optimization         | `Optuna`                                          |
| Utilities            | `tqdm`, `dotenv`, `langid`, `googlemaps`          |

> For detailed versions, see [`requirements.txt`](./requirements.txt)  
> *Note: This project was made possible thanks to the open-source community and the authors of these tools.*

---

## Project Structure

```bash
.
├── data/
│   ├── active_learning/
│   │   ├── active_learning_state.json
│   │   └── initial_labeled_data.csv
│   ├── intermediate/
│   │   └── excluded_places.csv
│   ├── processed/
│   │   ├── places_processed.csv
│   │   └── reviews_processed.csv
│   └── raw/
│       ├── places_raw.csv
│       └── reviews_raw.csv
├── logs/
│   └── optuna/
│       └── optuna_topic_modeling.db
├── models/
│   └── sentiment_model.pt
├── notebooks/
│   ├── active_learning_with_bertimbau.ipynb
│   ├── data_cleaning_places.ipynb
│   ├── data_cleaning_reviews.ipynb
│   └── topic_modeling.ipynb
├── results/
│   ├── clustering/
│   │   ├── negative_cluster_topic_summaries.csv
│   │   ├── negative_cluster_topics.json
│   │   ├── negative_clustered_reviews.json
│   │   ├── positive_cluster_topic_summaries.csv
│   │   ├── positive_cluster_topics.json
│   │   └── positive_clustered_reviews.json
│   ├── figures/
│   │   ├── active_learning/
│   │   │   ├── actual_vs_predicted_ratings.png
│   │   │   ├── average_error_by_rating_category.png
│   │   │   ├── model_evaluation_confusion_matrix.png
│   │   │   ├── prediction_vs_actual_confusion_matrix.png
│   │   │   ├── rating_distribution.png
│   │   │   ├── word_count_distribution.png
│   │   │   └── word_count_outliers.png
│   │   └── topic_modeling/
│   │       ├── negative_hdbscan_performance.png
│   │       ├── negative_umap_clusters.png
│   │       ├── positive_hdbscan_performance.png
│   │       ├── positive_umap_clusters.png
│   │       └── sentiment_distribution.png
│   └── predictions/
│       └── reviews_with_predictions.csv
├── scripts/
│   ├── fetch_places.py
│   └── fetch_reviews.py
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Data Collection

The data was collected using the Google Places API, focusing on businesses related to pet care (e.g., pet shops, veterinary clinics, grooming services) in Santo André, São Paulo.

- Approximately 10,000 reviews were collected.
- Due to API limitations, a maximum of 5 reviews per business was retrieved.
- The dataset includes the following metadata fields:

  - `Place ID`: Unique identifier of the business (as provided by Google Places).
  - `Place Name`: Name of the establishment.
  - `Review ID`: Unique identifier of each review.
  - `Author`: Username of the person who submitted the review.
  - `Rating`: Star rating given by the reviewer (1 to 5).
  - `Text`: Content of the review (written feedback).
  - `Date`: Date when the review was posted.
  - `Time`: Timestamp of the review (where available).
  - `Response`: Optional reply from the business owner (if present).

---

## Usage

To explore or reproduce the results of this project, follow these steps:

### 1. Install Dependencies

Make sure you have **Python 3.10+** installed. Then, create a virtual environment and install the required packages:

<details>
<summary><strong>Linux / macOS</strong></summary>

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

</details>

<details> <summary><strong>Windows</strong></summary>

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

</details> 

### 2. Prepare the Data

Raw data is stored in the `data/raw/` folder. Follow these steps to fetch or update the data:

1. **Fetch places data**: Start by executing the following script to fetch the places data using the Google Places API:

    ```bash
    python scripts/fetch_places.py
    ```

2. **Clean place-level data**: After fetching the places data, run the `data_cleaning_places.ipynb` notebook to preprocess and clean the place-related data.

3. **Fetch reviews data**: Next, execute the following script to fetch the reviews data for the places:

    ```bash
    python scripts/fetch_reviews.py
    ```

4. **Clean review text**: Finally, run the `data_cleaning_reviews.ipynb` notebook to preprocess and clean the review texts.

This ensures that the data is fetched and cleaned in the correct order before proceeding to the next steps.

### 3. Run the Notebooks

Open the notebooks in the `notebooks/` folder to run each part of the pipeline:

- **`active_learning_with_bertimbau.ipynb`**: Fine-tunes the sentiment prediction model using Active Learning with the cleaned review data.

- **`topic_modeling.ipynb`**: Extracts and clusters recurring topics from the reviews, both positive and negative.

### 4. View the Results

Generated outputs and visualizations are saved in the `results/` folder:

- `results/clustering/`: Clustered topics and review groupings.
- `results/figures/`: Evaluation plots, word count stats, and UMAP visualizations.
- `results/predictions/`: Final dataset with predicted ratings.

---

## Results

### Active Learning

- Fine-tuned sentiment model using **BERTimbau** with **Active Learning** to predict star ratings (1–5).
- **Entropy-based selection** improved performance while using minimal labeled data (starting with 500 labeled examples and adding 100 per iteration).

The model reached an impressive **accuracy of 99.54%**, demonstrating strong generalization across all sentiment classes:

- **Macro F1-score**: 0.9910 — balanced performance across all classes, despite class imbalance.  
- **Weighted F1-score**: 0.9954 — confirms high overall performance, accounting for label distribution.  
- **Precision** is consistently close to 1.0, indicating a low number of false positives.  
- **Recall** reached 1.0 for classes 2, 4, and 5, showing perfect classification for those labels.  

These results highlight the effectiveness of combining **BERTimbau** with **Active Learning**, especially in low-resource, imbalanced scenarios.

### Topic Modeling

- Topic modeling revealed consistent and meaningful patterns in both positive and negative reviews.
- Clusters reflected positive aspects such as excellent veterinarians, caring service, attentive professionals, high-quality grooming, and a friendly, reliable customer experience.
- UMAP + HDBSCAN clustering grouped reviews with similar semantic content, while KeyBERT helped summarize each cluster with clear and descriptive topic labels.
- The results highlighted specific aspects valued by customers — for example, fast service and friendly staff in positive reviews, and issues like poor service, pet mistreatment, health concerns, and dissatisfaction with care in negative reviews.

**Visualizations include:**

- UMAP plots of review embeddings  
- Cluster label summaries with topic keywords

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

Feel free to reach out if you’d like to talk about projects, ideas, or potential collaborations.

- **Author:** Ricardo Yoshitomi  
- **Email:** rcd.yos@gmail.com
- **LinkedIn:** [linkedin.com/in/ricardoyoshitomi](https://www.linkedin.com/in/ricardoyoshitomi/)  
- **GitHub:** [github.com/ricardo-yos](https://github.com/ricardo-yos)
