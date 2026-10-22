# Amazon Review Intelligence

### Exploring customer feedback through NLP, text representations, and unsupervised learning

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Gensim](https://img.shields.io/badge/Embeddings-Word2Vec-2563EB?style=flat-square)

**From 568,454 raw reviews to a structured workflow for exploring what customers say.**

This project investigates Amazon Fine Food Reviews using SQL, data cleaning, sentiment labeling, sparse text features, and learned word embeddings. It builds several representations of review text, groups review embeddings into five clusters, and explores review similarity through cosine scores.

The work demonstrates practical NLP development across the full exploratory workflow: understanding the source data, resolving duplicate records, engineering text features, training Word2Vec, and communicating results through visualizations.

**[Explore the notebook](AMAZON_NLP_PROJECT.ipynb)** · **[View results](#results-at-a-glance)** · **[Run locally](#run-locally)**

## Results at a glance

| Project milestone | Result |
| :--- | ---: |
| Reviews in the source SQLite database | **568,454** |
| Reviews retained after filtering and cleaning | **365,331** |
| Duplicate records removed after excluding neutral ratings | **160,481** |
| Bag-of-Words feature dimensions | **50,000** |
| Reviews represented in the saved Word2Vec output | **364,374** |
| Dimensions per Word2Vec review vector | **50** |
| K-Means clusters | **5** |
| Review vectors sampled for the PCA visualization | **10,000** |

Dataset and cleaning counts were cross-checked against the local SQLite database and CSV exports. Feature dimensions, cluster assignments, and figures below come from the saved notebook execution; model training was not repeated for this documentation review.

## Why this project matters

Large review collections contain useful customer language, but reading each review manually is difficult. This project explores building blocks for making that feedback easier to inspect.

| Business question | Capability explored in this project |
| :--- | :--- |
| Which words and phrases stand out across reviews? | TF-IDF weighting, a word cloud, and ranked bigrams |
| How can similar feedback be organized for investigation? | Word2Vec review embeddings and K-Means clustering |
| Can related reviews be surfaced from an existing review? | A cosine-similarity retrieval prototype |
| How reliable is the starting data? | Duplicate removal, helpfulness consistency checks, and missing-value inspection |

These capabilities could support product research and customer-experience analysis after further validation. The notebook is an exploratory analysis; business impact and retrieval quality have not yet been measured.

## Workflow

```text
Amazon review database
        |
        v
SQL extraction + rating-based sentiment labels
        |
        v
Duplicate removal + helpfulness consistency checks
        |
        v
HTML / URL cleanup + contraction expansion + text normalization
        |
        +--> Bag-of-Words / n-grams / TF-IDF --> Feature exploration
        |
        +--> Word2Vec --> Mean review vectors --> K-Means clustering
                                               Cosine-similarity prototype
                                               PCA visualization
```

### 1. Data preparation and sentiment labeling

The notebook reads the `Reviews` table from `database.sqlite` and converts star ratings into two sentiment labels:

| Original rating | Treatment |
| :--- | :--- |
| 1–2 stars | `negative` |
| 3 stars | Excluded from this analysis |
| 4–5 stars | `positive` |

Reviews are sorted by `ProductId`, then deduplicated using `UserId`, `ProfileName`, `Time`, `Summary`, and `Text`. Records where `HelpfulnessNumerator > HelpfulnessDenominator` are removed.

| Preparation stage | Remaining reviews |
| :--- | ---: |
| Original database | 568,454 |
| After excluding 42,640 three-star reviews | 525,814 |
| After removing 160,481 duplicates under the selected key | 365,333 |
| After removing two inconsistent helpfulness records | **365,331** |

The resulting labels contain **307,967 positive reviews (84.30%)** and **57,364 negative reviews (15.70%)**. This imbalance would need attention in a future supervised classifier. The labels are derived from ratings; the current notebook does not train a sentiment classifier.

### 2. Text preprocessing

The implemented pipeline:

- Removes URLs and HTML markup using regular expressions and Beautiful Soup.
- Expands common contractions, such as `won't` → `will not`.
- Removes tokens containing digits and replaces nonalphabetic characters with spaces.
- Filters tokens against a custom stopword set and converts retained tokens to lowercase.
- Saves the resulting text in a `CleanedText` column.

The cleaned dataset is exported to `preprocessed_reviews.csv`. After CSV loading, **957 empty cleaned-text entries** are omitted from Word2Vec tokenization, leaving **364,374 reviews** for the saved embedding and clustering stages. Sparse vectorization retains all 365,331 rows by replacing missing cleaned text with empty strings.

### 3. Text feature engineering

The notebook explores five text representations. Their configurations and recorded output shapes are:

| Representation | Configuration | Matrix shape |
| :--- | :--- | :--- |
| Bag-of-Words | `CountVectorizer(max_features=50000)` | `365331 × 50000` |
| Unigram + bigram counts | `ngram_range=(1, 2)`, `min_df=10`, `max_features=5000` | `365331 × 5000` |
| Unigram TF-IDF | `TfidfVectorizer(max_features=5000)` | `365331 × 5000` |
| Unigram + bigram TF-IDF | `ngram_range=(1, 2)`, `max_features=8000` | `365331 × 8000` |
| Mean Word2Vec review embeddings | Skip-gram word vectors averaged per review | `364374 × 50` |

The count and TF-IDF matrices use SciPy sparse storage. Only small slices are converted to dense arrays for inspection, avoiding dense materialization of the full document–vocabulary matrices.

### 4. Word2Vec and unsupervised grouping

A Gensim Word2Vec model is configured with:

```python
Word2Vec(
    sentences=sample_reviews,
    vector_size=50,
    window=5,
    min_count=3,
    workers=4,
    sg=1,  # Skip-gram
)
```

Each review is represented by the mean of its in-vocabulary word vectors, with a zero-vector fallback when no word vectors are available. K-Means then groups these review vectors using `n_clusters=5` and `random_state=42`.

| Cluster ID | Reviews in the saved execution |
| :--- | ---: |
| 0 | 58,200 |
| 1 | 92,592 |
| 2 | 107,325 |
| 3 | 44,891 |
| 4 | 61,366 |
| **Total** | **364,374** |

Cluster IDs are algorithmic assignments. The notebook does not yet establish named business themes or report cluster-quality metrics.

## Visual findings

### Customer language surfaced by TF-IDF

![Top 25 two-word phrases ranked by their summed TF-IDF weights](assets/images/top-tfidf-bigrams.png)

The saved chart highlights phrases such as **“gluten free,” “highly recommend,” “taste like,” “peanut butter,”** and **“great price.”** These provide starting points for exploring dietary preferences, product categories, taste, and purchase considerations.

The chart sums TF-IDF weights across documents. Its values represent aggregate term weights, rather than review counts, sentiment scores, or evidence of purchase causation.

### Exploring review clusters in two dimensions

![PCA projection of a 10,000-review sample colored by Word2Vec K-Means cluster](assets/images/word2vec-kmeans-pca.png)

The notebook projects a random sample of **10,000 review vectors** into two dimensions using PCA, then colors points by their K-Means assignments. This helps inspect the embedding space; visual separation alone does not establish semantic coherence or an optimal number of clusters.

Both images are exported directly from the notebook's saved outputs. The notebook also includes a TF-IDF word cloud and a small feature-matrix heatmap.

## Skills demonstrated

- **SQL and data preparation:** Querying SQLite, defining analysis cohorts, and investigating repeated reviews and inconsistent metadata.
- **NLP preprocessing:** Cleaning HTML and URLs, expanding contractions, and constructing a reusable normalized text column.
- **Feature engineering:** Working with Bag-of-Words, n-grams, TF-IDF, and sparse document–term matrices.
- **Representation learning:** Training domain-specific Word2Vec embeddings and aggregating word vectors into review vectors.
- **Unsupervised learning:** Applying K-Means and exploring high-dimensional representations with PCA.
- **Analysis communication:** Connecting text features to customer-feedback questions through charts and documented results.

**Technology stack:** Python, SQLite, pandas, NumPy, Beautiful Soup, NLTK, scikit-learn, Gensim, Matplotlib, Seaborn, WordCloud, and tqdm.

## Project files

```text
AMAZON_FOOD_DATA_PROJECT/
├── README.md
├── AMAZON_NLP_PROJECT.ipynb         # Analysis, code, and saved outputs
├── assets/
│   └── images/
│       ├── top-tfidf-bigrams.png
│       └── word2vec-kmeans-pca.png
├── database.sqlite                 # Download separately; source data
├── reviews_clean_file              # Generated CSV without a .csv extension
└── preprocessed_reviews.csv        # Generated data including CleanedText
```

The data files are local inputs and generated artifacts. They do not need to be committed for others to inspect the notebook and README.

## Run locally

### 1. Obtain the dataset

Download **Amazon Fine Food Reviews** from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), extract `database.sqlite`, and place it beside the notebook. The SQLite input is required; the two cleaned exports can be regenerated.

### 2. Create an environment

From the project directory, create a fresh virtual environment. Python 3.12 is a suggested starting point; the historical environment is not locked or fully validated.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab ipykernel numpy pandas matplotlib seaborn scikit-learn nltk beautifulsoup4 tqdm gensim wordcloud
python -m nltk.downloader punkt punkt_tab
python -m ipykernel install --user --name amazon-food-nlp --display-name "Amazon Food NLP"
python -m jupyter lab AMAZON_NLP_PROJECT.ipynb
```

Select the **Amazon Food NLP** kernel. The notebook also contains package-installation cells; use the selected kernel's environment when installing dependencies.

### 3. Update two machine-specific paths

The current notebook contains absolute paths from the development machine. Before running it, replace the database path assignment with:

```python
from pathlib import Path

db_path = str(Path("database.sqlite").resolve())
```

Keep the existing `sqlite3.connect(db_path)` call. In the later cell that reloads the preprocessed dataset before Word2Vec, use:

```python
final = pd.read_csv("preprocessed_reviews.csv")
```

Run the notebook from top to bottom with the project directory as the working directory. Processing the full corpus keeps several large objects in memory; runtime and memory requirements depend on the environment and have not been benchmarked here.

## Evaluation status and next steps

The completed work covers data preparation, text representations, visual exploration, and unsupervised modeling. Accuracy, F1, retrieval relevance, and cluster-quality scores are not reported in the current notebook.

<details>
<summary><strong>Technical review notes and improvement roadmap</strong></summary>

- **Correct retrieval row alignment.** Word2Vec input drops missing `CleanedText`, while `get_similar_review()` displays text using positions in the unfiltered DataFrame. Preserve the original row IDs or use the same filtered DataFrame for embeddings and display. Explicitly exclude the query index when ranking neighbors. The saved similarity examples should be treated as unvalidated until this is corrected.
- **Refine preprocessing.** Stopword membership is checked before lowercasing, so capitalized stopwords can survive. The custom set also includes negations such as `not` and `never`; preserving these would be useful for future sentiment modeling. Stemming and lemmatization are imported but are not applied in the current pipeline.
- **Validate cluster meaning and retrieval relevance.** Inspect representative reviews and keywords per cluster, evaluate candidate cluster counts with sampled silhouette analysis, and assess retrieval using human-reviewed examples or relevance labels. Five clusters are a chosen configuration, rather than a validated optimum.
- **Establish a clean reproducible run.** Saved notebook metadata and package-installation logs reflect different Python environments, and the Word2Vec training output includes low-level `word2vec_inner` messages. Resolve those messages, lock dependencies, seed the PCA sampling step, and record a fresh ordered execution. Multithreaded Word2Vec can vary between runs; see the [Gensim reproducibility guidance](https://radimrehurek.com/gensim/models/word2vec.html).
- **Extend to supervised evaluation.** Add a leakage-aware train/test split, fit vectorizers on training data only, and compare sentiment baselines using macro-F1, class-specific recall, and a confusion matrix. Review the effect of excluding neutral ratings and the strong positive-class imbalance.
- **Improve reuse.** Export the fitted models and preprocessing configuration, add a small demonstration dataset, and evaluate TF-IDF retrieval and sentence-embedding alternatives against the same relevance criteria.

</details>

## Dataset attribution and references

The local database contains reviews dated **October 8, 1999–October 26, 2012**. Dataset background and published corpus statistics are available from [Stanford SNAP: Amazon Fine Foods Reviews](https://snap.stanford.edu/data/web-FineFoods.html); the SQLite distribution used for this project is available on [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).

SNAP requests citation of:

> J. McAuley and J. Leskovec. *From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews.* WWW, 2013.

Technical references: [scikit-learn text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction), [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html), [scikit-learn K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and [cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html).

## GitHub publishing notes

Keep the repository focused on the notebook, documentation, and figures. The local database is approximately **356 MiB**, and the generated exports are approximately **269 MiB** and **186 MiB**. GitHub blocks ordinary Git files larger than **100 MiB**; see its [large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Before staging files, add the following entries to `.gitignore`, or configure Git LFS if you deliberately want to version the large data files:

```gitignore
.venv*/
.vscode/
.DS_Store
__pycache__/
.ipynb_checkpoints/
.env
database.sqlite
reviews_clean_file
preprocessed_reviews.csv
```

Suggested GitHub repository description:

> NLP exploration of 568K Amazon reviews: SQL data cleaning, sentiment labeling, BoW and TF-IDF features, Word2Vec embeddings, K-Means clustering, and a review-similarity prototype.

Suggested topics: `nlp`, `text-mining`, `customer-reviews`, `amazon-reviews`, `python`, `sql`, `tf-idf`, `word2vec`, `kmeans-clustering`, `unsupervised-learning`, `scikit-learn`, `data-science`.
