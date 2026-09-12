# Customer Segmentation using K-Means

## 📌 Project Overview

This project applies **K-Means Clustering** to segment credit card customers based on their spending behavior, payment patterns, credit usage, and transaction activity.

The main goal is to identify meaningful customer groups that can help businesses understand different customer behaviors and support **targeted marketing, customer engagement, risk assessment, and product development**.

Since K-Means is a distance-based algorithm, the project focuses strongly on **data preprocessing**, including handling missing values, applying **Log Transformation** to reduce skewness, and using **StandardScaler** to put all features on a comparable scale.

---

## 🎯 Project Objectives

- Explore and understand the credit card customer dataset.
- Handle missing values and unnecessary columns.
- Analyze feature distributions and identify skewed features.
- Apply **Log Transformation** to reduce the effect of highly skewed financial features.
- Standardize the features using **StandardScaler**.
- Apply **PCA** for dimensionality reduction and visualization.
- Determine a suitable number of clusters using:
  - Elbow Method
  - Silhouette Score

- Apply **K-Means Clustering** with `k = 5`.
- Analyze and interpret the characteristics of each customer segment.
- Visualize the resulting customer segments using PCA.

---

## 📊 Dataset

The project uses the **Credit Card Customer Dataset**, which contains information about credit card users and their financial behavior.

The dataset includes features related to:

- Credit limits
- Account balances
- Purchases
- Cash advances
- Payments
- Installment purchases
- Transaction frequency
- Credit card usage behavior

The dataset contains **8,950 customer records** and **17 behavioral features** after removing the customer ID.

### Important Features

| Feature                            | Description                                          |
| ---------------------------------- | ---------------------------------------------------- |
| `BALANCE`                          | Customer's outstanding balance                       |
| `BALANCE_FREQUENCY`                | Frequency of balance updates                         |
| `PURCHASES`                        | Total amount of purchases                            |
| `ONEOFF_PURCHASES`                 | Amount spent on one-off purchases                    |
| `INSTALLMENTS_PURCHASES`           | Amount spent on installment purchases                |
| `CASH_ADVANCE`                     | Amount of cash advances                              |
| `PURCHASES_FREQUENCY`              | Frequency of purchases                               |
| `ONEOFF_PURCHASES_FREQUENCY`       | Frequency of one-off purchases                       |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Frequency of installment purchases                   |
| `CASH_ADVANCE_FREQUENCY`           | Frequency of cash advances                           |
| `CASH_ADVANCE_TRX`                 | Number of cash advance transactions                  |
| `PURCHASES_TRX`                    | Number of purchase transactions                      |
| `CREDIT_LIMIT`                     | Customer's credit limit                              |
| `PAYMENTS`                         | Total payments made                                  |
| `MINIMUM_PAYMENTS`                 | Minimum payments made                                |
| `PRC_FULL_PAYMENT`                 | Percentage of payments made in full                  |
| `TENURE`                           | Number of months the customer has been with the bank |

---

## 🔍 Exploratory Data Analysis

The initial analysis examined:

- Dataset shape and structure
- Data types
- Missing values
- Duplicate records
- Descriptive statistics
- Feature distributions

The `CUST_ID` column was removed because it is an identifier and does not provide useful information for clustering.

### Original Feature Distributions

The original distributions show that several monetary features are **highly right-skewed**, with some features containing very large values.

![Original Feature Distributions](images/Original%20Feature%20Distributions.png)

---

## 🧹 Data Preprocessing

### 1. Handling Missing Values

Missing values were removed using `dropna()`.

The missing values represented a relatively small portion of the dataset, so removing these rows was considered appropriate for this analysis.

### 2. Removing Customer ID

`CUST_ID` was removed because it is only an identifier and has no meaningful relationship with customer behavior.

### 3. Log Transformation

Many financial features have highly skewed distributions.

A **Log Transformation** using `np.log1p()` was applied:

```python
transformer = FunctionTransformer(np.log1p)
Data = transformer.transform(data)
```

This helps:

- Reduce right skewness.
- Compress extreme values.
- Reduce the influence of very large monetary values.
- Make customer behavior differences easier to capture.

### Log-Transformed Feature Distributions

![Log-Transformed Feature Distributions](images/Log-Transformed%20Feature%20Distributions.png)

### 4. Feature Scaling

After the log transformation, **StandardScaler** was applied:

```python
scaler = StandardScaler()
scaledData = scaler.fit_transform(Data)
```

Standardization transforms the features so that they have approximately:

- Mean = `0`
- Standard deviation = `1`

This is especially important for K-Means because the algorithm relies on **Euclidean distance**.

![Scaled Feature Distributions](images/Scaled%20Feature%20Distributions.png)

---

## 📉 PCA Visualization

The dataset contains 17 features, which makes direct visualization difficult.

**Principal Component Analysis (PCA)** was used to reduce the data to two dimensions:

```python
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaledData)
```

The first two principal components provide a 2D representation of the customer data.

![PCA of Scaled Data](images/PCA%20of%20Scaled%20Data.png)

The PCA visualization shows that the customer data does not naturally form perfectly separated groups, which is consistent with the relatively low Silhouette Scores.

---

## 📐 Choosing the Number of Clusters

Two methods were used to evaluate different values of `k`.

### Elbow Method

The Elbow Method evaluates the **Within-Cluster Sum of Squares (WCSS)** for different numbers of clusters.

![Elbow Method](images/Elbow%20Method.png)

The elbow is not very sharp, but the curve begins to flatten around `k = 4` or `k = 5`.

### Silhouette Score

The Silhouette Score was calculated for values of `k` from 2 to 10.

![Silhouette Score](images/Silhouette%20Score.png)

The highest score was obtained with **k = 2**, with a score of approximately **0.25**.

However, because customer segmentation can benefit from more detailed and actionable groups, **k = 5** was selected for the final segmentation.

The relatively low Silhouette Score also indicates that the customer groups have some overlap rather than being completely separated.

---

## 🤖 K-Means Clustering

K-Means was applied using four clusters:

```python
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

kmeans.fit(scaledData)
```

The resulting cluster labels were added to the original dataset for further analysis.

---

## 📊 Customer Segments

The four clusters were analyzed based on their average behavioral characteristics.

### Cluster 0 — Transactors

- High `BALANCE`
- High `PURCHASES`
- High `PAYMENTS`
- High `CREDIT_LIMIT`
- High number of purchase transactions
- High one-off and installment purchases
- Very high `PRC_FULL_PAYMENT`

These customers are highly active and tend to pay their balances in full.

**Business interpretation:**
Potentially valuable customers with strong engagement and relatively healthy payment behavior.

---

### Cluster 1 — Revolvers

- High `BALANCE`
- Lower `PURCHASES`
- High `CASH_ADVANCE`
- High `CASH_ADVANCE_TRX`
- Very low `PRC_FULL_PAYMENT`

These customers tend to carry balances and make greater use of cash advances.

**Business interpretation:**
Potentially profitable customers because they carry balances, but they may also represent a higher-risk segment.

---

### Cluster 2 — Inactive / Low-Activity Customers

- Lowest `BALANCE`
- Lowest `PURCHASES`
- Lowest `PAYMENTS`
- Lower `CREDIT_LIMIT`
- Low purchase transaction activity
- Low overall card usage

These customers show relatively low engagement with their credit cards.

**Business interpretation:**
Potential target for customer engagement and activation campaigns.

---

### Cluster 3 — Installment Buyers

- Moderate to high `BALANCE`
- Moderate to high `PURCHASES`
- High `INSTALLMENTS_PURCHASES`
- High `PURCHASES_INSTALLMENTS_FREQUENCY`
- Lower `PRC_FULL_PAYMENT`

These customers show a strong preference for installment-based purchases.

**Business interpretation:**
Potential target for installment offers, financing products, and relevant promotional campaigns.

---

## 📈 Cluster Visualization

The four clusters were visualized using the first two PCA components.

![Customer Clusters](images/Customer%20Clusters%20%28k=4%29%20visualized%20with%20PCA.png)

The PCA plot shows some separation between the groups, but there is also considerable overlap.

This is expected because PCA is only a **2D projection of the original 17-dimensional feature space**, and the Silhouette Score also indicates that the clusters are not strongly separated.

---

## 🔎 Key Findings

The analysis identified four meaningful customer segments:

| Cluster | Segment                     | Main Characteristics                                     |
| ------- | --------------------------- | -------------------------------------------------------- |
| 0       | **Transactors**             | High spending, high payments, high full-payment ratio    |
| 1       | **Revolvers**               | High balance, high cash advances, low full-payment ratio |
| 2       | **Inactive / Low-Activity** | Low spending, low balance, low card usage                |
| 3       | **Installment Buyers**      | High installment purchases and installment frequency     |

These segments demonstrate how unsupervised learning can reveal different behavioral patterns without requiring predefined customer labels.

---

## 💡 Business Applications

The identified segments could be used to support:

- **Targeted Marketing** — Create different offers for different customer groups.
- **Customer Engagement** — Encourage inactive customers to use their cards more.
- **Installment Product Development** — Design offers for customers who frequently use installments.
- **Risk Assessment** — Monitor customers with high balances and cash advance activity.
- **Customer Retention** — Develop personalized strategies for highly valuable customers.

---

## 🛠️ Technologies & Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

### Machine Learning Techniques

- Log Transformation
- Standardization
- PCA
- K-Means Clustering
- Elbow Method
- Silhouette Score

---

## 📁 Project Structure

```text
Credit-Card-Customer-Segmentation/
│
├── data/
│   └── CC GENERAL.csv
│
├── images/
│   ├── Original Feature Distributions.png
│   ├── Log-Transformed Feature Distributions.png
│   ├── Scaled Feature Distributions.png
│   ├── PCA of Scaled Data.png
│   ├── Elbow Method.png
│   ├── Silhouette Score.png
│   └── Customer Clusters (k=5) visualized with PCA.png
│
├── notebook/
│   └── credit_card_customer_segmentation.ipynb
│
├── requirements.txt
│
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <https://github.com/Hager-Rabie/Credit-Card-Customer-Segmentationl>
cd Credit-Card-Customer-Segmentation
```

````

### 4. Install the required libraries

```bash
pip install -r requirements.txt
````

### 5. Run the Jupyter Notebook

```bash
jupyter notebook
```

Then open the notebook inside the `notebook/` folder.

---

## 📌 Conclusion

This project demonstrates a complete **K-Means customer segmentation workflow**, starting from data exploration and preprocessing to clustering, visualization, and business interpretation.

The preprocessing steps were particularly important because the dataset contains highly skewed financial features. Applying **Log Transformation** and **StandardScaler** helped make the data more suitable for distance-based clustering.

Using K-Means with `k = 4` resulted in four interpretable customer segments:

**Transactors, Revolvers, Inactive / Low-Activity Customers, and Installment Buyers.**

Although the relatively low Silhouette Score indicates that the clusters are not perfectly separated, the resulting segments still provide useful behavioral insights that can support customer-focused business strategies.

---

## 👩‍💻 Author

**Hager Rabie**

Machine Learning & AI Enthusiast
