# phase_2
# Predicting Customer Churn Using Machine Learning to Uncover Hidden Patterns

This project focuses on developing a machine learning model to predict customer churn by analyzing historical customer data. The goal is to uncover hidden patterns related to customer engagement, transaction history, demographics, and support interactions. By identifying high-risk customers early, businesses can implement targeted retention strategies, reduce churn rates, and enhance overall customer lifetime value. [cite: 1, 2, 3, 4, 5]

## Problem Statement

Customer churn, the loss of clients or subscribers, is a critical issue for subscription-based businesses. Acquiring new customers is often more costly than retaining existing ones. [cite: 1] Despite traditional efforts to mitigate churn, many underlying behavioral patterns and risk indicators remain undetected due to the complexity and volume of customer data. [cite: 2] This project aims to address this by building a predictive model. [cite: 3]

## Project Objectives

The primary objective is to develop a machine learning model that accurately predicts customer churn. [cite: 6] This involves analyzing historical customer data to uncover hidden behavioral patterns, which will enable proactive retention strategies and ultimately improve customer lifetime value. [cite: 6]

## Github Repository Link

[nanthitham1/nanthitha-predicting-customer-chum-using-machine-learning-to-uncover-hidden-patterns](https://www.google.com/search?q=nanthitham1/nanthitha-predicting-customer-chum-using-machine-learning-to-uncover-hidden-patterns) [cite: 1]

## Project Workflow

The project follows these key stages: [cite: 7]

1.  **Data Collection**: Gather customer data including transactions, demographics, usage patterns, and support logs. [cite: 7]
2.  **Data Preprocessing**: Handle missing values, encode categorical variables, normalize/scale numerical data, and perform feature engineering. [cite: 7]
3.  **Exploratory Data Analysis (EDA)**: Visualize churn trends and identify correlations and hidden patterns. [cite: 7]
4.  **Feature Selection**: Select relevant variables using statistical tests or model-based methods. [cite: 7]
5.  **Model Selection & Training**: Train various machine learning models (e.g., Logistic Regression, Random Forest, XGBoost, Neural Networks) and perform cross-validation. [cite: 7]
6.  **Model Evaluation**: Evaluate models using metrics like accuracy, precision, recall, F1-score, and AUC-ROC. [cite: 7]
7.  **Model Interpretation**: Use techniques like SHAP, LIME, or feature importance to understand key churn drivers. [cite: 8]
8.  **Deployment**: Integrate the model into a production system or dashboard for real-time churn prediction. [cite: 8]
9.  **Monitoring & Updating**: Continuously monitor model performance and retrain periodically with new data. [cite: 8]

## Data Description

The dataset includes the following features: [cite: 9, 11]

| Feature Name          | Description                                                                 |
| --------------------- | --------------------------------------------------------------------------- |
| CustomerID            | Unique identifier for each customer                                         |
| Gender                | Customer's gender (e.g., Male, Female)                                      |
| Age                   | Age of the customer                                                         |
| Tenure                | Number of months the customer has been with the company                     |
| Subscription Type     | Type of subscription plan (e.g., Basic, Premium, Family)                    |
| MonthlyCharges        | Amount charged monthly                                                      |
| TotalCharges          | Total amount charged to the customer                                        |
| PaymentMethod         | Method used by the customer to pay (e.g., Credit Card, Bank Transfer)       |
| ContractType          | Contract duration (e.g., Month-to-month, One year, Two year)                |
| InternetService       | Type of internet service (e.g., DSL, Fiber Optic, None)                     |
| OnlineSecurity        | Whether the customer has online security add-on (Yes/No)                    |
| TechSupport           | Whether the customer has technical support add-on (Yes/No)                  |
| Streaming Services    | Use of streaming services (e.g., TV, Movies, Music)                         |
| Customer SupportCalls | Number of times the customer contacted support                              |
| LastInteractionDate   | Date of the last customer activity                                          |
| Churn (Target)        | Whether the customer churned (1=Yes, 0=No)                                  |

## Data Preprocessing

The data preprocessing steps include:

1.  **Data Cleaning**: [cite: 12]
      * Remove duplicates. [cite: 12]
      * Handle missing values (numerical: fill with median/mean or interpolate; categorical: fill with mode or "Unknown"). [cite: 13, 14]
2.  **Feature Engineering**: [cite: 15]
      * Create new features (e.g., AverageMonthlySpend = TotalCharges / Tenure). [cite: 15]
      * Convert dates to calculate recency (e.g., days\_since\_last\_interaction). [cite: 15]
      * Bin continuous variables like age or tenure if helpful. [cite: 16]
3.  **Encoding Categorical Variables**: [cite: 17]
      * Label Encoding for binary features. [cite: 16]
      * One-Hot Encoding for multi-class features. [cite: 17]
4.  **Scaling/Normalization**: Use StandardScaler or MinMaxScaler for continuous features. [cite: 17]
5.  **Outlier Detection and Treatment**: Use Z-score or IQR methods to detect and optionally cap/transform outliers. [cite: 18]
6.  **Class Imbalance Handling**: Use SMOTE or class weights in model training if the target variable is imbalanced. [cite: 19, 20]
7.  **Train-Test Split**: Split the dataset (e.g., 80% training, 20% test) using stratified sampling. [cite: 21]
8.  **Save Preprocessed Data**: Optionally, export the cleaned and encoded dataset. [cite: 22]

## Exploratory Data Analysis (EDA)

EDA involves the following:

1.  **Understand the Target Variable**: Analyze value counts and distribution of the 'Churn' variable to identify class imbalance. [cite: 23]
2.  **Summary Statistics**: Use `df.describe()` for numerical features and `df.info()` for data types and null values. [cite: 24]
3.  **Univariate Analysis**: Use histograms or KDE plots for numerical features and bar plots for categorical features. [cite: 25, 26]
4.  **Bivariate Analysis**:
      * Churn vs. Numerical Variables: Boxplots or violin plots.
      * Churn vs. Categorical Variables: Stacked bar plots or heatmaps.
5.  **Correlation Analysis**: Use a correlation heatmap for numerical variables to reveal multicollinearity or redundant features. [cite: 27]
6.  **Feature Interactions**: Use pair plots or scatter matrices to explore interactions. [cite: 28]
7.  **Customer Segmentation (Optional)**: Apply clustering (e.g., KMeans) or PCA for dimensionality reduction to visualize customer groups. [cite: 28]

## Feature Engineering

Key feature engineering techniques include:

1.  **Basic Derived Features**:
      * `AverageMonthlySpend = TotalCharges / Tenure` (handle Tenure \> 0). [cite: 29]
      * `IsSenior = Age > 60`. [cite: 29]
      * `HasMultipleServices` (combine streaming, internet, and security flags). [cite: 29]
2.  **Customer Engagement Features**:
      * `EngagementScore` (weighted score of usage/activity-based features). [cite: 29]
      * `DaysSinceLastInteraction = Current date - LastInteractionDate`. [cite: 29]
      * `SupportCallRate = CustomerSupportCalls / Tenure`. [cite: 30]
3.  **Subscription Characteristics**:
      * `ContractLength` (map contract types to numerical values). [cite: 30]
      * `IsAutoPay` (based on payment method). [cite: 30]
      * `IsLongTermCustomer` (e.g., Tenure \> 24 months). [cite: 30]
4.  **Binary Flags for Add-ons**: Convert Yes/No add-on features to 1/0. [cite: 30]
5.  **Customer Lifetime Metrics**:
      * `LifetimeValue = MonthlyCharges * Tenure`. [cite: 30]
      * `ChurnRiskScore` (rule-based score from red flags like short tenure, high support calls). [cite: 30]
6.  **Interaction Features**:
      * `MonthlyCharges * ContractLength`. [cite: 30]
      * `Tenure * ContractLength`. [cite: 30]

**Tips for Good Feature Engineering**: [cite: 30]

  * Features must make real-world sense. [cite: 31]
  * Avoid data leakage. [cite: 31]
  * Evaluate feature importance later. [cite: 32]

## Model Building

The model building process involves:

1.  **Split Data**: Split data into training and testing sets (e.g., X\_train, X\_test, y\_train, y\_test). [cite: 33]
2.  **Select and Train Models**: Experiment with models like Logistic Regression, Random Forest, and XGBoost. [cite: 34, 35]
3.  **Evaluate Models**: Use metrics such as Accuracy, Precision, Recall, F1-score, and ROC AUC. [cite: 36, 37]
4.  **Tune the Best Model (Optional)**: Use GridSearchCV or RandomizedSearchCV for hyperparameter optimization. [cite: 38, 39]
5.  **Interpret the Model**: Use feature importance or SHAP to understand churn drivers. [cite: 39, 40]

## Visualization of Results & Model Insights

Visualizations and insights include:

1.  **Churn Distribution**: Bar plot to understand class imbalance. [cite: 41, 42]
2.  **Feature Importance**: Bar plot of top predictors from tree-based models. [cite: 42, 43, 44]
3.  **SHAP Summary Plot**: To explain model predictions globally and locally, showing impact on churn probability. [cite: 44, 45]
4.  **Confusion Matrix**: To visualize model performance in terms of true/false positives and negatives. [cite: 45, 46, 47]
5.  **ROC Curve**: To evaluate classification threshold and trade-offs, showing AUC. [cite: 47, 48]
6.  **Partial Dependence Plot (Optional)**: To show the marginal effect of features. [cite: 49, 50]

**Example Model Insights**: [cite: 50]

  * **Top Churn Drivers**: Short tenure, month-to-month contracts, high monthly charges, multiple support calls. [cite: 50]
  * **Customer Segments at Risk**: New users (\<6 months) on flexible contracts, users not using add-ons like online security or tech support. [cite: 50]
  * **Potential Business Actions**: Offer loyalty discounts, encourage annual plans, improve support experience. [cite: 51]

## Tools and Technologies Used

  * **Data Collection & Storage**: CSV / Excel / SQL, Pandas, NumPy. [cite: 51]
  * **Data Preprocessing**: Scikit-learn, Pandas Profiling / Sweetviz (optional). [cite: 51]
  * **Exploratory Data Analysis (EDA)**: Matplotlib / Seaborn, Plotly (optional). [cite: 51]
  * **Model Building**: Scikit-learn (Logistic Regression, Random Forest), XGBoost / LightGBM. [cite: 51, 52]
  * **Class Imbalance**: Imbalanced-learn (SMOTE). [cite: 53]
  * **Model Interpretation**: SHAP, LIME (optional), Feature Importance Plots. [cite: 53]
  * **Model Evaluation**: Scikit-learn (ROC curve, confusion matrix, F1 score, accuracy, precision, recall). [cite: 53]
  * **Model Deployment (Optional)**: Flask / FastAPI, Streamlit / Dash, Docker, AWS / Azure / GCP. [cite: 53]
  * **Version Control & Collaboration**: Git / GitHub, Jupyter Notebook. [cite: 53, 54]

## Team Members and Contributions

| Name             | Role                            | Responsibility                                                                                                                               |
| ---------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| PRIYADHARSHINI R | Lead / Data Engineer            | Oversee project development, coordinate team activities, ensure timely delivery of milestones, and contribute to documentation and final Data Engineer. [cite: 55] |
| NANDHITHA M      | Data Engineer                   | Collect data from APIs (e.g., Twitter), manage dataset storage, clean and preprocess text data, and ensure quality of input data. [cite: 55]      |
| Varshini.S, Vaishnavi.A | NLP Specialist / Data Scientist | Build sentiment and emotion classification models, perform feature engineering, and evaluate model performance using suitable metrics. [cite: 55] |
| Sonika.R         | Data Analyst / Visualization    | Conduct exploratory data analysis (EDA), generate insights, and develop visualizations such as word clouds, emotion trends, and sentiment. [cite: 55] |
