# FindIT

The digital world is increasingly accessible to children, with mobile apps playing a significant role in their entertainment, education, and social interaction. However, this increased access also brings potential risks, particularly regarding the collection and use of children's personal information. The Children's Online Privacy Protection Act (COPPA) in the United States, and similar regulations globally, aim to protect children's privacy online by requiring app developers to obtain parental consent before collecting data from users under 13.

This competition challenges you to develop a machine learning model that can predict whether a mobile app is likely to be at risk of violating COPPA. By identifying potentially non-compliant apps, we can help app stores, developers, and parents create a safer online environment for children. Your model will analyze a variety of app characteristics, including genre, target audience (implied by download ranges), privacy policy features, and developer information, to assess the likelihood of COPPA non-compliance.

# Problem Statement
The core objective is a binary classification task:
Target Variable: coppaRisk (boolean: true or false) - Predict whether an app is at risk of violating COPPA. true indicates a higher risk of non-compliance, while false suggests a lower risk.

# Metode
## Exploratory Data Analysis (EDA)

The EDA phase was carried out to understand the structure of the data, identify patterns, and discover initial insights from the dataset. This process included several analytical activities as follows:

**1. General Data Information Analysis**
The analysis began by using the `df.info()` function to get a summary of the DataFrame, including the number of non-null values in each column and their data types. This process helped identify columns with missing values and those requiring type conversion.

**2. Initial Data Exploration**
An inspection of the first 10 rows of the dataset was done using `df.head(10)` to provide an initial overview of the data format and values present in the dataset.

**3. Unique Value Analysis**
Unique values in the 'downloads' column were examined using `df['downloads'].unique()` to identify inconsistencies or variations that needed correction.

**4. Categorical Distribution Analysis**
A frequency analysis was performed on the 'primaryGenreName' column using `df['primaryGenreName'].value_counts()` to count the distribution of each category in that column.

**5. Missing Values Identification**
Missing values were counted using `df.isnull().sum()` to get the total number of missing entries in each column. A focused check was also done on rows with missing values in the 'appAge' column using `df.loc[df['appAge'].isnull()]`.

**6. Correlation Analysis**
Correlation between numerical features was calculated and visualized using a heatmap via `sns.heatmap(df_copy.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')`. This analysis helped to understand the relationships among numerical features in the dataset.

**7. Developer Country Analysis**
Unique values and distributions in the 'developerCountry' column were examined using `df['developerCountry'].unique()` and `df['developerCountry'].value_counts()`. A frequency comparison of 'developerCountry' was also conducted between the training and test datasets using `test['developerCountry'].value_counts()` to identify categories unique to one dataset.

## Data Cleaning

The data cleaning phase focused on handling missing values, duplicates, and data inconsistencies. This process involved:

**1.  Inconsistent Value Correction**
Inconsistent values in the 'downloads' column were corrected by replacing incorrect entries such as '1 - 1' with '0 - 1', '5 - 1' with '5 - 10', and similar corrections in both training and testing datasets.

** 2. Handling Missing Values in Downloads**
Missing values in the 'downloads' column were filled using the `map_download_range` function based on the `userRatingCount` value. If the value was still missing after mapping, it was filled with 'Unknown' or an appropriate range inferred from `userRatingCount`.

**3. Handling Duplicates**
Duplicate entries were identified and counted using `df.duplicated().sum()` and `df[df.duplicated()]`. Although there were indications of duplicates, further analysis comparing df.iloc[2151] == df.iloc[2405] showed that these rows were not entirely identical across all columns.


**4. Dropping Columns with High Missing Values**
The column 'hasTermsOfServiceLink' was dropped using `df.drop(columns=['hasTermsOfServiceLink'])` due to a large number of missing values and insignificant correlation based on EDA findings.

**5. Imputation of User Ratings**
Missing values in the 'averageUserRating' column were filled with 0 to maintain data consistency.

**6. Dropping Columns with Low Correlation**
Columns 'appContentBrandSafetyRating' and 'adSpent' were dropped using `df.drop(columns=['appContentBrandSafetyRating', 'adSpent'])` due to high missing value counts and low correlation with the target, as determined from Mutual Information and Cramer's V results.

**7. Imputing Categorical Columns**
Missing values in categorical columns such as 'countryCode', 'hasPrivacyLink', 'hasTermsOfServiceLinkRating', and 'isCorporateEmailScore' were filled with the string `undetermine`.

**8.Data Format Standardization**
Standardization was applied to the 'developerCountry', 'countryCode', and 'primaryGenreName' columns by converting all values to uppercase and stripping spaces using `.str.strip().str.upper()` to ensure consistency.

**9. Country Name Correction**
Inconsistent country names were corrected, such as replacing 'VIETNAM' with 'VIET NAM' to ensure accurate frequency encoding.

**10. Numerical Imputation with MICE**
Missing values in the appAge column were filled using `IterativeImputer` (MICE - Multiple Imputation by Chained Equations), which imputes missing numeric values based on existing data patterns.

## Data Preprocessing

Tahap preprocessing mempersiapkan data untuk pemodelan dengan melakukan encoding fitur kategorikal dan scaling fitur numerik:

**1. Frequency Encoding**
Frequency encoding was applied to the 'developerCountry', 'countryCode', and 'primaryGenreName' columns using the `frequency_encode_safe function`. This function calculates the frequency of categories from the training data and maps them to both training and test sets. Categories not present in the training set were assigned a value of -1.

**2. Ordinal Encoding**
Several categorical columns were converted using ordinal encoding:
- downloads: Converted using the `encode_download_column` function, which maps download ranges to ordinal numerical values (e.g., '0 - 1' becomes 0, '1 - 5' becomes 1).
- appDescriptionBrandSafetyRating and mfaRating: Mapped using `ordinal_rating_map` with the mapping 'low': 0, 'medium': 1, 'high': 2.
- isCorporateEmailScore: Mapped with '0.0': 0, '99.0': 1, 'undetermine': -1.
- hasPrivacyLink: Mapped with 'False': 0, 'True': 1, 'undetermine': -1.
- hasTermsOfServiceLinkRating: Mapped with 'low': 0, 'high': 1, 'undetermine': -1.

**3. One-Hot Encoding**
One-hot encoding was applied to the 'deviceType' column using `OneHotEncoder` to convert it into multiple binary columns, as this column has a small number of unordered categories.

**4. Dropping Original Columns**
The original categorical columns that were encoded were dropped from the DataFrame to avoid data redundancy.

**5. Dataset Splitting**
The dataset was split into training and testing sets using `train_test_split` with the `stratify=y` parameter to ensure the same class distribution in both sets.

**6. Standard Scaling**
Standard scaling was applied to the numerical features using `StandardScaler` to normalize feature scales, which is crucial for models sensitive to feature magnitude.

**7. Oversampling with SMOTE**
SMOTE (Synthetic Minority Oversampling Technique) was applied to the scaled training data using `SMOTE(random_state=42)` to address class imbalance in the dataset.

**8. Modeling and Evaluation**
After preprocessing, various machine learning models were trained and evaluated, including XGBoost, LightGBM, Random Forest, SVM, and Logistic Regression, as well as ensemble techniques such as Voting and Stacking Classifiers. Hyperparameter tuning was conducted using `GridSearchCV`, and model evaluation was done using `classification_report`, `roc_auc_score`, and `confusion_matrix`.

# Evaluation
Submissions will be evaluated based on a suitable classification metric, likely:

AUC (Area Under the ROC Curve)
AUC, or Area Under the ROC Curve, is a single metric that summarizes the overall effectiveness of a classifier. Its value ranges from 0 to 1, where 0.5 suggests the model performs no better than random guessing, and a value of 1 indicates perfect classification performance.
