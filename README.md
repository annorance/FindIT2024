# FindIT

The digital world is increasingly accessible to children, with mobile apps playing a significant role in their entertainment, education, and social interaction. However, this increased access also brings potential risks, particularly regarding the collection and use of children's personal information. The Children's Online Privacy Protection Act (COPPA) in the United States, and similar regulations globally, aim to protect children's privacy online by requiring app developers to obtain parental consent before collecting data from users under 13.

This competition challenges you to develop a machine learning model that can predict whether a mobile app is likely to be at risk of violating COPPA. By identifying potentially non-compliant apps, we can help app stores, developers, and parents create a safer online environment for children. Your model will analyze a variety of app characteristics, including genre, target audience (implied by download ranges), privacy policy features, and developer information, to assess the likelihood of COPPA non-compliance.

# Problem Statement
The core objective is a binary classification task:
Target Variable: coppaRisk (boolean: true or false) - Predict whether an app is at risk of violating COPPA. true indicates a higher risk of non-compliance, while false suggests a lower risk.

# Metode
## Exploratory Data Analysis (EDA)

Tahap EDA dilakukan untuk memahami struktur data, mengidentifikasi pola, dan menemukan insight awal dari dataset. Proses EDA mencakup beberapa aktivitas analisis sebagai berikut:

**1. Analisis Informasi Umum Data**
Analisis dimulai dengan menggunakan fungsi `df.info()` untuk mendapatkan ringkasan DataFrame, termasuk jumlah non-null pada setiap kolom dan tipe datanya. Proses ini membantu mengidentifikasi kolom dengan missing values dan tipe data yang perlu dikonversi.

**2. Eksplorasi Data Awal**
Dilakukan pemeriksaan terhadap 10 baris pertama dataset menggunakan `df.head(10)` untuk memberikan gambaran awal tentang format data dan nilai-nilai yang ada dalam dataset.

**3. Analisis Nilai Unik**
Pemeriksaan nilai unik dilakukan pada kolom 'downloads' menggunakan `df['downloads'].unique()` untuk mengidentifikasi inkonsistensi atau variasi yang perlu diperbaiki dalam data.

**4. Analisis Distribusi Kategorikal**
Analisis frekuensi dilakukan pada kolom 'primaryGenreName' menggunakan `df['primaryGenreName'].value_counts()` untuk menghitung distribusi setiap kategori dalam kolom tersebut.

**5. Identifikasi Missing Values**
Penghitungan missing values dilakukan menggunakan `df.isnull().sum()` untuk mendapatkan jumlah missing values di setiap kolom. Selain itu, dilakukan pemeriksaan khusus pada baris dengan missing values di kolom 'appAge' menggunakan `df.loc[df['appAge'].isnull()]`.

**6. Analisis Korelasi**
Analisis korelasi antar fitur numerik dilakukan dengan menghitung matriks korelasi dan memvisualisasikannya menggunakan heatmap `sns.heatmap(df_copy.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')`. Analisis ini membantu memahami hubungan antar fitur numerik dalam dataset.

**7. Analisis Negara Developer**
Dilakukan pemeriksaan nilai unik dan distribusi pada kolom 'developerCountry' menggunakan `df['developerCountry'].unique()` dan `df['developerCountry'].value_counts()`. Perbandingan frekuensi 'developerCountry' juga dilakukan antara data train dan test menggunakan `test['developerCountry'].value_counts()` untuk mengidentifikasi kategori yang hanya ada di satu dataset.

## Data Cleaning

Tahap cleaning data berfokus pada penanganan missing values, duplikat, dan inkonsistensi data. Proses cleaning meliputi:

**1. Koreksi Nilai Inkonsisten**
Dilakukan perbaikan nilai inkonsisten pada kolom 'downloads' dengan mengganti nilai yang salah seperti '1 - 1' menjadi '0 - 1', '5 - 1' menjadi '5 - 10', dan koreksi serupa lainnya pada training dan testing dataset.

** 2. Penanganan Missing Values pada Downloads**
Missing values pada kolom 'downloads' diisi menggunakan fungsi `map_download_range` berdasarkan nilai `userRatingCount`. Apabila nilai downloads masih kosong setelah mapping, diisi dengan 'Unknown' atau berdasarkan rentang `userRatingCount` yang sesuai.

**3. Penanganan Duplikat**
Identifikasi dan penghitungan duplikat dilakukan menggunakan `df.duplicated().sum()` dan `df[df.duplicated()]`. Meskipun terdapat indikasi duplikat, analisis lebih lanjut dengan perbandingan `df.iloc[2151] == df.iloc[2405]` menunjukkan bahwa baris tersebut tidak sepenuhnya identik di semua kolom.

**4. Penghapusan Kolom dengan Missing Values Tinggi**
Kolom 'hasTermsOfServiceLink' dihapus menggunakan `df.drop(columns=['hasTermsOfServiceLink'])` karena memiliki banyak missing values dan korelasi yang tidak signifikan berdasarkan hasil EDA.

**5. Imputasi Rating Pengguna**
Missing values pada kolom 'averageUserRating' diisi dengan nilai 0 untuk menjaga konsistensi data.

**6. Penghapusan Kolom dengan Korelasi Rendah**
Kolom 'appContentBrandSafetyRating' dan 'adSpent' dihapus menggunakan `df.drop(columns=['appContentBrandSafetyRating', 'adSpent'])` karena memiliki banyak missing values dan korelasi rendah dengan target berdasarkan hasil Mutual Information dan Cramer's V.

**7. Imputasi Kolom Kategorikal**
Missing values pada kolom kategorikal `countryCode`, `hasPrivacyLink`, `hasTermsOfServiceLinkRating`, dan `isCorporateEmailScore` diisi dengan string 'undetermine'.

**8. Standardisasi Format Data**
Dilakukan standardisasi format pada kolom 'developerCountry', 'countryCode', dan 'primaryGenreName' dengan mengubah nilai menjadi huruf kapital dan menghapus spasi menggunakan `.str.strip().str.upper()` untuk menjaga konsistensi.

**9. Koreksi Nama Negara**
Perbaikan nama negara yang inkonsisten dilakukan, seperti mengganti 'VIETNAM' menjadi 'VIET NAM' untuk memastikan frequency encoding yang akurat.

**10. Imputasi Numerik dengan MICE**
Missing values pada kolom 'appAge' diisi menggunakan `IterativeImputer` (MICE - Multiple Imputation by Chained Equations) untuk mengisi missing values pada kolom numerik berdasarkan pola data yang ada.

## Data Preprocessing

Tahap preprocessing mempersiapkan data untuk pemodelan dengan melakukan encoding fitur kategorikal dan scaling fitur numerik:

**1. Frequency Encoding**
Frequency encoding diterapkan pada kolom `developerCountry`, `countryCode`, dan `primaryGenreName` menggunakan fungsi `frequency_encode_safe`. Fungsi ini menghitung frekuensi kategori dari data training dan memetakannya ke data training dan testing. Kategori yang tidak ada di data training diisi dengan nilai -1.

**2. Ordinal Encoding**
Beberapa kolom kategorikal dikonversi menggunakan ordinal encoding:
- Kolom 'downloads': Menggunakan fungsi `encode_download_column` untuk mengkonversi rentang unduhan menjadi representasi numerik ordinal (contoh: '0 - 1' menjadi 0, '1 - 5' menjadi 1).
- Kolom 'appDescriptionBrandSafetyRating' dan 'mfaRating': Menggunakan `ordinal_rating_map` dengan mapping 'low': 0, 'medium': 1, 'high': 2.
- Kolom 'isCorporateEmailScore': Dikonversi dengan mapping '0.0': 0, '99.0': 1, 'undetermine': -1.
- Kolom 'hasPrivacyLink': Dikonversi dengan mapping 'False': 0, 'True': 1, 'undetermine': -1.
- Kolom 'hasTermsOfServiceLinkRating': Dikonversi dengan mapping 'low': 0, 'high': 1, 'undetermine': -1.

**3. One-Hot Encoding**
One-hot encoding diterapkan pada kolom 'deviceType' menggunakan `OneHotEncoder` untuk mengubahnya menjadi beberapa kolom biner, karena kolom ini memiliki kategori tanpa urutan intrinsik dan jumlah kategori yang relatif kecil.

**4. Penghapusan Kolom Asli**
Kolom kategorikal asli yang telah di-encode dihapus dari DataFrame untuk menghindari redundansi data.

**5. Pembagian Dataset**
Dataset dipisahkan menjadi training dan testing set menggunakan `train_test_split` dengan parameter `stratify=y` untuk memastikan distribusi kelas target yang sama di kedua set.

**6. Standard Scaling**
Standard scaling diterapkan pada fitur numerik menggunakan `StandardScaler` untuk menormalisasi skala data, yang penting untuk model yang sensitif terhadap skala fitur.

**7. Oversampling dengan SMOTE**
SMOTE (Synthetic Minority Oversampling Technique) diterapkan pada data training yang telah di-scale menggunakan `SMOTE(random_state=42)` untuk mengatasi ketidakseimbangan kelas dalam dataset.

**8. Pemodelan dan Evaluasi**
Setelah preprocessing, dilakukan pelatihan dan evaluasi berbagai model machine learning termasuk XGBoost, LightGBM, Random Forest, SVM, Logistic Regression, serta ensemble Voting dan Stacking Classifiers. Hyperparameter tuning dilakukan menggunakan `GridSearchCV`, dan evaluasi model menggunakan `classification_report`, `roc_auc_score`, dan `confusion_matrix`.

# Evaluation
Submissions will be evaluated based on a suitable classification metric, likely:

AUC (Area Under the ROC Curve)
AUC, or Area Under the ROC Curve, is a single metric that summarizes the overall effectiveness of a classifier. Its value ranges from 0 to 1, where 0.5 suggests the model performs no better than random guessing, and a value of 1 indicates perfect classification performance.
