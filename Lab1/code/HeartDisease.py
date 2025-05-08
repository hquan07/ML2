import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
heart_df = pd.read_csv(url, header=None, names=column_names)

heart_df = heart_df.replace('?', np.nan)

numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']

for col in numeric_cols:
    heart_df[col] = pd.to_numeric(heart_df[col])

for col in categorical_cols:
    heart_df[col] = pd.to_numeric(heart_df[col], errors='coerce')

print("Heart Disease Dataset Overview:")
print(f"Shape: {heart_df.shape}")
print("\nSample data:")
print(heart_df.head())

print("\nData types:")
print(heart_df.dtypes)
print("\nMissing values:")
print(heart_df.isnull().sum())

# Descriptive statistics
print("\nDescriptive statistics:")
print(heart_df.describe())

# Feature classification
print("\nFeature Classification:")
feature_types = []
for column in heart_df.columns:
    is_numeric = pd.api.types.is_numeric_dtype(heart_df[column])
    unique_count = heart_df[column].nunique()

    if column in numeric_cols and unique_count > 10:
        discrete_continuous = "Continuous"
    else:
        discrete_continuous = "Discrete"

    if column in numeric_cols:
        quant_qual = "Quantitative"
    else:
        quant_qual = "Qualitative"

    if column in numeric_cols and unique_count > 10:
        num_cat = "Numerical"
    else:
        num_cat = "Categorical"

    feature_types.append({
        'Feature': column,
        'Discrete/Continuous': discrete_continuous,
        'Quantitative/Qualitative': quant_qual,
        'Numerical/Categorical': num_cat
    })

feature_df = pd.DataFrame(feature_types)
print(feature_df)

imputer = SimpleImputer(strategy='median')
heart_df_imputed = heart_df.copy()

heart_df_imputed[numeric_cols] = imputer.fit_transform(heart_df[numeric_cols])

for col in categorical_cols:
    mode_value = heart_df[col].mode()[0]
    heart_df_imputed[col] = heart_df[col].fillna(mode_value)

print("\nMissing values after imputation:")
print(heart_df_imputed.isnull().sum())

# Calculate correlation matrix
corr_matrix = heart_df_imputed.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_pairs.append({
            'Feature1': corr_matrix.columns[i],
            'Feature2': corr_matrix.columns[j],
            'Correlation': corr_matrix.iloc[i, j]
        })

corr_pairs_df = pd.DataFrame(corr_pairs)
most_correlated_pair = corr_pairs_df.loc[corr_pairs_df['Correlation'].abs().idxmax()]
print("\nMost correlated pair of features:")
print(most_correlated_pair)

# Visualization of correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Heart Disease Features')
plt.tight_layout()
plt.savefig('visualizations/HeartDisease/heart_correlation.png')

# PCA Analysis
features = heart_df_imputed.drop('target', axis=1)
target = heart_df_imputed['target']

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_features)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nPCA Analysis:")
print(f"Explained variance ratio: {explained_variance}")
print(f"Cumulative explained variance: {cumulative_variance}")

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('visualizations/HeartDisease/heart_pca_variance.png')

binary_target = target.apply(lambda x: 0 if x == 0 else 1)

# Visualize data in 2D
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=binary_target, cmap='coolwarm',
                      alpha=0.6, edgecolors='w')
plt.colorbar(scatter, label='Heart Disease Present')
plt.title('Heart Disease Data - First Two Principal Components')
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/HeartDisease/heart_pca_2d.png')

# Analyze PCA with varying number of components
component_count = [2, 3, 5, 8]
reconstruction_errors = []

for n_components in component_count:
    pca_reduced = PCA(n_components=n_components)
    reduced_features = pca_reduced.fit_transform(scaled_features)
    reconstructed_features = pca_reduced.inverse_transform(reduced_features)
    error = np.mean(np.square(scaled_features - reconstructed_features))
    reconstruction_errors.append(error)

    print(f"\nPCA with {n_components} components:")
    print(f"Reconstruction error: {error:.4f}")
    print(f"Explained variance: {np.sum(pca_reduced.explained_variance_ratio_):.4%}")

plt.figure(figsize=(10, 6))
plt.plot(component_count, reconstruction_errors, 'o-', color='purple')
plt.title('Reconstruction Error vs. Number of Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Reconstruction Error')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/HeartDisease/heart_reconstruction_error.png')

pca = PCA(n_components=len(features.columns))
pca_full = pca.fit_transform(scaled_features)

least_significant_idx = np.argsort(pca.explained_variance_)[:2]  # Get indices of 2 lowest variance components
print(f"\nLeast significant component indices: {least_significant_idx}")
print(f"Their explained variance: {pca.explained_variance_ratio_[least_significant_idx]}")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_full[:, least_significant_idx[0]], pca_full[:, least_significant_idx[1]],
                      c=binary_target, cmap='coolwarm', alpha=0.6, edgecolors='w')
plt.colorbar(scatter, label='Heart Disease Present')
plt.title('Heart Disease Data - Two Least Significant Principal Components')
plt.xlabel(f'PC{least_significant_idx[0] + 1} ({pca.explained_variance_ratio_[least_significant_idx[0]]:.2%} variance)')
plt.ylabel(f'PC{least_significant_idx[1] + 1} ({pca.explained_variance_ratio_[least_significant_idx[1]]:.2%} variance)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/HeartDisease/heart_pca_least_significant.png')

def calculate_categorical_stats(dataframe, categorical_columns):
    results = {}

    for col in categorical_columns:
        mode_value = dataframe[col].mode()[0]
        value_counts = dataframe[col].value_counts(normalize=True)
        entropy = -np.sum(value_counts * np.log2(value_counts))

        gini = 1 - np.sum(value_counts ** 2)

        results[col] = {
            'Mode': mode_value,
            'Entropy': entropy,
            'Gini Impurity': gini
        }

    return results

cat_stats = calculate_categorical_stats(heart_df_imputed, categorical_cols)
print("\nStatistics for categorical variables:")
for col, stats in cat_stats.items():
    print(f"{col}: {stats}")

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = pd.DataFrame(confusion_matrix).fillna(0).values
    n = chi2.sum()
    r, k = confusion_matrix.shape

    chi2_stat = np.sum((chi2 - np.outer(chi2.sum(axis=1), chi2.sum(axis=0)) / n) ** 2 /
                       (np.outer(chi2.sum(axis=1), chi2.sum(axis=0)) / n))

    phi2 = chi2_stat / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


cat_associations = {}
for i, col1 in enumerate(categorical_cols[:-1]):  # Exclude target for now
    for col2 in categorical_cols[i + 1:]:
        pair_df = heart_df_imputed[[col1, col2]].dropna()
        if len(pair_df) > 0:
            v = cramers_v(pair_df[col1], pair_df[col2])
            cat_associations[(col1, col2)] = v

sorted_associations = sorted(cat_associations.items(), key=lambda x: x[1], reverse=True)
print("\nStrongest associations between categorical variables (Cramer's V):")
for (col1, col2), v in sorted_associations[:5]:
    print(f"{col1} and {col2}: {v:.4f}")