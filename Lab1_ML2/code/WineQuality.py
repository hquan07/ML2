import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

os.makedirs('visualizations/WineQuality', exist_ok=True)

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_df = pd.read_csv(url, sep=';')

print("Wine Quality Dataset Overview:")
print(f"Shape: {wine_df.shape}")
print("\Sample data:")
print(wine_df.head())

print("\nData types:")
print(wine_df.dtypes)
print("\nMissing values:")
print(wine_df.isnull().sum())

print("\nDescriptiive statistics:")
print(wine_df.describe())

print("\Fearture Classification:")
feature_types = []
for column in wine_df.columns:
    is_numeric = pd.api.types.is_numeric_dtype(wine_df[column])
    unique_count = wine_df[column].nunique()

    if is_numeric:
        if unique_count < 20:  # Arbitrary threshold
            discrete_continuous = "Discrete"
        else:
            discrete_continuous = "Continuous"
    else:
        discrete_continuous = "Discrete"

    if is_numeric:
        quant_qual = "Quantitative"
    else:
        quant_qual = "Qualitative"

    if is_numeric and not (wine_df[column].dtype == 'int64' and unique_count < 10):
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

# Calculate mean for each feature
feature_means = wine_df.mean()
print("\nFeature Means:")
print(feature_means)

# Calculate variance for each feature
feature_variances = wine_df.var()
print("\nFeature Variances:")
print(feature_variances)

# Calculate covariance matrix
covariance_matrix = wine_df.cov()
print("\nCovariance Matrix:")
print(covariance_matrix)

# Visualize mean and variance
plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 2, 1)
feature_means.plot(kind='bar', ax=ax1)
plt.title('Mean Values of Wine Features')
plt.ylabel('Mean')
plt.xticks(rotation=90)

ax2 = plt.subplot(1, 2, 2)
feature_variances.plot(kind='bar', ax=ax2)
plt.title('Variance of Wine Features')
plt.ylabel('Variance')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('visualizations/WineQuality/wine_mean_variance.png')

# Visualize covariance matrix
plt.figure(figsize=(12, 10))
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Covariance Matrix of Wine Quality Features')
plt.tight_layout()
plt.savefig('visualizations/WineQuality/wine_covariance.png')

# Calculate correlation matrix
corr_matrix = wine_df.corr()
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

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Wine Quality Features')
plt.tight_layout()
plt.savefig('visualizations/WineQuality/wine_correlation.png')

# PCA Analysis
features = wine_df.drop('quality', axis=1)
target = wine_df['quality']

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_features)

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
plt.savefig('visualizations/WineQuality/wine_pca_variance.png')

# Visualize data in 2D
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=target, cmap='viridis',
                      alpha=0.6, edgecolors='w')
plt.colorbar(scatter, label='Wine Quality')
plt.title('Wine Quality Data - First Two Principal Components')
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/WineQuality/wine_pca_2d.png')

# Analyze PCA with varying number of components
component_count = [2, 3, 5, 7]
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
plt.savefig('visualizations/WineQuality/wine_reconstruction_error.png')

# Try PCA with lowest eigenvalues
pca = PCA(n_components=len(features.columns))
pca_full = pca.fit_transform(scaled_features)

least_significant_idx = np.argsort(pca.explained_variance_)[:2]
print(f"\nLeast significant component indices: {least_significant_idx}")
print(f"Their explained variance: {pca.explained_variance_ratio_[least_significant_idx]}")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_full[:, least_significant_idx[0]], pca_full[:, least_significant_idx[1]],
                      c=target, cmap='viridis', alpha=0.6, edgecolors='w')
plt.colorbar(scatter, label='Wine Quality')
plt.title('Wine Quality Data - Two Least Significant Principal Components')
plt.xlabel(f'PC{least_significant_idx[0] + 1} ({pca.explained_variance_ratio_[least_significant_idx[0]]:.2%} variance)')
plt.ylabel(f'PC{least_significant_idx[1] + 1} ({pca.explained_variance_ratio_[least_significant_idx[1]]:.2%} variance)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/WineQuality/wine_pca_least_significant.png')