# Data Analysis and Principal Component Analysis (PCA) Project

## Overview

This project performs comprehensive data analysis and dimensionality reduction using Principal Component Analysis (PCA) on two different datasets:
1. **Wine Quality Dataset** - Chemical properties and quality ratings of wines
2. **Heart Disease Dataset** - Medical attributes related to heart disease diagnosis

The analysis includes feature classification, statistical measures, correlation analysis, data preprocessing, and PCA with varying component configurations.

## Datasets

### Wine Quality Dataset
- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/wine+quality
- **Features**: 11 physicochemical properties (input) + wine quality rating (output)
- **Samples**: 1,599 red wine samples
- **Target**: Quality score (0-10 scale)

### Heart Disease Dataset
- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/heart+disease
- **Features**: 13 medical attributes + heart disease diagnosis
- **Samples**: 303 patient records
- **Target**: Presence of heart disease (0 = no disease, 1-4 = disease present)

## Project Structure

```
├── data_analysis_report.md       # Comprehensive analysis report
├── README.md                     # This file
├── code/
│   ├── wine_quality_analysis.py  # Wine dataset analysis and PCA
│   ├── heart_disease_analysis.py # Heart disease dataset analysis and PCA
├── visualizations/
│   ├── wine_correlation.png      # Correlation heatmap for wine dataset
│   ├── wine_pca_variance.png     # Explained variance for wine PCA
│   ├── wine_pca_2d.png           # 2D PCA visualization for wine dataset
│   ├── heart_correlation.png     # Correlation heatmap for heart disease dataset
│   ├── heart_pca_variance.png    # Explained variance for heart disease PCA
│   ├── heart_pca_2d.png          # 2D PCA visualization for heart disease dataset
```

## Features and Implementation

### Data Analysis
- Feature classification (discrete/continuous, quantitative/qualitative, numerical/categorical)
- Missing value detection and handling
- Descriptive statistics calculation
- Correlation analysis and visualization
- Data quality assessment

### Principal Component Analysis
- Standardization of features
- PCA application with full component extraction
- Explained variance analysis and visualization
- Component selection methodology
- Comparison between highest and lowest variance components
- Reconstruction error analysis with varying component counts
- 2D data visualization using different component combinations

## Key Findings

### Wine Quality Dataset
- No missing values, clean data
- Most features are continuous and quantitative
- Optimal PCA configuration: 5 components (81% variance)
- Strongest correlations: density-fixed acidity (0.67), alcohol-density (-0.50)
- 2D visualization shows meaningful patterns related to wine quality

### Heart Disease Dataset
- Missing values in 'ca' and 'thal' columns
- Mix of continuous and categorical features
- Optimal PCA configuration: 8 components (88% variance)
- Strongest correlations: thalach-age (-0.40), oldpeak-exang (0.39)
- 2D visualization shows moderate separation between disease classes

## Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: PCA implementation and preprocessing
- **Matplotlib/Seaborn**: Data visualization
- **SciPy**: Statistical calculations

## Running the Code

### Requirements
- Python 3.6+
- Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

### Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execution
```bash
python code/wine_quality_analysis.py
python code/heart_disease_analysis.py
```

## Results and Conclusions

1. **PCA Effectiveness**: PCA effectively reduces dimensionality while preserving essential data structure for both datasets.

2. **Dataset Complexity**: Wine Quality dataset has more concentrated information distribution (fewer components needed), while Heart Disease dataset shows more complex relationships requiring more components.

3. **Component Selection**: The optimal number of components is dataset-specific:
   - Wine Quality: 5 components (balance between dimensionality reduction and information preservation)
   - Heart Disease: 8 components (higher complexity requires more components)

4. **Practical Applications**:
   - Wine Quality: Model simplification with minimal information loss
   - Heart Disease: Dimensionality reduction while maintaining diagnostic power

5. **Limitations**: Linear nature of PCA may miss non-linear relationships, and categorical variables require special handling.

## Future Work

- Explore non-linear dimensionality reduction techniques (t-SNE, UMAP)
- Apply supervised dimensionality reduction methods
- Investigate feature engineering tailored to each domain
- Develop hybrid approaches for mixed variable types
- Implement predictive models using the reduced dimensions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the datasets
- The scikit-learn team for their excellent implementation of PCA and preprocessing tools
- The pandas and matplotlib development teams for their data analysis and visualization libraries