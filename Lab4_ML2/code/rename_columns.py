import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("C:/ML/Labwork4/dataset/RandomForest/adult/adult.csv", header=None)

# Gán tên cột
df.columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

# Lưu lại
df.to_csv("C:/ML/Labwork4/dataset/RandomForest/adult/adult_renamed.csv", index=False)
