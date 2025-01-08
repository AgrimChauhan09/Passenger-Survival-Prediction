# Passenger Survival Prediction

This project predicts the survival of passengers on the Titanic using machine learning techniques. The dataset used for this project includes passenger details such as age, sex, class, and fare, among others. The prediction model achieves an accuracy of **84.92%**.

## Libraries and Tools Used
The following Python libraries are used in this project:

```python
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

## Dataset
The datasets used are:
1. **train.csv**: Contains the training data for the model.
2. **test.csv**: Contains the test data for evaluating the model.

## Project Workflow

1. **Data Loading**:
   Load the `train.csv` and `test.csv` files using pandas.

   ```python
   warnings.filterwarnings('ignore')
   train_data = pd.read_csv('train.csv')
   test_data = pd.read_csv('test.csv')
   ```

2. **Exploratory Data Analysis (EDA)**:
   Use `matplotlib` and `seaborn` for visualizing the data to understand the distributions, missing values, and relationships between features.

3. **Data Preprocessing**:
   - Handle missing values.
   - Encode categorical variables.
   - Normalize or scale numerical features if needed.

4. **Feature Selection**:
   Select features that are relevant for predicting survival, such as:
   - `Pclass`
   - `Sex`
   - `Age`
   - `Fare`
   - `Embarked`

5. **Model Training**:
   Train a `RandomForestClassifier` using the training dataset.

   ```python
   X = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
   y = train_data['Survived']

   # Convert categorical features to numerical
   X = pd.get_dummies(X, drop_first=True)
   
   # Split data into training and validation sets
   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

   # Train the Random Forest Classifier
   model = RandomForestClassifier(random_state=42)
   model.fit(X_train, y_train)
   ```

6. **Evaluation**:
   Evaluate the model using the validation set and calculate the accuracy.

   ```python
   y_pred = model.predict(X_val)
   accuracy = accuracy_score(y_val, y_pred)
   print(f'Validation Accuracy: {accuracy * 100:.2f}%')
   ```

7. **Prediction**:
   Use the trained model to predict survival for the test dataset.

## Results
The model achieves an accuracy of **84.92%** on the validation set.

## Visualizations
- **Survival Rate by Class**
  ```python
  sns.barplot(x='Pclass', y='Survived', data=train_data)
  plt.title('Survival Rate by Passenger Class')
  plt.show()
  ```

- **Survival Rate by Gender**
  ```python
  sns.barplot(x='Sex', y='Survived', data=train_data)
  plt.title('Survival Rate by Gender')
  plt.show()
  ```

## Usage
1. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. Place the `train.csv` and `test.csv` files in the project directory.

3. Run the Python script to train the model and generate predictions.

4. Modify the code to explore additional features or try different machine learning models to improve accuracy.

