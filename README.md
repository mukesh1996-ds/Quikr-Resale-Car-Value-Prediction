# Car Price Prediction Model

## Overview
This project involves creating a machine learning model to predict the price of used cars based on their attributes. The dataset used for training was sourced from Quikr, and the final model is deployed using a pipeline for preprocessing and prediction.

---

## Dataset Information
- **Source**: `quikr_car.csv`
- **Columns**:
  - `name`: Name of the car model
  - `company`: Manufacturer of the car
  - `year`: Year of manufacture
  - `Price`: Price of the car (target variable)
  - `kms_driven`: Kilometers the car has been driven
  - `fuel_type`: Fuel type of the car

---

## Data Cleaning and Preprocessing
The dataset required significant cleaning to make it suitable for modeling. Below are the steps performed:

1. **Fixing `year` Column**:
   - Retained only numeric values.
   - Converted to integer datatype.

2. **Handling `Price` Column**:
   - Removed rows with `Ask For Price`.
   - Converted values to integers after removing commas.

3. **Fixing `kms_driven` Column**:
   - Extracted numeric values.
   - Converted to integer datatype.
   - Removed rows with NaN values.

4. **Handling `fuel_type` Column**:
   - Dropped rows with missing values.

5. **Simplifying `name` Column**:
   - Kept the first three words of the car name.

6. **Filtering Outliers**:
   - Removed entries with a price greater than 6 million.

The cleaned dataset was saved as `Cleaned_data_car.csv`.

---

## Model Training

### Features and Target
- **Features**: `name`, `company`, `year`, `kms_driven`, `fuel_type`
- **Target**: `Price`

### Preprocessing
- Used `OneHotEncoder` for categorical variables (`name`, `company`, `fuel_type`).
- Combined preprocessing and model training using `make_pipeline` and `make_column_transformer`.

### Model
- Linear Regression was chosen for the task.

### Training and Validation
- Split the data into training and testing sets using an 80-20 split.
- Iterated the train-test split 1000 times with different random states to evaluate model consistency.
- Achieved the best R² score using a specific random state.

---

## Model Deployment
The final trained model was saved as a pickle file (`linearegressionmodel.pkl`).

### Prediction Example
The following example shows how to use the model for prediction:

```python
import pandas as pd
import pickle

# Load the model
model = pickle.load(open("linearegressionmodel.pkl", 'rb'))

# Predict car price
data = pd.DataFrame([["Maruti Suzuki Swift", "Maruti", 2019, 100, "Petrol"]],
                    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
price = model.predict(data)
print(f"Predicted Price: {price[0]}")
```

---

## Streamlit App
A Streamlit app was developed for real-time user interaction with the model.

### Features
1. Input car details:
   - Car Name
   - Car Company
   - Year of Manufacture
   - Kilometers Driven
   - Fuel Type
2. Predicts and displays the car price.

### Running the App
1. Ensure the `linearegressionmodel.pkl` file is in the project directory.
2. Install required libraries:
   ```bash
   pip install streamlit pandas scikit-learn
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

### Error Handling
The app validates user inputs against the categories present in the training dataset. If an unsupported category is entered, the app displays an error message.

---

## Files in the Repository
1. `quikr_car.csv`: Raw dataset.
2. `Cleaned_data_car.csv`: Cleaned dataset.
3. `linearegressionmodel.pkl`: Trained Linear Regression model.
4. `app.py`: Streamlit application for deployment.

---