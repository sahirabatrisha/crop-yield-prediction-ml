from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# Step 1: Remove top 1% production outliers
production_threshold = df['production'].quantile(0.99)
df = df[df['production'] <= production_threshold]

# Step 2: Log-transform planted_area
df['planted_area'] = np.log1p(df['planted_area'])

# Step 3: Group rare crop species as 'other'
species_counts = df['crop_species'].value_counts()
rare_species = species_counts[species_counts < 10].index
df['crop_species'] = df['crop_species'].apply(lambda x: 'other' if x in rare_species else x)

# Step 4: Define features and target
features = ['district', 'crop_species', 'planted_area', 'THI']
target = 'production'

X = df[features]
y = np.log1p(df[target])  # Apply log1p transformation to target

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # Reverse log1p to get actual predictions
    y_true = np.expm1(y_test)      # Reverse actual values

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    results.append({
        'Model': name,
        'R² Score': round(r2, 4),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4)
    })

# Show sorted model performance
results_df = pd.DataFrame(results).sort_values(by='R² Score', ascending=False)
print("Model Performance (Cleaned Data with Fixes):")
print(results_df)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Preprocessing pipeline
categorical = ['district', 'crop_species']
numeric = ['planted_area', 'THI']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', StandardScaler(), numeric)
])

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
import time

# Define the pipeline
svr_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', SVR())
])

# Define a more manageable parameter grid (you can expand later)
param_grid = {
    'regressor__kernel': ['rbf', 'poly'],
    'regressor__C': [1, 10, 100],
    'regressor__gamma': ['scale', 0.01, 0.1],
    'regressor__epsilon': [0.1, 0.2]
}

# Start timing
start_time = time.time()

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(
    svr_pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2  # Show progress
)

grid_search.fit(X_train, y_train)

# End timing
end_time = time.time()

# Print best parameters and score
print("🔍 Best Parameters for SVR:")
print(grid_search.best_params_)

print("\n📈 Best R² Score from Grid Search:")
print(round(grid_search.best_score_, 4))

# Print timing
print(f"\n⏱️ Time taken: {round(end_time - start_time, 2)} seconds")

