from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# Predict using the best estimator from GridSearch
best_model = grid_search.best_estimator_
y_pred_log = best_model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Reverse log transform
y_true = np.expm1(y_test)      # Reverse log transform on true values

# Evaluation metrics
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("\n📊 Evaluation on Test Set (Best SVR):")
print(f"R² Score: {r2:.4f}")
print(f"MAE     : {mae:.2f}")
print(f"RMSE    : {rmse:.2f}")

# Show sample of predicted vs actual values
print("\n🔍 Predicted vs Actual (Sample):")
comparison_df = pd.DataFrame({
    'Actual Production': y_true,
    'Predicted Production': y_pred
}).reset_index(drop=True)

# Show first 10 rows
print(comparison_df.head(10).to_string(index=False))

import matplotlib.pyplot as plt

# Plot R² Score
plt.figure(figsize=(10, 5))
plt.barh(results_df['Model'], results_df['R² Score'], color='skyblue')
plt.xlabel("R² Score")
plt.title("Model Comparison – R² Score")
plt.gca().invert_yaxis()
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()

# Plot MAE
plt.figure(figsize=(10, 5))
plt.barh(results_df['Model'], results_df['MAE'], color='salmon')
plt.xlabel("MAE")
plt.title("Model Comparison – Mean Absolute Error")
plt.gca().invert_yaxis()
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()

# Plot RMSE
plt.figure(figsize=(10, 5))
plt.barh(results_df['Model'], results_df['RMSE'], color='lightgreen')
plt.xlabel("RMSE")
plt.title("Model Comparison – Root Mean Squared Error")
plt.gca().invert_yaxis()
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()

districts = df['district'].unique()
crop_species = df['crop_species'].unique()

# Calculate average planted_area and THI for each district
district_avg_features = df.groupby('district')[['planted_area', 'THI']].mean().reset_index()

best_crops = {}

print("Analysing best crop for each district...")

for district in districts:

    avg_features = district_avg_features[district_avg_features['district'] == district]
    if avg_features.empty:
        print(f"Warning: No average features found for district {district}. Skipping.")
        continue

    avg_planted_area = avg_features.iloc[0]['planted_area']
    avg_THI = avg_features.iloc[0]['THI']

    predictions_for_district = {}

    for species in crop_species:

        hypothetical_input = pd.DataFrame([{
            'district': district,
            'crop_species': species,
            'planted_area': avg_planted_area,
            'THI': avg_THI
        }])

        try:
            predicted_production_log = best_model.predict(hypothetical_input)
            predicted_production = np.expm1(predicted_production_log)[0] # inverse transform and get the single value
            predictions_for_district[species] = predicted_production
        except Exception as e:
            # handle cases where a specific district/species combination might cause issues
            print(f"Error predicting for district {district}, species {species}: {e}")
            predictions_for_district[species] = 0 # Assign 0 or some other indicator

    # find the crop with the maximum predicted production in this district
    if predictions_for_district:
        # filter out crops with 0 prediction if necessary, or find max among all
        best_crop_in_district = max(predictions_for_district, key=predictions_for_district.get)
        best_crops[district] = {
            'best_crop': best_crop_in_district,
            'predicted_production': predictions_for_district[best_crop_in_district]
        }
    else:
         best_crops[district] = {
            'best_crop': 'N/A',
            'predicted_production': 0
        }


print("\nBest Crop by District (Based on predicted production with average features):")

best_crops_df = pd.DataFrame.from_dict(best_crops, orient='index')
best_crops_df = best_crops_df.reset_index().rename(columns={'index': 'district'})

print(best_crops_df.to_string(index=False))

# Save the results to a CSV file
output_filename = 'best_crop_by_district.csv'
best_crops_df.to_csv(output_filename, index=False)

print(f"\nResults saved to {output_filename}")

try:
    from google.colab import files
    files.download(output_filename)
except ImportError:
    print("google.colab not found. Skipping file download.")

