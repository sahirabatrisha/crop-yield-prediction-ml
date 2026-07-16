from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# Load area and production data
area_df = pd.read_csv('/content/drive/MyDrive/WIA1006/crops_district_area.csv')
prod_df = pd.read_csv('/content/drive/MyDrive/WIA1006/crops_district_production.csv')

# Load climate data from CSV (e.g., climate_by_state.csv)
climate_df = pd.read_csv('/content/drive/MyDrive/WIA1006/climate_by_state.csv')

# Preview samples
print("Area data sample:")
print(area_df.head().to_string(index=False))

print("\nProduction data sample:")
print(prod_df.head().to_string(index=False))

print("\nClimate variables by state:")
print(climate_df.to_string(index=False))

