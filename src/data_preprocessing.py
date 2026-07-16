# Crop Type Comparison
area_crop_types = set(area_df['crop_type'].dropna().unique())
prod_crop_types = set(prod_df['crop_type'].dropna().unique())

print("Unique crop_type in area_df:", sorted(area_crop_types))
print("Unique crop_type in prod_df:", sorted(prod_crop_types))

print("\nCrop types only in area_df:", sorted(area_crop_types - prod_crop_types))
print("Crop types only in prod_df:", sorted(prod_crop_types - area_crop_types))

# Crop Species Comparison
area_crop_species = set(area_df['crop_species'].dropna().unique())
prod_crop_species = set(prod_df['crop_species'].dropna().unique())

print("\nUnique crop_species in area_df:", sorted(area_crop_species))
print("Unique crop_species in prod_df:", sorted(prod_crop_species))

print("\nCrop species only in area_df:", sorted(area_crop_species - prod_crop_species))
print("Crop species only in prod_df:", sorted(prod_crop_species - area_crop_species))

def clean_and_merge_crop_data(area_df, prod_df, climate_df):
    import pandas as pd

    # Ensure consistent column naming for merging
    climate_df = climate_df.rename(columns={'State': 'state'})

    # 1. Normalize key text fields
    def normalize(df):
        for col in ['crop_type', 'crop_species', 'state', 'district']:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .str.replace(' ', '_')
                )
        return df

    area_df = normalize(area_df)
    prod_df = normalize(prod_df)
    climate_df = normalize(climate_df)

    # 2. Standardize known crop_type and crop_species inconsistencies
    prod_df['crop_type'] = prod_df['crop_type'].replace({'vegetable': 'vegetables'})
    area_df['crop_species'] = area_df['crop_species'].replace({'luffa': 'angled_loofah'})

    # 3. Remove unwanted crop species
    excluded_species = ['chinese_potato', 'mint', 'snake_gourd']
    area_df = area_df[~area_df['crop_species'].isin(excluded_species)]
    prod_df = prod_df[~prod_df['crop_species'].isin(excluded_species)]

    # 4. Merge area and production data
    merged_df = pd.merge(
        area_df,
        prod_df,
        on=['date', 'state', 'district', 'crop_type', 'crop_species'],
        how='inner'
    )

    # 5. Drop rows with missing or zero planted area
    merged_df = merged_df.dropna(subset=['planted_area', 'production'])
    merged_df = merged_df[merged_df['planted_area'] > 0]

    # 6. Merge with climate data
    final_df = pd.merge(merged_df, climate_df, on='state', how='left')

    return final_df

# Clean and merge data with climate
df = clean_and_merge_crop_data(area_df, prod_df, climate_df)

print("Merged clean data shape:", df.shape)
print(df.head())

# Save full dataset to CSV
df.to_csv('clean_crop_production_data.csv', index=False)

# Allow download in Google Colab
from google.colab import files
files.download('clean_crop_production_data.csv')

