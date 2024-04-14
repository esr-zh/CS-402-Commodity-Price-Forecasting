import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
commodity = 'coffee'

df = pd.read_csv(f'{commodity}_data_c.csv')

# Split the dataset into training and testing sets (80-20)
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# Function to normalize and write a subset of the dataset to a TS file
def normalize_and_write_to_file(df, file_name):
    # Reset index to ensure alignment
    df = df.reset_index(drop=True)
    
    # Select columns to normalize
    features = ['Open', 'High', 'Low', 'Volume']
    X = df[features]

    # Normalize the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_normalized = scaler.fit_transform(X)

    # Combine the normalized features with the OT column
    df_normalized = pd.DataFrame(X_normalized, columns=features)
    df_normalized['OT'] = df['OT'].values  # Ensure alignment by directly assigning values

    # Define the TS file format
    def format_ts(row):
        data_str = ','.join(map(str, row[:-1]))  # Convert all but the last column (OT) to string
        return f"{data_str}:{row['OT']}\n"

    # Write to a TS file
    with open(file_name, 'w') as f:
        f.write("@problemName coffee\n")
        f.write("@timeStamps false\n")
        f.write("@missing false\n")
        f.write("@univariate false\n")
        f.write(f"@dimensions {len(features)}\n")
        f.write("@equalLength true\n")
        f.write(f"@seriesLength {len(features)}\n")
        f.write("@classLabel true " + ' '.join(map(str, sorted(df['OT'].unique()))) + "\n")
        f.write("@data\n")
        for index, row in df_normalized.iterrows():
            f.write(format_ts(row))
            
# Normalize and write training and testing sets to separate files
normalize_and_write_to_file(train_df, f'./{commodity}/{commodity}_TRAIN.ts')
normalize_and_write_to_file(test_df, f'./{commodity}/{commodity}_TEST.ts')
