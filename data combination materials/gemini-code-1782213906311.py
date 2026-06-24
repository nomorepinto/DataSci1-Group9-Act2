import pandas as pd

# Define the input and output file names
input_file = 'combined.csv'  # Change this to your actual input file name
output_file = '2021-2025PhilippineExports.csv'

try:
    # Load the CSV file
    df = pd.read_csv(input_file)
    
    # Filter the DataFrame for entries from 2021 onwards
    # (This assumes the 'Year' column contains numeric values like 2020, 2021, etc.)
    filtered_df = df[df['Year'] >= 2021]
    
    # Save the filtered data to a new CSV file without the row index
    filtered_df.to_csv(output_file, index=False)
    
    print(f"Successfully filtered data! Saved rows from 2021 onwards to '{output_file}'.")
    print(f"Original row count: {len(df)}")
    print(f"Filtered row count: {len(filtered_df)}")

except FileNotFoundError:
    print(f"Error: The file '{input_file}' could not be found. Please check the file path.")
except KeyError:
    print("Error: The column 'Year' was not found in the CSV file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")