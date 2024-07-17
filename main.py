import pandas as pd
import csv
import unicodedata

def clean_text(text):
    if isinstance(text, str):
        # Normalize Unicode characters
        normalized_text = unicodedata.normalize('NFKD', text)
        
        # Replace various types of quotation marks with standard single quotes
        quotation_marks = [''', ''', '′', '`', '´', ''', ''']
        for mark in quotation_marks:
            normalized_text = normalized_text.replace(mark, "'")
        
        # Remove any remaining non-ASCII characters
        cleaned_text = ''.join(char for char in normalized_text if ord(char) < 128)
        
        return cleaned_text
    return text

def read_csv_file(file_path):
    # Try different encodings
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, quoting=csv.QUOTE_ALL)
            # Apply clean_text to all string columns
            for column in df.select_dtypes(include=['object']):
                df[column] = df[column].apply(clean_text)
            return df
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Unable to read the CSV file with any of the attempted encodings: {encodings}")

# Usage
file_path = '/home/minjilee/Desktop/political_compass/dataset/Political_Compass_Dataset(Sheet1).csv'
import pandas as pd
import csv
import unicodedata
import matplotlib.pyplot as plt

# Your existing code for clean_text and read_csv_file functions here

# Usage
file_path = '/home/minjilee/Desktop/political_compass/dataset/Political_Compass_Dataset(Sheet1).csv'
try:
    df = read_csv_file(file_path)
    print(df.head())

    # Plotting
    plt.figure(figsize=(12, 8))

    # Assuming your CSV has 'x' and 'y' columns for the political compass coordinates
    plt.scatter(df['x'], df['y'], alpha=0.6)
    
    plt.title('Political Compass Scatter Plot')
    plt.xlabel('Economic Left/Right')
    plt.ylabel('Social Libertarian/Authoritarian')
    
    # Add a vertical and horizontal line to divide the plot into quadrants
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axvline(x=0, color='k', linestyle='--')
    
    # Set equal scaling and show the plot
    plt.axis('equal')
    plt.grid(True)
    plt.show()

except ValueError as e:
    print(f"Error: {e}")