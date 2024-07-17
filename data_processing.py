import pandas as pd
import glob

def read_file_with_encoding(file_path, encoding='utf-8'):
    try:
        return pd.read_csv(file_path, encoding=encoding, usecols=['Q', 'SD', 'D', 'A', 'SA'])
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='ISO-8859-1', usecols=['Q', 'SD', 'D', 'A', 'SA'])

def load_data(dataset_dir):
    all_files = glob.glob(dataset_dir + "*.csv")
    dfs = [read_file_with_encoding(file) for file in all_files]
    return pd.concat(dfs, ignore_index=True)

def analyze_responses(combined_df):
    unique_questions = combined_df['Q'].unique()
    question_responses = {question: {'responses': [], 'most_common': None} for question in unique_questions}

    for question in unique_questions:
        responses = combined_df[combined_df['Q'] == question][['SD', 'D', 'A', 'SA']].dropna(how='all')
        if not responses.empty:
            selected_responses = responses.idxmax(axis=1)
            question_responses[question]['responses'] = selected_responses.tolist()
            question_responses[question]['most_common'] = selected_responses.mode().iloc[0]
    
    return question_responses