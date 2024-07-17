import pandas as pd

def calculate_consistency(question_responses):
    consistency = {}
    most_common_responses = {}
    significant_changes = []

    for question, data in question_responses.items():
        responses = data['responses']
        most_common = data['most_common']
        if responses:
            consistency[question] = responses.count(most_common) / len(responses)
            most_common_responses[question] = most_common
            
            # Check for significant changes
            for i in range(1, len(responses)):
                prev_response = responses[i-1]
                current_response = responses[i]
                if (prev_response == 'D' and current_response == 'SA') or (prev_response == 'A' and current_response == 'SD') or (prev_response == 'SD' and current_response == 'A') or (prev_response == 'SA' and current_response == 'D'):
                    significant_changes.append({'Question': question, 'Run': i, 'Previous': prev_response, 'Current': current_response})
        else:
            consistency[question] = 0
            most_common_responses[question] = None
    
    return consistency, most_common_responses, significant_changes

def prepare_consistency_df(consistency, most_common_responses):
    consistency_df = pd.DataFrame({
        'Question': list(consistency.keys()),
        'Consistency': list(consistency.values()),
        'Most Common Response': [most_common_responses[q] for q in consistency.keys()]
    }).dropna(subset=['Question'])
    
    consistency_df['Short Question'] = ['Q' + str(i+1) for i in range(len(consistency_df))]
    return consistency_df