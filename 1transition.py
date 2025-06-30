import pandas as pd
import numpy as np
import re

# Load the data
df = pd.read_excel('rawdata.xlsx')

# Clean and prepare data
df['Current Stage'] = df['Current Stage'].astype(str)
df['Previous Stage'] = df['Previous Stage'].astype(str)
df['Outstanding Principal'] = pd.to_numeric(df['Outstanding Principal'])

# Extract quarter from date
df['Date'] = pd.to_datetime(df['Date'])
df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)

# Clean product names for Excel sheet safety
def clean_sheet_name(name):
    """Remove invalid Excel sheet name characters"""
    return re.sub(r'[\\/*?:[\]]', '', name)[:31]

# Create transition matrix function (fixed float initialization)
def calculate_weighted_transition_matrix(df):
    stages = ['1', '2', '3', 'New']
    # Initialize as float to avoid dtype issues
    transition_weights = pd.DataFrame(0.0, index=stages, columns=stages)
    
    if df.empty:
        transition_matrix = transition_weights.copy()
        return transition_matrix, transition_weights
    
    # Group by previous and current stage, sum outstanding amounts
    grouped = df.groupby(['Previous Stage', 'Current Stage'])['Outstanding Principal'].sum().unstack(fill_value=0)
    
    # Fill the transition matrix
    for prev_stage in stages:
        for curr_stage in stages:
            if prev_stage in grouped.index and curr_stage in grouped.columns:
                transition_weights.loc[prev_stage, curr_stage] = grouped.loc[prev_stage, curr_stage]
    
    # Calculate transition probabilities
    row_sums = transition_weights.sum(axis=1)
    # Handle zero sums to avoid division errors
    transition_matrix = transition_weights.div(row_sums.replace(0, 1), axis=0).fillna(0)
    
    return transition_matrix, transition_weights

# Generate all possible transition combinations
stages = ['1', '2', '3', 'New']
transitions = [f'Stage {s1} to {s2}' for s1 in stages for s2 in stages]

# Create final result containers
final_weights = pd.DataFrame(columns=['Quarter', 'Product Type'] + transitions)
final_probs = pd.DataFrame(columns=['Quarter', 'Product Type'] + transitions)

# Process data by quarter and product
quarters = sorted(df['Quarter'].unique())
products = sorted(df['Loan Product'].unique())

for quarter in quarters:
    q_df = df[df['Quarter'] == quarter]
    
    for product in products:
        # Filter for current quarter and product
        product_df = q_df[q_df['Loan Product'] == product]
        
        if len(product_df) == 0:
            continue
            
        # Calculate matrices
        matrix, weights = calculate_weighted_transition_matrix(product_df)
        
        # Prepare row data
        weight_row = {'Quarter': quarter, 'Product Type': product}
        prob_row = {'Quarter': quarter, 'Product Type': product}
        
        # Add all transition values
        for i, s1 in enumerate(stages):
            for j, s2 in enumerate(stages):
                trans_name = f'Stage {s1} to {s2}'
                weight_row[trans_name] = weights.loc[s1, s2]
                prob_row[trans_name] = matrix.loc[s1, s2]
        
        # Add to final results
        final_weights = pd.concat([final_weights, pd.DataFrame([weight_row])], ignore_index=True)
        final_probs = pd.concat([final_probs, pd.DataFrame([prob_row])], ignore_index=True)

# Add overall quarter summary
for quarter in quarters:
    q_df = df[df['Quarter'] == quarter]
    if len(q_df) == 0:
        continue
            
    matrix, weights = calculate_weighted_transition_matrix(q_df)
    
    weight_row = {'Quarter': quarter, 'Product Type': 'All Products'}
    prob_row = {'Quarter': quarter, 'Product Type': 'All Products'}
    
    for i, s1 in enumerate(stages):
        for j, s2 in enumerate(stages):
            trans_name = f'Stage {s1} to {s2}'
            weight_row[trans_name] = weights.loc[s1, s2]
            prob_row[trans_name] = matrix.loc[s1, s2]
    
    final_weights = pd.concat([final_weights, pd.DataFrame([weight_row])], ignore_index=True)
    final_probs = pd.concat([final_probs, pd.DataFrame([prob_row])], ignore_index=True)

# Add overall product summary
for product in products:
    product_df = df[df['Loan Product'] == product]
    if len(product_df) == 0:
        continue
            
    matrix, weights = calculate_weighted_transition_matrix(product_df)
    
    weight_row = {'Quarter': 'All Quarters', 'Product Type': product}
    prob_row = {'Quarter': 'All Quarters', 'Product Type': product}
    
    for i, s1 in enumerate(stages):
        for j, s2 in enumerate(stages):
            trans_name = f'Stage {s1} to {s2}'
            weight_row[trans_name] = weights.loc[s1, s2]
            prob_row[trans_name] = matrix.loc[s1, s2]
    
    final_weights = pd.concat([final_weights, pd.DataFrame([weight_row])], ignore_index=True)
    final_probs = pd.concat([final_probs, pd.DataFrame([prob_row])], ignore_index=True)

# Add grand total summary
matrix, weights = calculate_weighted_transition_matrix(df)

weight_row = {'Quarter': 'All Quarters', 'Product Type': 'All Products'}
prob_row = {'Quarter': 'All Quarters', 'Product Type': 'All Products'}

for i, s1 in enumerate(stages):
    for j, s2 in enumerate(stages):
        trans_name = f'Stage {s1} to {s2}'
        weight_row[trans_name] = weights.loc[s1, s2]
        prob_row[trans_name] = matrix.loc[s1, s2]

final_weights = pd.concat([final_weights, pd.DataFrame([weight_row])], ignore_index=True)
final_probs = pd.concat([final_probs, pd.DataFrame([prob_row])], ignore_index=True)

# Export to Excel
with pd.ExcelWriter('transition_matrices.xlsx') as writer:
    # Format numeric columns
    num_format = '#,##0.00'
    
    # Write weights
    final_weights.to_excel(writer, sheet_name='Transition Amounts', index=False)
    ws = writer.sheets['Transition Amounts']
    for col in range(2, len(transitions) + 2):  # Start from column C
        ws.set_column(col, col, 15, writer.book.add_format({'num_format': num_format}))
    
    # Write probabilities
    final_probs.to_excel(writer, sheet_name='Transition Probabilities', index=False)
    ws = writer.sheets['Transition Probabilities']
    for col in range(2, len(transitions) + 2):
        ws.set_column(col, col, 15, writer.book.add_format({'num_format': '0.00%'}))

print("Transition matrices successfully exported to 'transition_matrices.xlsx'")
print("Sheets created:")
print("- Transition Amounts: Contains outstanding principal amounts for each transition")
print("- Transition Probabilities: Contains probability percentages for each transition")