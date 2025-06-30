import pandas as pd
import numpy as np
import re
from datetime import datetime

# Load the data
df = pd.read_excel('rawdata.xlsx')

# Clean and prepare data
df['Current Stage'] = df['Current Stage'].astype(str)
df['Previous Stage'] = df['Previous Stage'].astype(str)
df['Outstanding Principal'] = pd.to_numeric(df['Outstanding Principal'])

# Extract quarter from date
df['Date'] = pd.to_datetime(df['Date'])
df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)

# Define stages
stages = ['1', '2', '3', 'New']
transitions = [f'Stage {s1} to {s2}' for s1 in stages for s2 in stages]

# Function to determine fiscal year (July 17 to July 16)
def get_fiscal_year(date):
    if date.month == 7 and date.day >= 17:
        return f"{date.year}-{date.year+1}"
    elif date.month > 7:
        return f"{date.year}-{date.year+1}"
    else:
        return f"{date.year-1}-{date.year}"

# Apply fiscal year calculation
df['Fiscal Year'] = df['Date'].apply(get_fiscal_year)

# Create transition matrix function
def calculate_weighted_transition_matrix(df):
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

# Create result containers using lists for efficiency
weights_rows = []
probs_rows = []

# Process data by quarter and product
quarters = sorted(df['Quarter'].unique())
products = sorted(df['Loan Product'].unique())

# Process quarterly product-specific data
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
        for s1 in stages:
            for s2 in stages:
                trans_name = f'Stage {s1} to {s2}'
                weight_row[trans_name] = weights.loc[s1, s2]
                prob_row[trans_name] = matrix.loc[s1, s2]
        
        # Add to results
        weights_rows.append(weight_row)
        probs_rows.append(prob_row)

# Add overall quarter summary
for quarter in quarters:
    q_df = df[df['Quarter'] == quarter]
    if len(q_df) == 0:
        continue
            
    matrix, weights = calculate_weighted_transition_matrix(q_df)
    
    weight_row = {'Quarter': quarter, 'Product Type': 'All Products'}
    prob_row = {'Quarter': quarter, 'Product Type': 'All Products'}
    
    for s1 in stages:
        for s2 in stages:
            trans_name = f'Stage {s1} to {s2}'
            weight_row[trans_name] = weights.loc[s1, s2]
            prob_row[trans_name] = matrix.loc[s1, s2]
    
    weights_rows.append(weight_row)
    probs_rows.append(prob_row)

# Add overall product summary
for product in products:
    product_df = df[df['Loan Product'] == product]
    if len(product_df) == 0:
        continue
            
    matrix, weights = calculate_weighted_transition_matrix(product_df)
    
    weight_row = {'Quarter': 'All Quarters', 'Product Type': product}
    prob_row = {'Quarter': 'All Quarters', 'Product Type': product}
    
    for s1 in stages:
        for s2 in stages:
            trans_name = f'Stage {s1} to {s2}'
            weight_row[trans_name] = weights.loc[s1, s2]
            prob_row[trans_name] = matrix.loc[s1, s2]
    
    weights_rows.append(weight_row)
    probs_rows.append(prob_row)

# Add grand total summary
matrix, weights = calculate_weighted_transition_matrix(df)

weight_row = {'Quarter': 'All Quarters', 'Product Type': 'All Products'}
prob_row = {'Quarter': 'All Quarters', 'Product Type': 'All Products'}

for s1 in stages:
    for s2 in stages:
        trans_name = f'Stage {s1} to {s2}'
        weight_row[trans_name] = weights.loc[s1, s2]
        prob_row[trans_name] = matrix.loc[s1, s2]

weights_rows.append(weight_row)
probs_rows.append(prob_row)

# Convert to DataFrames
final_weights = pd.DataFrame(weights_rows)
final_probs = pd.DataFrame(probs_rows)

# Create annual probabilities by raising quarterly matrices to 4th power
annual_rows = []

for _, row in final_probs.iterrows():
    # Create matrix from row data
    matrix = np.zeros((4, 4))
    for i, s1 in enumerate(stages):
        for j, s2 in enumerate(stages):
            trans_name = f'Stage {s1} to {s2}'
            matrix[i, j] = row[trans_name]
    
    # Normalize rows to ensure valid probability matrix
    row_sums = matrix.sum(axis=1)
    for i in range(4):
        if row_sums[i] > 0:
            matrix[i] /= row_sums[i]
        else:
            # If no transitions from this state, maintain current state
            matrix[i] = np.eye(4)[i]
    
    # Raise to 4th power for annual probabilities
    annual_matrix = np.linalg.matrix_power(matrix, 4)
    
    # Create annual row
    annual_row = {
        'Quarter': row['Quarter'],
        'Product Type': row['Product Type']
    }
    
    for i, s1 in enumerate(stages):
        for j, s2 in enumerate(stages):
            trans_name = f'Stage {s1} to {s2}'
            annual_row[trans_name] = annual_matrix[i, j]
    
    annual_rows.append(annual_row)

annual_probs = pd.DataFrame(annual_rows)

# Create fiscal year averages using consistent matrix exponentiation
# First, create a mapping from quarter to fiscal year
quarter_to_fiscal = {}
for quarter in quarters:
    # Convert quarter string to period
    period = pd.Period(quarter)
    # Get start date of quarter
    start_date = period.start_time.date()
    # Apply fiscal year function
    quarter_to_fiscal[quarter] = get_fiscal_year(start_date)

# Create fiscal year summary using matrix exponentiation
fiscal_avg_rows = []

# Group by Fiscal Year and Product Type
for fiscal_year in df['Fiscal Year'].unique():
    # Filter data for this fiscal year
    fiscal_df = df[df['Fiscal Year'] == fiscal_year]
    
    # Get quarters in this fiscal year
    quarters_in_year = sorted(fiscal_df['Quarter'].unique())
    
    for product in products + ['All Products']:
        # Filter for current product
        if product == 'All Products':
            product_df = fiscal_df.copy()
        else:
            product_df = fiscal_df[fiscal_df['Loan Product'] == product]
            
        if len(product_df) == 0:
            continue
        
        # Calculate annual matrix using full year's data
        annual_matrix, _ = calculate_weighted_transition_matrix(product_df)
        
        # Raise to 4th power for annual probabilities
        matrix_vals = annual_matrix.values.astype(float)
        
        # Normalize rows
        row_sums = matrix_vals.sum(axis=1)
        for i in range(4):
            if row_sums[i] > 0:
                matrix_vals[i] /= row_sums[i]
            else:
                matrix_vals[i] = np.eye(4)[i]
        
        annual_matrix_vals = np.linalg.matrix_power(matrix_vals, 4)
        
        # Extract specific transitions
        stage1_to_3 = annual_matrix_vals[stages.index('1'), stages.index('3')]
        stage2_to_3 = annual_matrix_vals[stages.index('2'), stages.index('3')]
        
        fiscal_avg_rows.append({
            'Year': fiscal_year,
            'Product Type': product,
            'Stage 1 to 3': stage1_to_3,
            'Stage 2 to 3': stage2_to_3,
            'Quarters Included': ", ".join(quarters_in_year)
        })

# Create DataFrame
fiscal_avg = pd.DataFrame(fiscal_avg_rows)

# Export to Excel
with pd.ExcelWriter('transition_matrices.xlsx') as writer:
    # Format numeric columns
    num_format = '#,##0.00'
    pct_format = '0.00%'
    
    # Write weights
    final_weights.to_excel(writer, sheet_name='Transition Amounts', index=False)
    ws = writer.sheets['Transition Amounts']
    for col in range(2, len(transitions) + 2):  # Start from column C
        ws.set_column(col, col, 18, writer.book.add_format({'num_format': num_format}))
    
    # Write quarterly probabilities
    final_probs.to_excel(writer, sheet_name='Quarterly Probabilities', index=False)
    ws = writer.sheets['Quarterly Probabilities']
    for col in range(2, len(transitions) + 2):
        ws.set_column(col, col, 18, writer.book.add_format({'num_format': pct_format}))
    
    # Write annual probabilities
    annual_probs.to_excel(writer, sheet_name='Annual Probabilities', index=False)
    ws = writer.sheets['Annual Probabilities']
    for col in range(2, len(transitions) + 2):
        ws.set_column(col, col, 18, writer.book.add_format({'num_format': pct_format}))
    
    # Write fiscal year averages
    fiscal_avg.to_excel(writer, sheet_name='Fiscal Year Averages', index=False)
    ws = writer.sheets['Fiscal Year Averages']
    # Set column widths
    ws.set_column(0, 0, 12)  # Year
    ws.set_column(1, 1, 25)  # Product Type
    ws.set_column(2, 2, 15, writer.book.add_format({'num_format': pct_format}))  # Stage 1 to 3
    ws.set_column(3, 3, 15, writer.book.add_format({'num_format': pct_format}))  # Stage 2 to 3
    ws.set_column(4, 4, 30)  # Quarters Included

print("Transition matrices successfully exported to 'transition_matrices.xlsx'")
print("Sheets created:")
print("- Transition Amounts: Outstanding principal amounts")
print("- Quarterly Probabilities: Quarterly transition probabilities")
print("- Annual Probabilities: Annual transition probabilities (quarterly matrix^4)")
print("- Fiscal Year Averages: Yearly probabilities using full fiscal year data")
print("   - Calculated using matrix exponentiation on aggregated fiscal year data")
print("   - Consistent with Annual Probabilities sheet methodology")