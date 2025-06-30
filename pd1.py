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
stage_names = {'1': 'Stage 1', '2': 'Stage 2', '3': 'Stage 3', 'New': 'New'}
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
quarterly_matrices = []  # Store for verification

for _, row in final_probs.iterrows():
    # Create matrix from row data
    matrix = np.zeros((4, 4))
    for i, s1 in enumerate(stages):
        for j, s2 in enumerate(stages):
            trans_name = f'Stage {s1} to {s2}'
            matrix[i, j] = row[trans_name]
    
    # Store quarterly matrix for verification
    quarterly_matrices.append({
        'Quarter': row['Quarter'],
        'Product Type': row['Product Type'],
        'Matrix': matrix.copy()
    })
    
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

# Create fiscal year summary using matrix exponentiation
fiscal_avg_rows = []
fiscal_verification = []  # For detailed verification

# First, create a mapping from quarter to fiscal year
quarter_to_fiscal = {}
for quarter in quarters:
    # Convert quarter string to period
    period = pd.Period(quarter)
    # Get start date of quarter
    start_date = period.start_time.date()
    # Apply fiscal year function
    quarter_to_fiscal[quarter] = get_fiscal_year(start_date)

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
        
        # Calculate aggregated matrix for the fiscal year
        annual_matrix, weights_matrix = calculate_weighted_transition_matrix(product_df)
        weights_matrix = weights_matrix.rename(index=stage_names, columns=stage_names)
        
        # Store raw weights for verification
        weights_verif = weights_matrix.stack().reset_index()
        weights_verif.columns = ['From Stage', 'To Stage', 'Outstanding Principal']
        weights_verif['Year'] = fiscal_year
        weights_verif['Product Type'] = product
        fiscal_verification.append(weights_verif)
        
        # Prepare for exponentiation
        matrix_vals = annual_matrix.values.astype(float)
        
        # Normalize rows
        row_sums = matrix_vals.sum(axis=1)
        for i in range(4):
            if row_sums[i] > 0:
                matrix_vals[i] /= row_sums[i]
            else:
                matrix_vals[i] = np.eye(4)[i]
        
        # Raise to 4th power for annual probabilities
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

# Create DataFrames
fiscal_avg = pd.DataFrame(fiscal_avg_rows)
fiscal_verification = pd.concat(fiscal_verification, ignore_index=True)

# Create verification data for quarterly averages
quarterly_verification = final_probs[['Quarter', 'Product Type', 'Stage 1 to 3', 'Stage 2 to 3']].copy()
quarterly_verification['Fiscal Year'] = quarterly_verification['Quarter'].map(quarter_to_fiscal)

# Create annual probabilities verification
annual_verification = annual_probs[['Quarter', 'Product Type', 'Stage 1 to 3', 'Stage 2 to 3']].copy()

# Create matrix verification data
matrix_verification = []
for item in quarterly_matrices:
    quarter = item['Quarter']
    product = item['Product Type']
    matrix = item['Matrix']
    
    # Quarterly matrix
    for i, s1 in enumerate(stages):
        for j, s2 in enumerate(stages):
            matrix_verification.append({
                'Quarter': quarter,
                'Product Type': product,
                'Period': 'Quarterly',
                'From Stage': stage_names[stages[i]],
                'To Stage': stage_names[stages[j]],
                'Probability': matrix[i, j]
            })
    
    # Annual matrix (quarterly^4)
    row_sums = matrix.sum(axis=1)
    for i in range(4):
        if row_sums[i] > 0:
            matrix[i] /= row_sums[i]
        else:
            matrix[i] = np.eye(4)[i]
    
    annual_matrix = np.linalg.matrix_power(matrix, 4)
    for i, s1 in enumerate(stages):
        for j, s2 in enumerate(stages):
            matrix_verification.append({
                'Quarter': quarter,
                'Product Type': product,
                'Period': 'Annual',
                'From Stage': stage_names[stages[i]],
                'To Stage': stage_names[stages[j]],
                'Probability': annual_matrix[i, j]
            })

matrix_verification = pd.DataFrame(matrix_verification)

# Export to Excel
with pd.ExcelWriter('transition_matrices.xlsx') as writer:
    # Format numeric columns
    num_format = '#,##0.00'
    pct_format = '0.00%'
    
    # Main sheets
    final_weights.to_excel(writer, sheet_name='Transition Amounts', index=False)
    final_probs.to_excel(writer, sheet_name='Quarterly Probabilities', index=False)
    annual_probs.to_excel(writer, sheet_name='Annual Probabilities', index=False)
    fiscal_avg.to_excel(writer, sheet_name='Fiscal Year Averages', index=False)
    
    # Verification sheets
    quarterly_verification.to_excel(writer, sheet_name='Quarterly Verification', index=False)
    annual_verification.to_excel(writer, sheet_name='Annual Verification', index=False)
    fiscal_verification.to_excel(writer, sheet_name='Fiscal Year Weights', index=False)
    matrix_verification.to_excel(writer, sheet_name='Matrix Verification', index=False)
    
    # Get all sheets for formatting
    sheets = writer.sheets
    
    # Format sheets
    for sheet_name, ws in sheets.items():
        if sheet_name in ['Transition Amounts']:
            # Format as numbers
            for col in range(2, len(transitions) + 2):
                ws.set_column(col, col, 18, writer.book.add_format({'num_format': num_format}))
        
        elif sheet_name in ['Quarterly Probabilities', 'Annual Probabilities', 
                           'Fiscal Year Averages', 'Quarterly Verification',
                           'Annual Verification']:
            # Format as percentages
            for col in range(2, 10):  # Adjust based on actual columns
                ws.set_column(col, col, 18, writer.book.add_format({'num_format': pct_format}))
        
        elif sheet_name == 'Fiscal Year Weights':
            # Format as numbers
            ws.set_column('C:C', 18, writer.book.add_format({'num_format': num_format}))
        
        elif sheet_name == 'Matrix Verification':
            # Format as percentages
            ws.set_column('E:E', 18, writer.book.add_format({'num_format': pct_format}))
        
        # Freeze headers
        ws.freeze_panes(1, 0)

print("Transition matrices successfully exported to 'transition_matrices.xlsx'")
print("Verification sheets created:")
print("1. Quarterly Verification: Shows Stage 1→3 and Stage 2→3 probabilities for each quarter")
print("2. Annual Verification: Shows Stage 1→3 and Stage 2→3 probabilities from annual projections")
print("3. Fiscal Year Weights: Shows raw outstanding amounts used in fiscal year calculations")
print("4. Matrix Verification: Shows full transition matrices for both quarterly and annual projections")