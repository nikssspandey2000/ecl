"""
Script to process ECL data, calculate transition matrices, and save results to SQLite and Excel.
"""

import pandas as pd
import sqlite3
import logging

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configurable filenames ---
EXCEL_FILE = "00. Raw ECL 2081.12.27.xlsx"
DB_FILE = "ecl_database.db"
OUTPUT_FILE = "01. HistoricalPDCalculated.xlsx"

# --- Mapping for Quarter Number ---
quarter_mapping = {
    "Asadh 76": 1, "Ashoj 76": 2, "poush 76": 3, "chaitra 76": 4,
    "Asadh 77": 5, "Asoj 77": 6, "Poush 77": 7, "Chaitra 77": 8,
    "Asadh 78": 9, "Ashoj 78": 10, "Poush 78": 11, "Chaitra 78": 12,
    "Asadh 79": 13, "Ashoj 79": 14, "Poush 79": 15, "Chaitra 79": 16,
    "Asadh 80": 17, "Ashoj 80": 18, "Poush 80": 19, "Chaitra 80": 20,
    "Asadh 81": 21, "Ashoj 81": 22, "Poush 81": 23, "Chaitra 81": 24,
    "Asadh 82": 25
}

def read_and_combine_excel(excel_file, quarter_mapping):
    """Reads all sheets, cleans columns, and combines into one DataFrame."""
    try:
        sheets = pd.read_excel(excel_file, sheet_name=None)
    except Exception as e:
        logging.error(f"Error reading Excel file: {e}")
        raise
    all_data = []
    for sheet_name, df in sheets.items():
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df["Quarter"] = sheet_name
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    combined_df["Quarter_Number"] = combined_df["Quarter"].map(quarter_mapping)
    return combined_df

def assign_stage(df):
    """Assigns numeric stage based on Default Flag."""
    df["Stage"] = df["Default Flag"].apply(
        lambda x: 1 if str(x).strip().lower() == "pass loan" else
                  2 if str(x).strip().lower() == "watchlist loan" else 3
    )
    return df

def add_previous_stages(df):
    """Adds previous and previous-previous quarter stages."""
    lookup_df = df[["MainCode", "Quarter_Number", "Stage"]].copy()
    df = df.merge(
        lookup_df.rename(columns={"Stage": "Previous Quarter", "Quarter_Number": "Quarter_Number_prev"}),
        how="left",
        left_on=["MainCode", df["Quarter_Number"] - 1],
        right_on=["MainCode", "Quarter_Number_prev"]
    ).drop(columns=["Quarter_Number_prev"])
    df = df.merge(
        lookup_df.rename(columns={"Stage": "Previous Previous Quarter", "Quarter_Number": "Quarter_Number_prev2"}),
        how="left",
        left_on=["MainCode", df["Quarter_Number"] - 2],
        right_on=["MainCode", "Quarter_Number_prev2"]
    ).drop(columns=["Quarter_Number_prev2"])
    df["Previous Quarter"] = df["Previous Quarter"].fillna("New")
    df["Previous Previous Quarter"] = df["Previous Previous Quarter"].fillna("New")
    return df

def compute_final_stage(row):
    """Computes the max stage among current, previous, and previous-previous."""
    values = [row["Stage"]]
    for col in ["Previous Quarter", "Previous Previous Quarter"]:
        val = row[col]
        if pd.api.types.is_number(val) or str(val).isdigit():
            values.append(int(val))
    return max(values)

def process_transition_matrix(df):
    """Processes and returns transition matrix and summary."""
    selected_columns = [
        'Quarter_Number', 'Final Stage', 'Previous Quarter', 'Loan Class',
        'MainCode', 'O/S Interest', 'O/S Principal'
    ]
    df_selected = df[selected_columns].copy()
    df_selected['Previous Quarter'] = df_selected['Previous Quarter'].fillna('New')
    grouped = df_selected.groupby(
        ['Quarter_Number', 'Loan Class', 'Previous Quarter', 'Final Stage']
    ).agg({
        'O/S Interest': 'sum',
        'O/S Principal': 'sum'
    }).reset_index()
    grouped.rename(columns={
        'Previous Quarter': 'From Stage',
        'Final Stage': 'To Stage',
        'O/S Interest': 'O/S interest',
        'O/S Principal': 'O/S principle'
    }, inplace=True)
    total_by_from_stage = df_selected.groupby(
        ['Quarter_Number', 'Loan Class', 'Previous Quarter']
    ).agg({
        'O/S Interest': 'sum',
        'O/S Principal': 'sum'
    }).reset_index()
    total_by_from_stage.rename(columns={
        'O/S Interest': 'Total O/S Interest',
        'O/S Principal': 'Total O/S Principal',
        'Previous Quarter': 'From Stage'
    }, inplace=True)
    grouped = grouped.merge(
        total_by_from_stage, on=['Quarter_Number', 'Loan Class', 'From Stage'], how='left'
    )
    grouped['% O/S Interest Transfer'] = (
        grouped['O/S interest'] / grouped['Total O/S Interest'] * 100
    ).round(2)
    grouped['% O/S Principal Transfer'] = (
        grouped['O/S principle'] / grouped['Total O/S Principal'] * 100
    ).round(2)
    return df_selected, grouped

import numpy as np

def calculate_annual_transition_matrix(summary_df, output_file):
    """
    For each Quarter_Number and Loan Class, calculate the Markov annual transition matrix
    by multiplying the quarterly transition matrix 4 times, and save to Excel.
    """
    results = []
    stages = [1, 2, 3]
    for (quarter, loan_class), group in summary_df.groupby(['Quarter_Number', 'Loan Class']):
        # Build 3x3 matrix for this group
        matrix = np.zeros((3, 3))
        for i, from_stage in enumerate(stages):
            for j, to_stage in enumerate(stages):
                val = group.loc[
                    (group['From Stage'] == from_stage) & (group['To Stage'] == to_stage),
                    '% O/S Principal Transfer'
                ]
                matrix[i, j] = val.values[0] if not val.empty else 0.0
        # Normalize rows to sum to 100 (if not already)
        matrix = np.where(matrix.sum(axis=1, keepdims=True) == 0, matrix, 
                          matrix / matrix.sum(axis=1, keepdims=True) * 100)
        # Convert to Markov probabilities (0-1)
        matrix_markov = matrix / 100.0
        # Multiply matrix 4 times
        annual_matrix = np.linalg.matrix_power(matrix_markov, 4)
        # Convert back to percentage
        annual_matrix = np.round(annual_matrix * 100, 2)
        # Store results in long format
        for i, from_stage in enumerate(stages):
            for j, to_stage in enumerate(stages):
                results.append({
                    'Quarter_Number': quarter,
                    'Loan Class': loan_class,
                    'From Stage': from_stage,
                    'To Stage': to_stage,
                    'Annual % O/S Principal Transfer': annual_matrix[i, j]
                })
    # Create DataFrame and save to Excel
    annual_df = pd.DataFrame(results)
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
        annual_df.to_excel(writer, sheet_name='AnnualTransitionMatrix', index=False)
    logging.info("Annual Markov transition matrix saved to Excel.")


def main():
    try:
        combined_df = read_and_combine_excel(EXCEL_FILE, quarter_mapping)
        combined_df = assign_stage(combined_df)
        combined_df = add_previous_stages(combined_df)
        combined_df["Final Stage"] = combined_df.apply(compute_final_stage, axis=1)
        conn = sqlite3.connect(DB_FILE)
        combined_df.to_sql("ecl_data", conn, if_exists="replace", index=False)
        logging.info("Data saved with Previous/PrevPrev/Final Stage to database.")
        df_selected, grouped = process_transition_matrix(combined_df)
        df_selected.to_sql('transition_matrix', conn, if_exists='replace', index=False)
        grouped.to_sql('SummaryTransitionMatrix', conn, if_exists='replace', index=False)
        
        with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
            combined_df.to_excel(writer, sheet_name='2.2', index=False)
            df_selected.to_excel(writer, sheet_name='TransitionMatrix', index=False)
            grouped.to_excel(writer, sheet_name='SummaryTransitionMatrix', index=False)

        # Now append the annual matrix
        calculate_annual_transition_matrix(grouped, OUTPUT_FILE)       
        conn.close()
        logging.info("Data processed and saved to database and Excel file.")
    except Exception as e:
        logging.error(f"Processing failed: {e}")

if __name__ == "__main__":
    main()







# --- Add this to your main() after SummaryTransitionMatrix is created ---
# calculate_annual_transition_matrix(grouped, OUTPUT_FILE)


# --- Suggestions for Improvement ---
# 1. Add error handling for file/database operations.
# 2. Modularize code into functions for readability and reuse.
# 3. Use logging instead of print for better control.
# 4. Consider vectorizing 'compute_final_stage' for performance.
# 5. Use config/variables for filenames to avoid hardcoding.
# 6. Add a docstring at the top explaining the script's purpose.