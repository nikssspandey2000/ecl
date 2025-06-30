import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

# --------------------------
# Configuration
# --------------------------
input_file = "historical_data.xlsx"       # Your input Excel file
output_file = "calibratedpd.xlsx"         # Output Excel workbook

# --------------------------
# Load Data
# --------------------------
df = pd.read_excel(input_file)
df["PD (%)"] = df["PD (%)"] / 100  # Convert PD to decimal

# Split target and features
y = df["PD (%)"]
X = df.drop(columns=["PD (%)"])

# --------------------------
# Regression
# --------------------------
model = LinearRegression()
model.fit(X, y)
coefficients = dict(zip(X.columns, model.coef_))

# --------------------------
# Normalize Coefficients
# --------------------------
abs_coeffs = {k: abs(v) for k, v in coefficients.items()}
total = sum(abs_coeffs.values())
weights = {k: v / total for k, v in abs_coeffs.items()}

# --------------------------
# Monte Carlo Simulation
# --------------------------
n_simulations = 10000
simulated_data = {}

for col in X.columns:
    mean = df[col].mean()
    std = df[col].std()
    simulated_data[col] = np.random.normal(loc=mean, scale=std, size=n_simulations)

# Standardize and compute systematic factor X
X_sim = np.zeros(n_simulations)
for col in X.columns:
    mean = df[col].mean()
    std = df[col].std()
    z_scores = (simulated_data[col] - mean) / std
    X_sim += weights[col] * z_scores

# --------------------------
# PIT PD via Vasicek Model
# --------------------------
pd_ttc = y.mean()
rho = 0.15
inv_pd = norm.ppf(pd_ttc)
pit_pds = norm.cdf((inv_pd - np.sqrt(rho) * X_sim) / np.sqrt(1 - rho))

# --------------------------
# Create Plot Buffers
# --------------------------
plot_images = {}

# 1. Regression Coefficients
plt.figure(figsize=(8, 4))
sns.barplot(x=list(coefficients.keys()), y=list(coefficients.values()))
plt.title("Regression Coefficients")
plt.xticks(rotation=45)
plt.tight_layout()
buf1 = io.BytesIO()
plt.savefig(buf1, format='png')
buf1.seek(0)
plot_images['Regression Coefficients'] = buf1
plt.close()

# 2. Normalized Weights
plt.figure(figsize=(8, 4))
sns.barplot(x=list(weights.keys()), y=list(weights.values()))
plt.title("Normalized Weights")
plt.xticks(rotation=45)
plt.tight_layout()
buf2 = io.BytesIO()
plt.savefig(buf2, format='png')
buf2.seek(0)
plot_images['Normalized Weights'] = buf2
plt.close()

# 3. PIT PD Distribution
plt.figure(figsize=(8, 5))
plt.hist(pit_pds, bins=100, color='skyblue', edgecolor='black', density=True)
plt.title("Distribution of PIT PDs from Monte Carlo Simulation")
plt.xlabel("PIT PD")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
buf3 = io.BytesIO()
plt.savefig(buf3, format='png')
buf3.seek(0)
plot_images['PIT PD Distribution'] = buf3
plt.close()

# --------------------------
# Save to Excel (with plots)
# --------------------------
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name="Original Data", index=False)
    pd.DataFrame.from_dict(coefficients, orient="index", columns=["Coefficient"]).to_excel(writer, sheet_name="Regression Coeff", index_label="Variable")
    pd.DataFrame.from_dict(weights, orient="index", columns=["Normalized Weight"]).to_excel(writer, sheet_name="Normalized Weights", index_label="Variable")
    pd.DataFrame(simulated_data).to_excel(writer, sheet_name="Monte Carlo Simulations", index=False)
    pd.DataFrame({"Systematic X": X_sim}).to_excel(writer, sheet_name="Systematic Factor X", index=False)
    pd.DataFrame({"PIT PD": pit_pds}).to_excel(writer, sheet_name="PIT PDs", index=False)

    # Insert plots into a sheet
    workbook = writer.book
    worksheet = workbook.add_worksheet("Visualization")
    writer.sheets["Visualization"] = worksheet

    row = 0
    for title, img_buf in plot_images.items():
        worksheet.write(row, 0, title)
        worksheet.insert_image(row + 1, 0, title + ".png", {'image_data': img_buf})
        row += 22

# --------------------------
# Summary stats
# --------------------------
print(f"Average PIT PD: {np.mean(pit_pds):.4%}")
print(f"95th Percentile PIT PD: {np.percentile(pit_pds, 95):.4%}")
