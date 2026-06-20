import json

notebook_path = 'FinalProject.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Clear existing injected code cells to start fresh, just in case
new_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'Improving Export Efficiency (Unit Value Analysis)' in source or 'High Market Concentration' in source or 'Data Preprocessing Complete' in source:
            continue
    new_cells.append(cell)
nb['cells'] = new_cells
cells = nb['cells']

growth_idx = -1
strengths_idx = -1

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if '1. Growth & Efficiency' in source:
            growth_idx = i
        elif '2. Strengths & Vulnerabilities' in source:
            strengths_idx = i

def create_code_cell(source_code):
    lines = source_code.split('\n')
    source = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "generated_final_" + str(abs(hash(source_code)) % 100000),
        "metadata": {},
        "outputs": [],
        "source": source
    }

code_prep = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# 1. Load Data
df = pd.read_csv('combined.csv')

# 2. Strict Filtering
# Guard against zero/null/negative Quantity and FOB
initial_len = len(df)
df = df[(df['Quantity'] > 0) & (df['FOB'] > 0)].copy()
df.dropna(subset=['Quantity', 'FOB', 'Year', 'Commodity', 'Country Of Destination'], inplace=True)
print(f"Removed {initial_len - len(df)} rows due to zero, negative, or missing values.")

# 3. Create Targets
df['Unit Value'] = df['FOB'] / df['Quantity']
df['Log Unit Value'] = np.log1p(df['Unit Value'])

print("Data Preprocessing Complete. First 5 rows:")
display(df.head())"""

code_uv_eda = """# --- EDA Layer: Plot the distribution and average growth of log unit values ---
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['Log Unit Value'], bins=50, kde=True)
plt.title('Distribution of Log Unit Value')
plt.xlabel('Log(Unit Value + 1)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
avg_log_uv = df.groupby('Year')['Log Unit Value'].mean().reset_index()
sns.lineplot(data=avg_log_uv, x='Year', y='Log Unit Value', marker='o')
plt.title('Average Growth of Log Unit Values (5-Year Span)')
plt.xlabel('Year')
plt.ylabel('Average Log Unit Value')
plt.xticks(avg_log_uv['Year'].astype(int))
plt.grid(True)

plt.tight_layout()
plt.show()"""

code_uv_model = """# --- Unit Value Modeling (Linear Regression vs XGBoost) ---

# Prepare Features and Target
X = df[['Year', 'Commodity', 'Country Of Destination']].copy()
y = df['Log Unit Value']

# Train/Test Split (80/20)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Feature Engineering for XGBoost (Frequency Encoding) ---
freq_commodity = X_train_raw['Commodity'].value_counts(normalize=True)
freq_country = X_train_raw['Country Of Destination'].value_counts(normalize=True)

X_train_xgb = X_train_raw.copy()
X_test_xgb = X_test_raw.copy()

X_train_xgb['Commodity_Freq'] = X_train_xgb['Commodity'].map(freq_commodity).fillna(0)
X_train_xgb['Country_Freq'] = X_train_xgb['Country Of Destination'].map(freq_country).fillna(0)
X_test_xgb['Commodity_Freq'] = X_test_xgb['Commodity'].map(freq_commodity).fillna(0)
X_test_xgb['Country_Freq'] = X_test_xgb['Country Of Destination'].map(freq_country).fillna(0)

features_xgb = ['Year', 'Commodity_Freq', 'Country_Freq']

# Train XGBoost
xgb_model = XGBRegressor(random_state=42, n_estimators=100, max_depth=6)
xgb_model.fit(X_train_xgb[features_xgb], y_train)

y_pred_train_xgb = xgb_model.predict(X_train_xgb[features_xgb])
y_pred_test_xgb = xgb_model.predict(X_test_xgb[features_xgb])

# --- Feature Engineering for Linear Regression (One-Hot Encoding top 10) ---
top_10_commodities = X_train_raw['Commodity'].value_counts().nlargest(10).index
top_10_countries = X_train_raw['Country Of Destination'].value_counts().nlargest(10).index

X_train_lr = X_train_raw.copy()
X_test_lr = X_test_raw.copy()

X_train_lr['Commodity_Group'] = X_train_lr['Commodity'].apply(lambda x: x if x in top_10_commodities else 'Other')
X_train_lr['Country_Group'] = X_train_lr['Country Of Destination'].apply(lambda x: x if x in top_10_countries else 'Other')
X_test_lr['Commodity_Group'] = X_test_lr['Commodity'].apply(lambda x: x if x in top_10_commodities else 'Other')
X_test_lr['Country_Group'] = X_test_lr['Country Of Destination'].apply(lambda x: x if x in top_10_countries else 'Other')

# One-hot encode dropping the first category to avoid dummy variable trap
X_train_lr_encoded = pd.get_dummies(X_train_lr[['Year', 'Commodity_Group', 'Country_Group']], drop_first=True)
# Ensure test set has same columns
X_test_lr_encoded = pd.get_dummies(X_test_lr[['Year', 'Commodity_Group', 'Country_Group']], drop_first=True)
X_test_lr_encoded = X_test_lr_encoded.reindex(columns=X_train_lr_encoded.columns, fill_value=0)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_lr_encoded, y_train)

y_pred_train_lr = lr_model.predict(X_train_lr_encoded)
y_pred_test_lr = lr_model.predict(X_test_lr_encoded)

# --- Reporting Results ---
print("XGBoost Results:")
print(f"  Train R2: {r2_score(y_train, y_pred_train_xgb):.4f} | Train MSE: {mean_squared_error(y_train, y_pred_train_xgb):.4f}")
print(f"  Test R2:  {r2_score(y_test, y_pred_test_xgb):.4f} | Test MSE:  {mean_squared_error(y_test, y_pred_test_xgb):.4f}")

print("\\nLinear Regression Results:")
print(f"  Train R2: {r2_score(y_train, y_pred_train_lr):.4f} | Train MSE: {mean_squared_error(y_train, y_pred_train_lr):.4f}")
print(f"  Test R2:  {r2_score(y_test, y_pred_test_lr):.4f} | Test MSE:  {mean_squared_error(y_test, y_pred_test_lr):.4f}")

# Plot XGBoost Feature Importances
plt.figure(figsize=(6, 4))
importances = xgb_model.feature_importances_
sns.barplot(x=importances, y=features_xgb)
plt.title('XGBoost Feature Importances for Log Unit Value')
plt.show()"""

code_hhi = """# --- High Market Concentration: HHI Index ---

yearly_totals = df.groupby('Year')['FOB'].sum().reset_index().rename(columns={'FOB': 'Total Yearly FOB'})
country_yearly = df.groupby(['Year', 'Country Of Destination'])['FOB'].sum().reset_index()

country_yearly = pd.merge(country_yearly, yearly_totals, on='Year')
country_yearly['Market Share (%)'] = (country_yearly['FOB'] / country_yearly['Total Yearly FOB']) * 100
country_yearly['Share Squared'] = country_yearly['Market Share (%)'] ** 2

hhi_per_year = country_yearly.groupby('Year')['Share Squared'].sum().reset_index().rename(columns={'Share Squared': 'HHI'})

plt.figure(figsize=(10, 6))
sns.lineplot(data=hhi_per_year, x='Year', y='HHI', marker='s', color='red')
plt.title('Herfindahl-Hirschman Index (HHI) 5-Year Trendline')
plt.xlabel('Year')
plt.ylabel('HHI')
plt.axhline(y=1500, color='orange', linestyle='--', label='Moderate Concentration (1500)')
plt.axhline(y=2500, color='darkred', linestyle='--', label='High Concentration (2500)')
plt.xticks(hhi_per_year['Year'].astype(int))
plt.legend()
plt.grid(True)
plt.show()"""

code_kmeans = """# --- High Market Concentration: K-Means Clustering ---

# Aggregate across the entire time span to cluster countries by their overall reliance
country_agg = df.groupby('Country Of Destination').agg(
    Total_FOB=('FOB', 'sum')
).reset_index()

total_fob_all = country_agg['Total_FOB'].sum()
country_agg['Overall Market Share (%)'] = (country_agg['Total_FOB'] / total_fob_all) * 100

# Prepare data for clustering
X_cluster = country_agg[['Total_FOB', 'Overall Market Share (%)']].copy()

# Critical: Scale the data so FOB (millions) doesn't dominate Market Share (0-100)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 1. Elbow Plot to justify k
inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, marker='o')
plt.title('Elbow Plot for K-Means Clustering')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)

# 2. Fit final K-Means with k=3 based on assumed/justified elbow
k_optimal = 3
kmeans_final = KMeans(n_clusters=k_optimal, random_state=42, n_init='auto')
country_agg['Cluster'] = kmeans_final.fit_predict(X_scaled)

plt.subplot(1, 2, 2)
sns.scatterplot(
    data=country_agg, 
    x='Overall Market Share (%)', 
    y='Total_FOB', 
    hue='Cluster', 
    palette='viridis', 
    s=100
)
plt.yscale('log') # Log scale for better visibility of heavily right-skewed FOB
plt.title(f'K-Means Clustering of Destination Countries (k={k_optimal})')
plt.xlabel('Overall Market Share (%)')
plt.ylabel('Total FOB (Log Scale)')
plt.legend(title='Cluster')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Top 5 Countries and their Clusters:")
display(country_agg.sort_values(by='Total_FOB', ascending=False).head())"""

cells_to_insert_growth = [create_code_cell(code_prep), create_code_cell(code_uv_eda), create_code_cell(code_uv_model)]
cells_to_insert_strengths = [create_code_cell(code_hhi), create_code_cell(code_kmeans)]

if strengths_idx != -1:
    for c in reversed(cells_to_insert_strengths):
        cells.insert(strengths_idx + 1, c)

if growth_idx != -1:
    for c in reversed(cells_to_insert_growth):
        cells.insert(growth_idx + 1, c)

nb['cells'] = cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
