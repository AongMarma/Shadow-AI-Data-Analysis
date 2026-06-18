import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('Form Responses 1 - Form Responses 1.csv')

# Filter for Shadow AI Users
df_shadow = df[df['Shadow AI User'] == 'Shadow AI'].copy()
print(f"Number of Shadow AI User rows: {len(df_shadow)}")

# The dependent variable needs to be a measure of 'usage' since the category itself is now constant.
# We will use Q6 (Frequency of use) as the measure of Shadow AI Usage.
usage_col = 'Q6. How frequently do you use public generative AI tools (e.g., ChatGPT, Gemini, Claude) to complete official work tasks? (5-point scale: 1 = Never, 2 = Rarely, 3 = Sometimes, 4 = Often, 5 = Daily)'

print("\nDistribution of Usage Frequency (Q6):")
print(df_shadow[usage_col].value_counts().sort_index())

# Define independent variables (excluding AUP Effectiveness as requested)
ivs = ['workload pressure 2', 'Perceived Productivity Gain', 'Lack of Approved AI Tools', 'Subjective Norms', 'Risk Awareness']
X = df_shadow[ivs]
X = sm.add_constant(X)

# 1. Linear Regression (Continuous 1-5 scale)
y_linear = df_shadow[usage_col]
try:
    ols_model = sm.OLS(y_linear, X).fit()
    print("\n--- Linear Regression (Dependent: Frequency of Use 1-5) ---")
    print(ols_model.summary())
except Exception as e:
    print("Linear regression error:", e)

# 2. Logistic Regression (Binary: High Usage vs Low/Medium Usage)
# Let's define High Usage as 4 (Often) and 5 (Daily), and Low/Medium as 1-3.
df_shadow['High_Usage'] = df_shadow[usage_col].apply(lambda x: 1 if x >= 4 else 0)
y_logistic = df_shadow['High_Usage']

try:
    logit_model = sm.Logit(y_logistic, X).fit()
    print("\n--- Logistic Regression (Dependent: High Usage >= 4) ---")
    print(logit_model.summary())
except Exception as e:
    print("Logistic regression error:", e)