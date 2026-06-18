import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('Form Responses 1 - Form Responses 1.csv')

# Filter for Shadow AI Users
df_shadow = df[df['Shadow AI User'] == 'Shadow AI'].copy()

# Define variables
usage_col = 'Q6. How frequently do you use public generative AI tools (e.g., ChatGPT, Gemini, Claude) to complete official work tasks? (5-point scale: 1 = Never, 2 = Rarely, 3 = Sometimes, 4 = Often, 5 = Daily)'
ivs = ['workload pressure 2', 'Perceived Productivity Gain', 'Lack of Approved AI Tools', 'Subjective Norms', 'Risk Awareness']

y_shadow = df_shadow[usage_col]
X_shadow = df_shadow[ivs]
X_shadow_sm = sm.add_constant(X_shadow)

# Fit OLS
ols_model = sm.OLS(y_shadow, X_shadow_sm).fit()

# Plot 1: Coefficient Plot (Forest Plot)
err_series = ols_model.params - ols_model.conf_int()[0]
coef_df = pd.DataFrame({'coef': ols_model.params.values[1:],
                        'err': err_series.values[1:],
                        'varname': err_series.index.values[1:]})

plt.figure(figsize=(10, 6))
plt.errorbar(x=coef_df['coef'], y=coef_df['varname'], xerr=coef_df['err'], fmt='o', color='#1f77b4', capsize=5, markersize=8, elinewidth=2)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.title('OLS Regression Coefficients (95% Confidence Intervals)\nPredicting AI Usage Frequency', fontsize=14)
plt.xlabel('Coefficient Value (Effect Size)', fontsize=12)
plt.ylabel('Independent Variables', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ols_coefficients.png', dpi=300)
plt.close()

# Plot 2: Scatter plot of the most significant relationship
plt.figure(figsize=(8, 6))
sns.regplot(x=df_shadow['Perceived Productivity Gain'], y=y_shadow, 
            scatter_kws={'alpha':0.6, 's':50}, 
            line_kws={'color':'red', 'linewidth':2})
plt.title('Usage Frequency vs. Perceived Productivity Gain\n(The only significant driver in the OLS model)', fontsize=14)
plt.xlabel('Perceived Productivity Gain (Scale)', fontsize=12)
plt.ylabel('AI Usage Frequency (1-5 Scale)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ols_scatter_productivity.png', dpi=300)
plt.close()

print("Plots generated successfully.")