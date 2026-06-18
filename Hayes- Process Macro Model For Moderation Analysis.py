import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('Form Responses 1 - Form Responses 1.csv')

# Identify columns
q6_col = [c for c in df.columns if 'Q6' in c][0]
aup_col = [c for c in df.columns if 'AUP Effectiveness' in c][0]
# Using Q13 as the proxy for Data Security Risk (Data Sensitivity) based on previous context
dv_col = [c for c in df.columns if 'Q13' in c][0] 

print(f"Using IV (X): {q6_col}")
print(f"Using Mod (W): {aup_col}")
print(f"Using DV (Y): {dv_col}")

# Filter for Shadow AI users
shadow_cols = [c for c in df.columns if 'Shadow AI' in c]
shadow_mask = (df[shadow_cols[0]] == 'Shadow AI')
if len(shadow_cols) > 1:
    shadow_mask = shadow_mask | (df[shadow_cols[1]] == 'Shadow AI')
df_filtered = df[shadow_mask].copy()

# Prepare variables
data = df_filtered[[q6_col, aup_col, dv_col]].copy()
data.columns = ['X', 'W', 'Y']
data = data.apply(pd.to_numeric, errors='coerce').dropna()

# Process Macro usually allows mean-centering W and X for the interaction term construction
X = data['X']
W = data['W']
Y = data['Y']

# Mean centering
X_c = X - X.mean()
W_c = W - W.mean()
data['X_c'] = X_c
data['W_c'] = W_c
data['Int'] = X_c * W_c

# Fit model
exog = sm.add_constant(data[['X_c', 'W_c', 'Int']])
model = sm.OLS(Y, exog).fit()

# Format output to look like Hayes PROCESS Macro Model 1
print("\n**************************************************************************")
print("Model : 1")
print("    Y : Data Security Risk (Q13)")
print("    X : Shadow AI Usage (Q6)")
print("    W : AUP Effectiveness")
print("Sample Size:", len(data))
print("**************************************************************************")

print("\nOUTCOME VARIABLE:")
print("Data Security Risk (Q13)")

print(f"\nModel Summary")
print(f"          R       R-sq        MSE          F        df1        df2          p")
print(f"{np.sqrt(model.rsquared):11.4f}{model.rsquared:11.4f}{model.mse_resid:11.4f}{model.fvalue:11.4f}{model.df_model:11.0f}{model.df_resid:11.0f}{model.f_pvalue:11.4f}")

print("\nModel")
print(f"              coeff         se          t          p       LLCI       ULCI")

def print_row(name, param, se, t, p, llci, ulci):
    print(f"{name:<10}{param:11.4f}{se:11.4f}{t:11.4f}{p:11.4f}{llci:11.4f}{ulci:11.4f}")

print_row("constant", model.params['const'], model.bse['const'], model.tvalues['const'], model.pvalues['const'], model.conf_int()[0]['const'], model.conf_int()[1]['const'])
print_row("X", model.params['X_c'], model.bse['X_c'], model.tvalues['X_c'], model.pvalues['X_c'], model.conf_int()[0]['X_c'], model.conf_int()[1]['X_c'])
print_row("W", model.params['W_c'], model.bse['W_c'], model.tvalues['W_c'], model.pvalues['W_c'], model.conf_int()[0]['W_c'], model.conf_int()[1]['W_c'])
print_row("Int_1", model.params['Int'], model.bse['Int'], model.tvalues['Int'], model.pvalues['Int'], model.conf_int()[0]['Int'], model.conf_int()[1]['Int'])

print("\nProduct terms key:")
print("Int_1 :    X x W")

print("\n******************** CONDITIONAL EFFECTS OF X ON Y AT VALUES OF W ********************")
# W values: -1 SD, Mean, +1 SD
w_sd = W.std()
w_mean_raw = W.mean()

print(f"              W     Effect         se          t          p       LLCI       ULCI")
for w_c_val in [-w_sd, 0, w_sd]:
    # conditional effect b1 + b3*W
    effect = model.params['X_c'] + model.params['Int'] * w_c_val
    # SE of conditional effect
    var_b1 = model.cov_params().loc['X_c', 'X_c']
    var_b3 = model.cov_params().loc['Int', 'Int']
    cov_b1_b3 = model.cov_params().loc['X_c', 'Int']
    se = np.sqrt(var_b1 + (w_c_val**2)*var_b3 + 2*w_c_val*cov_b1_b3)
    
    t = effect / se
    p = 2 * (1 - stats.t.cdf(abs(t), model.df_resid))
    llci = effect - stats.t.ppf(0.975, model.df_resid) * se
    ulci = effect + stats.t.ppf(0.975, model.df_resid) * se
    
    w_raw_val = w_c_val + w_mean_raw
    print(f"{w_raw_val:15.4f}{effect:11.4f}{se:11.4f}{t:11.4f}{p:11.4f}{llci:11.4f}{ulci:11.4f}")

# Data for moderation graph if needed
with open("process_output.txt", "w") as f:
    f.write(model.summary().as_text())