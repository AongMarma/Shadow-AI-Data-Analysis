import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

def run_comprehensive_analysis(file_path):
    # 1. LOAD DATASET
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # 2. IDENTIFY AND MAP VARIABLES
    # Dynamically locate specific columns based on substrings
    q6_col = [c for c in df.columns if 'Q6' in c][0]
    dv_col = [c for c in df.columns if 'Q13' in c][0]
    aup_col = [c for c in df.columns if 'AUP Effectiveness' in c][0]
    lack_tools_col = 'Lack of Approved AI Tools'
    
    # All continuous Independent Variables to evaluate
    iv_cols = [q6_col, 'workload pressure 2', 'Perceived Productivity Gain', 
               lack_tools_col, 'Subjective Norms', 'Risk Awareness']
    
    # 3. FILTER FOR SHADOW AI USERS ONLY
    shadow_cols = [c for c in df.columns if 'Shadow AI' in c]
    shadow_mask = (df[shadow_cols[0]] == 'Shadow AI')
    if len(shadow_cols) > 1:
        shadow_mask = shadow_mask | (df[shadow_cols[1]] == 'Shadow AI')
    df_filtered = df[shadow_mask].copy()
    
    print(f"Total Shadow AI observations identified: {len(df_filtered)}\n")
    
    # =========================================================================
    # PART 1: HAYES PROCESS MACRO SIMULATION (MODEL 1) FOR Q6 * AUP -> Q13
    # =========================================================================
    print("Executing Hayes PROCESS Model 1 Simulation (Q6 * AUP Effectiveness)...")
    process_data = df_filtered[[q6_col, aup_col, dv_col]].copy()
    process_data.columns = ['X', 'W', 'Y']
    process_data = process_data.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Mean-centering continuous variables to match PROCESS behavior
    x_mean, w_mean = process_data['X'].mean(), process_data['W'].mean()
    w_std = process_data['W'].std()
    
    process_data['X_c'] = process_data['X'] - x_mean
    process_data['W_c'] = process_data['W'] - w_mean
    process_data['Int_1'] = process_data['X_c'] * process_data['W_c']
    
    X_matrix = sm.add_constant(process_data[['X_c', 'W_c', 'Int_1']])
    process_model = sm.OLS(process_data['Y'], X_matrix).fit()
    
    # Hierarchical R2 change for the interaction block
    X_main = sm.add_constant(process_data[['X_c', 'W_c']])
    main_model = sm.OLS(process_data['Y'], X_main).fit()
    r2_chng = process_model.rsquared - main_model.rsquared
    f_stat_chng = (r2_chng / 1) / ((1 - process_model.rsquared) / process_model.df_resid)
    p_val_chng = stats.f.sf(f_stat_chng, 1, process_model.df_resid)
    
    print(process_model.summary())
    print(f"\nInteraction R2-Change: {r2_chng:.4f} | F-Chng: {f_stat_chng:.4f} | p-value: {p_val_chng:.4f}\n")
    
    # =========================================================================
    # PART 2: CURVILINEAR MODERATION ANALYSIS (X^2 * W -> Y)
    # =========================================================================
    print("Executing Curvilinear Moderation Analysis (Q6^2 * AUP Effectiveness)...")
    process_data['X_sq'] = process_data['X_c'] ** 2
    process_data['Quad_Int'] = process_data['X_sq'] * process_data['W_c']
    
    X_curve_A = sm.add_constant(process_data[['X_c', 'X_sq', 'W_c', 'Int_1']])
    res_A = sm.OLS(process_data['Y'], X_curve_A).fit()
    
    X_curve_B = sm.add_constant(process_data[['X_c', 'X_sq', 'W_c', 'Int_1', 'Quad_Int']])
    res_B = sm.OLS(process_data['Y'], X_curve_B).fit()
    
    r2_curve_chng = res_B.rsquared - res_A.rsquared
    f_curve_chng = ((r2_curve_chng) / 1) / ((1 - res_B.rsquared) / res_B.df_resid)
    p_curve_chng = stats.f.sf(f_curve_chng, 1, res_B.df_resid)
    
    print(f"Curvilinear R2-Change (X^2*W): {r2_curve_chng:.4f} | p-value: {p_curve_chng:.4f}\n")
    
    # =========================================================================
    # PART 3: COMPARATIVE MODERATION TRACKING & SIMPLE SLOPES
    # =========================================================================
    print("Calculating Comparative Interaction Effects & Simple Slopes across all IVs...")
    moderation_summary = []
    
    for iv in iv_cols:
        analysis_df = df_filtered[[iv, dv_col, aup_col]].copy()
        analysis_df.columns = ['X', 'Y', 'W']
        analysis_df = analysis_df.apply(pd.to_numeric, errors='coerce').dropna()
        
        iv_m, w_m, w_s = analysis_df['X'].mean(), analysis_df['W'].mean(), analysis_df['W'].std()
        analysis_df['X_c'] = analysis_df['X'] - iv_m
        analysis_df['W_c'] = analysis_df['W'] - w_m
        analysis_df['Int'] = analysis_df['X_c'] * analysis_df['W_c']
        
        X_reg = sm.add_constant(analysis_df[['X_c', 'W_c', 'Int']])
        m_fit = sm.OLS(analysis_df['Y'], X_reg).fit()
        
        # Calculate Simple Slope specifically at High W (+1 SD) and Low W (-1 SD)
        slopes_report = {}
        for w_val, w_lab in zip([-w_s, w_s], ['Low_AUP', 'High_AUP']):
            effect = m_fit.params['X_c'] + m_fit.params['Int'] * w_val
            var_b1 = m_fit.cov_params().loc['X_c', 'X_c']
            var_b3 = m_fit.cov_params().loc['Int', 'Int']
            cov_b1_b3 = m_fit.cov_params().loc['X_c', 'Int']
            se = np.sqrt(var_b1 + (w_val**2)*var_b3 + 2*w_val*cov_b1_b3)
            p_val = 2 * (1 - stats.t.cdf(abs(effect / se), m_fit.df_resid))
            slopes_report[w_lab] = f"{effect:.3f} (p={p_val:.3f})"
            
        moderation_summary.append({
            'IV': iv.split('.')[0] if 'Q6' in iv else iv,
            'Interaction_Coef': f"{m_fit.params['Int']:.3f}",
            'Interaction_p': f"{m_fit.pvalues['Int']:.3f}",
            'Low_AUP_Slope(-1SD)': slopes_report['Low_AUP'],
            'High_AUP_Slope(+1SD)': slopes_report['High_AUP']
        })
        
        # Plotting the significant driver: Lack of Approved AI Tools
        if iv == lack_tools_col:
            plt.figure(figsize=(8, 5.5))
            x_range = np.array([analysis_df['X_c'].min(), analysis_df['X_c'].max()])
            for w_val, w_label, col, style in zip([-w_s, 0, w_s], ['Low AUP (-1 SD)', 'Mean AUP', 'High AUP (+1 SD)'], ['green', 'blue', 'red'], [':', '-', '--']):
                y_range = m_fit.params['const'] + m_fit.params['X_c']*x_range + m_fit.params['W_c']*w_val + m_fit.params['Int']*(x_range * w_val)
                plt.plot(x_range + iv_m, y_range, label=w_label, color=col, linestyle=style, linewidth=2)
            
            plt.title('Simple Slopes Analysis: AUP Moderating Lack of Approved AI Tools', fontsize=11, fontweight='bold', pad=12)
            plt.xlabel('Lack of Approved AI Tools')
            plt.ylabel('Data Security Risk / Sensitivity (Q13)')
            plt.legend(frameon=True, shadow=True)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig('simple_slopes_lack_tools.png', dpi=300)
            print("Saved simple slopes visual plot to 'simple_slopes_lack_tools.png'")

    summary_df = pd.DataFrame(moderation_summary)
    print("\n" + "="*80)
    print("                    COMPARATIVE MODERATION MATRIX SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

if __name__ == '__main__':
    # Pass your local file string here
    run_comprehensive_analysis('Form Responses 1 - Form Responses 1.csv')