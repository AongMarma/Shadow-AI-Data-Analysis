import pandas as pd
import numpy as np

# 1. Define the function to calculate Cronbach's Alpha
def cronbach_alpha(df):
    """
    Calculates Cronbach's alpha for a given DataFrame of items.
    """
    # Drop rows with missing values to ensure accurate variance calculation
    df = df.dropna()
    k = df.shape[1]
    
    # If there's 1 or fewer columns, alpha can't be calculated
    if k <= 1:
        return np.nan
        
    # Calculate variance of each item and the variance of the total scores
    item_variances = df.var(ddof=1)
    total_variance = df.sum(axis=1).var(ddof=1)
    
    if total_variance == 0:
        return np.nan
        
    # Apply Cronbach's Alpha formula
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha

# 2. Load your dataset
# Make sure the filename matches exactly with your file
file_name = 'Untitled spreadsheet - Real Form Responses .csv'
df = pd.read_csv(file_name)

# 3. Define your constructs (Q16 has been removed from Workload Pressure)
constructs = {
    'Workload Pressure': [
        'Q14. I frequently work under strong time pressure.', 
        'Q15. My daily workload is often overwhelming.'
    ],
    'Perceived Productivity Gain': [
        'Q17. Public AI tools significantly improve my work efficiency.', 
        'Q18. Public AI tools help me meet my deadlines faster.', 
        'Q19. Using public AI tools increases the overall quality of my output.'
    ],
    'Lack of Approved AI Tools': [
        'Q20. My organization does not provide adequate enterprise-grade AI tools.', 
        'Q21. My organization’s IT restrictions make my work less efficient. ', 
        'Q22. I use external AI tools because the internal software provided by my company is insufficient or outdated.'
    ],
    'Subjective Norms': [
        'Q23. Many of my colleagues use public AI tools for their work tasks.', 
        'Q24. Managers in my department generally tolerate AI tool usage if it improves productivity.', 
        'Q25. There is an unspoken expectation in my workplace to use whatever tools are necessary to get the job done quickly.'
    ],
    'Risk Awareness': [
        'Q26. I clearly understand the general data security risks associated with using public AI tools.', 
        'Q27. I am well aware that information entered into public AI platforms may be retained by third parties and used to train future models.', 
        'Q28. I recognize that sharing sensitive corporate data with unauthorized AI tools could cause significant financial or reputational harm to my organization.'
    ],
    'AUP Effectiveness': [
        'Q30. Policy Clarity', 
        'Q31. Penalty Severity', 
        'Q32. Monitoring Certainty'
    ]
}

# 4. Perform Reliability Check
print("--- RELIABILITY CHECK: CRONBACH'S ALPHA ---")
for name, columns in constructs.items():
    # Only keep columns that actually exist in the dataframe to avoid errors
    existing_columns = [col for col in columns if col in df.columns]
    
    if len(existing_columns) > 1:
        alpha_value = cronbach_alpha(df[existing_columns])
        print(f"{name} ({len(existing_columns)} items): {alpha_value:.4f}")
    else:
        print(f"{name}: Cannot calculate (requires at least 2 valid columns)")


# 5. Perform Validity Check (Discriminant Validity via Pearson Correlation)
print("\n--- VALIDITY CHECK: PEARSON CORRELATION MATRIX ---")
# Create an empty DataFrame to hold the composite scores
composite_scores = pd.DataFrame()

# Calculate the mean score for each construct per respondent
for name, columns in constructs.items():
    existing_columns = [col for col in columns if col in df.columns]
    if len(existing_columns) > 1:
        composite_scores[name] = df[existing_columns].mean(axis=1)

# Generate and print the correlation matrix
correlation_matrix = composite_scores.corr(method='pearson')

# Round to 3 decimal places for readability
print(correlation_matrix.round(3).to_string())