from Parser import df_LM, df_Hd
from Get_dates import final_merged_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_LM.rename(columns={'accession number': 'accession'}, inplace=True)
df_Hd.rename(columns={'accession number': 'accession'}, inplace=True)

df_LM_merged = pd.merge(final_merged_data, df_LM, on='accession', how='inner')
df_Hd_merged = pd.merge(final_merged_data, df_Hd, on='accession', how='inner')

df_LM_merged['tf-idf_quintile'] = pd.qcut(df_LM_merged['tf-idf'], 5, labels=False)
df_Hd_merged['tf-idf_quintile'] = pd.qcut(df_Hd_merged['tf-idf'], 5, labels=False)
#df_LM_merged['proportion_weight_quintile'] = pd.qcut(df_LM_merged['proportion weight'], 5, labels=False)
#df_Hd_merged['proportion_weight_quintile'] = pd.qcut(df_Hd_merged['proportion weight'], 5, labels=False)

def adjusted_line_quintiles_plot(df1, df2, quintile_col, y_col, title):
    # Calculating the median instead of the mean
    df1_grouped = df1.groupby(quintile_col)[y_col].median()
    df2_grouped = df2.groupby(quintile_col)[y_col].median()
    
    # X-axis positions
    r = np.arange(len(df1_grouped))

    # Plotting the lines
    plt.plot(r, df1_grouped*100, color='black', marker='o', label='Fin-Neg', linestyle='-', linewidth=2)
    plt.plot(r, df2_grouped*100, color='gray', marker='x', label='H4N-Inf', linestyle=':', linewidth=2)

    # Labels, Title, and Ticks
    plt.xlabel('Quintile', fontweight='bold')
    plt.ylabel('Median Filing Period Excess Return (%)')
    plt.title(title)
    plt.xticks(r, ['Low', '2', '3', '4', 'High'])
    
    # Adding the legend
    plt.legend(loc='upper right')
    
    # Displaying the plot
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Plotting tf-idf quintiles against median excess returns
adjusted_line_quintiles_plot(df_LM_merged, df_Hd_merged, 'tf-idf_quintile', 'excess_return', 'TF-IDF Quintiles vs Excess Return')

# Plotting proportion weight quintiles against median excess returns
#adjusted_line_quintiles_plot(df_LM_merged, df_Hd_merged, 'proportion_weight_quintile', 'excess_return', 'Proportion Weight Quintiles vs Excess Return')
