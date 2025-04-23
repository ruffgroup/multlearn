import pandas as pd

# Load the Excel file
file_path = '/shares/zne.uzh/multlearn/nipype/model2/ROI/snpm_stats_con1_8_0_pos.csv'
df_roi = pd.read_csv(file_path, sep=",")
df_roi = df_roi.iloc[:,1:]
df_roi = df_roi.style.format(decimal='.', thousands=',', precision=4)

# Convert to LaTeX
latex_table = df_roi.hide().to_latex()
print(latex_table)