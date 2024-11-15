import pandas as pd

# Load the Excel file
file_path = '/shares/zne.uzh/multlearn/nipype/model2/ROI/stats_con24_8_0_neg.csv'
df_roi = pd.read_csv(file_path, sep="\t")
df_roi = df_roi.iloc[:,1:]
df_roi = df_roi.style.format(decimal='.', thousands=',', precision=2)

# Convert to LaTeX
latex_table = df_roi.hide().to_latex()
print(latex_table)
