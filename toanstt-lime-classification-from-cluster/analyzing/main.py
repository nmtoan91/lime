import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
# 0:  groundtruth label
# 1: predict label
# 2: is_correct
# 3: explaining_label
# 4: intercept_conflict
# 5: local_pred_conflict
# 6: score_conflict
#data = np.load('toanstt/data/data.npy')
#data_name = 'iris'
data_name = 'covtype'
data = np.load(f'toanstt/data/data_{data_name}.npy')

column_names = ['groundtruth_label','predict label','is_correct', 'explaining_label','intercept_conflict','local_pred_conflict','score_conflict']


input_attributes = [4,5,6]
n_class = data.shape[1]
dfs = []

for l in range(n_class):
    d = data[:,l,:]
    dfs.append(pd.DataFrame(d, columns=column_names))

sheet_names = [str(i) for i in range(n_class)]
with pd.ExcelWriter(f'toanstt/analyzing/data/data_{data_name}.xlsx') as writer:
    for df, sheet in zip(dfs, sheet_names):
        df.to_excel(writer, sheet_name=sheet, index=False)

data_new = np.empty((n_class*len(input_attributes), 2+7),dtype=object); row_index=0

for l in range(n_class):
    d = data[:,l,:]
    y = d[:,2]
    for i in range(len(input_attributes)):
        #for input_attribute in input_attributes:
        input_attribute = input_attributes[i]
        x = d[:,input_attribute]
        pearson_corr, pearson_pvalue = stats.pearsonr(x, y)
        spearman_corr, spearman_pvalue = stats.spearmanr(x, y)
        mse = mean_squared_error(x, y)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(x, y)
        
        print(f"Pearson correlation coefficient: {pearson_corr:.4f} (p-value: {pearson_pvalue:.4f})")
        print(f"Spearman correlation coefficient: {spearman_corr:.4f} (p-value: {spearman_pvalue:.4f})")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}",'\n\n')
        pass
        data_new[row_index,0 ] = l
        data_new[row_index,1 ] = column_names[i+4]
        data_new[row_index,2 ] = pearson_corr
        data_new[row_index,3 ] = pearson_pvalue
        data_new[row_index,4 ] = spearman_corr
        data_new[row_index,5 ] = spearman_pvalue
        data_new[row_index,6 ] = mse
        data_new[row_index,7 ] = rmse
        data_new[row_index,8 ] = mae
        row_index+=1
    
pass
column_names = ['explaining_class','variable','pearson_corr','pearson_pvalue','spearman_corr','spearman_pvalue','mse','rmse','mae']
df2 = pd.DataFrame(data_new, columns=column_names)
df2.to_csv(f'toanstt/analyzing/data/output_{data_name}.csv',index=False) 
pass



