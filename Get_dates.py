import pandas as pd
import os
import re
import numpy as np
import statsmodels.api as sm

base_path = '/Users/isabella/Desktop/tickers/sec-edgar-filings'

def extract_date_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        first_line = file.readline().strip()
        date_match = re.search(r': (\d{8})$', first_line)
        if date_match:
            return date_match.group(1)
    return None


filing_dates = {}


for cik in os.listdir(base_path):
    cik_dir = os.path.join(base_path, cik, '10-Q')
    if os.path.isdir(cik_dir):
        for accession_number in os.listdir(cik_dir):
            file_path = os.path.join(cik_dir, accession_number, 'full-submission.txt')
            if os.path.isfile(file_path):
                date = extract_date_from_file(file_path)
                if date:
                    filing_dates[accession_number] = date

ticker_mapping = {}
with open("/Users/isabella/Desktop/sec.gov_include_ticker.txt", "r") as f:
    for line in f:
        ticker, cik = line.strip().split("\t")
        # Ensure CIKs have the format '0000002969' to match folder names
        formatted_cik = str(cik).zfill(10)
        ticker_mapping[formatted_cik] = ticker

tickers = {}
for cik_folder in os.listdir(base_path):
    # Ensure it's a directory and in the mapping
    if os.path.isdir(os.path.join(base_path, cik_folder)) and cik_folder in ticker_mapping:
        ticker = ticker_mapping[cik_folder]
        cik_path = os.path.join(base_path, cik_folder, "10-Q")
        
        # 3. Assign the ticker to all the accession number folders/files under that CIK.
        for accession_folder in os.listdir(cik_path):
            accession_path = os.path.join(cik_path, accession_folder)
            if os.path.isdir(accession_path):
                tickers[accession_folder] = ticker

def get_trading_dates(filing_date, n, crsp_data):
    all_trading_dates = crsp_data['date'].unique()
    after_date = all_trading_dates[all_trading_dates > filing_date]
    return after_date[:n]

def compute_beta(stock_returns, market_returns):
    X = sm.add_constant(market_returns)  # Add a constant to the model (i.e., intercept)
    model = sm.OLS(stock_returns, X).fit()
    # Ensure that the model has at least two parameters before returning the beta
    if len(model.params) > 1:
        return model.params[1]
    else:
        return None 

# Convert the filing dates to datetime format
filing_dates = {key: pd.to_datetime(value) for key, value in filing_dates.items()}

# Convert it into a DataFrame for easier merging
filing_data = pd.DataFrame(list(filing_dates.items()), columns=['accession', 'filing_date'])

# Load the CRSP data and convert the date column to datetime format
crsp_data = pd.read_csv('/Users/isabella/Desktop/CRSP.csv', parse_dates=['date'])

filing_data['filing_date'] = pd.to_datetime(filing_data['filing_date'], format='%Y%m%d')
# Now, we need to add the TICKER information to the filing_data DataFrame based on the tickers mapping
crsp_data['TICKER'] = crsp_data['TICKER'].str.lower()
filing_data['TICKER'] = filing_data['accession'].map(tickers)
common_tickers = set(filing_data['TICKER'].unique()) & set(crsp_data['TICKER'].unique())
filing_data = filing_data[filing_data['TICKER'].isin(common_tickers)]
crsp_data = crsp_data[crsp_data['TICKER'].isin(common_tickers)]
crsp_data = crsp_data.sort_values(by=['TICKER', 'date'])
filing_data = filing_data.sort_values(by=['TICKER', 'filing_date'])
crsp_data['RET'] = pd.to_numeric(crsp_data['RET'], errors='coerce')

crsp_data = crsp_data.groupby(['TICKER', 'date']).reset_index()
'''crsp_data = crsp_data.groupby(['TICKER', 'date']).agg({
    'RET': 'median', 
    'sprtrn': 'first',
    'COMNAM': 'first',
    'CUSIP': 'first',
    'PERMNO': 'first'  # or some other aggregation function depending on its nature
}).reset_index()'''

filing_data = filing_data.drop_duplicates(subset=['TICKER', 'filing_date'])

merged_dfs = []

unique_tickers = filing_data['TICKER'].unique()

for ticker in unique_tickers:
    subset_filing = filing_data[filing_data['TICKER'] == ticker]
    subset_crsp = crsp_data[crsp_data['TICKER'] == ticker]
    
    if not subset_crsp.empty:  # Only merge if there's data for that ticker in crsp_data
        merged = pd.merge_asof(subset_filing, subset_crsp, left_on='filing_date', right_on='date', by='TICKER', direction='backward')
        merged_dfs.append(merged)

final_merged_data = pd.concat(merged_dfs)


# Compute beta and excess returns
for ticker in final_merged_data['TICKER'].unique():
    ticker_data = final_merged_data[final_merged_data['TICKER'] == ticker]
    
    for index, row in ticker_data.iterrows():
        start_date = row['filing_date'] - pd.Timedelta(days=121)
        end_date = row['filing_date'] - pd.Timedelta(days=1)
        
        window_data = crsp_data[(crsp_data['TICKER'] == ticker) & (crsp_data['date'] >= start_date) & (crsp_data['date'] <= end_date)]
        if not window_data.empty and not window_data[['RET', 'sprtrn']].isna().any().any():
            beta = compute_beta(window_data['RET'], window_data['sprtrn'])
            
            if beta is not None:
                post_filing_dates = get_trading_dates(row['filing_date'], 3, crsp_data)
                post_filing_data = crsp_data[(crsp_data['TICKER'] == ticker) & (crsp_data['date'].isin(post_filing_dates))]
                
                if not post_filing_data.empty:
                    excess_return = np.sum(post_filing_data['RET'] - beta * post_filing_data['sprtrn'])
                    final_merged_data.loc[index, 'excess_return'] = excess_return
                else:
                    final_merged_data.loc[index, 'excess_return'] = np.nan  # set NaN if post_filing_data is empty
            else:
                final_merged_data.loc[index, 'excess_return'] = np.nan  # Assign NaN or another appropriate value
        else:
            final_merged_data.loc[index, 'excess_return'] = np.nan  # set NaN if window_data is empty or has NaN values

print(final_merged_data[['accession', 'TICKER', 'filing_date', 'excess_return']])

