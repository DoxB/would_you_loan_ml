import pandas as pd
from fredapi import Fred

# FRED API Key
fred_api_key = "-"  # Replace with your actual FRED API key
fred = Fred(api_key=fred_api_key)

# Define the Treasury series IDs for 1-year to 10-year
treasury_series = {
    "1-Year": "DGS1",
    "2-Year": "DGS2",
    "3-Year": "DGS3",
    "5-Year": "DGS5",
    "7-Year": "DGS7",
    "10-Year": "DGS10"
}

# Fetch the data for each series
start_date = "2000-01-01"
end_date = "2024-12-31"
treasury_data = {}

for name, series_id in treasury_series.items():
    treasury_data[name] = fred.get_series(series_id, start_date=start_date, end_date=end_date)

# Combine the data into a single DataFrame
treasury_df = pd.DataFrame(treasury_data)
treasury_df.index = pd.to_datetime(treasury_df.index)

# Resample the data to monthly frequency
monthly_treasury_df = treasury_df.resample('M').mean()

# Save to CSV
monthly_treasury_df.to_csv("monthly_treasury_yields.csv")
print("Data saved to 'monthly_treasury_yields.csv'")
