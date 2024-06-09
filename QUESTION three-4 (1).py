#!/usr/bin/env python
# coding: utf-8

# ## Q3. How has the distribution of sunshine hours influenced the geographical distribution of solar energy generation capacity in the UK up to 2022?
# 

# In[2]:


import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
from IPython.display import display


# ### Obtain average annual sunshine hours for different regions in the UK.
# 
# ### Sunshine Hours Scotland

# In[3]:


# Download the data from the Met Office website
url = "https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Sunshine/date/Scotland.txt"
response = requests.get(url)


# In[4]:


# URL to the dataset
url = "https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Sunshine/date/Scotland.txt"


# In[5]:


# Call the the data
response = requests.get(url)
data_str = response.text


# In[6]:


# Display the first few lines to understand the structure
print(data_str.split('\n')[:10])


# In[7]:


# Read the data into a pandas DataFrame
data = pd.read_csv(StringIO(data_str), delim_whitespace=True, skiprows=6)


# In[8]:


# Display the first few rows of the DataFrame
print(data.head())


# In[9]:


# Manually set the column names to ensure correctness
data.columns = ['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'win', 'spr', 'sum', 'aut', 'ann']

# Convert the 'year' column to integer
data['year'] = data['year'].astype(int)


# In[10]:


#Handling NaN values

# Convert all other columns to numeric and handle non-numeric values
for column in data.columns[1:]:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Fill NaN values with the mean of each column
data.fillna(data.mean(), inplace=True)

# Set the 'year' column as the index
data = data.set_index('year')

# Calculate the mean sunshine hours for each year
mean_sunshine_hours = data.mean(axis=1)

# Display the mean sunshine hours for each year
print(mean_sunshine_hours)


# In[11]:


# Display as a table 

#Rename the columns for clarity
mean_sunshine_hours.columns = ['Year', 'Mean Sunshine Hours']


# In[12]:


# Convert the Series to a DataFrame for easier plotting
mean_sunshine_hours = mean_sunshine_hours.reset_index()
mean_sunshine_hours.columns = ['Year', 'Mean Sunshine Hours']

# Display the DataFrame as a table
print(mean_sunshine_hours)


# In[13]:


# Plot as bar chart
plt.figure(figsize=(12, 6))
plt.bar(mean_sunshine_hours['Year'], mean_sunshine_hours['Mean Sunshine Hours'], color='purple')
plt.title('Average Sunshine Hours Each Year in Scotland')
plt.xlabel('Year')
plt.ylabel('Mean Sunshine Hours')
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
plt.grid(True)
plt.show()


# ### Sunshine Hours Midlands

# In[14]:


# URL to the dataset for the Midlands
url_midlands = "https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Sunshine/date/Midlands.txt"

# Fetch the data
response_midlands = requests.get(url_midlands)
data_str_midlands = response_midlands.text

# Read the data into a pandas DataFrame
data_midlands = pd.read_csv(StringIO(data_str_midlands), delim_whitespace=True, skiprows=6)

# Manually set the column names to ensure correctness
data_midlands.columns = ['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'win', 'spr', 'sum', 'aut', 'ann']

# Convert the 'year' column to integer
data_midlands['year'] = data_midlands['year'].astype(int)

# Convert all other columns to numeric and handle non-numeric values
for column in data_midlands.columns[1:]:
    data_midlands[column] = pd.to_numeric(data_midlands[column], errors='coerce')

# Fill NaN values with the mean of each column
data_midlands.fillna(data_midlands.mean(), inplace=True)

# Set the 'year' column as the index
data_midlands.set_index('year', inplace=True)

# Calculate the mean sunshine hours for each year
mean_sunshine_hours_midlands = data_midlands[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']].mean(axis=1)

# Convert the Series to a DataFrame for easier plotting
mean_sunshine_hours_midlands = mean_sunshine_hours_midlands.reset_index()
mean_sunshine_hours_midlands.columns = ['Year', 'Mean Sunshine Hours']

# Display the DataFrame as a table
print(mean_sunshine_hours_midlands)

# Plot as a bar chart
plt.figure(figsize=(12, 6))
plt.bar(mean_sunshine_hours_midlands['Year'], mean_sunshine_hours_midlands['Mean Sunshine Hours'], color='salmon')
plt.title('Average Sunshine Hours Each Year in the Midlands')
plt.xlabel('Year')
plt.ylabel('Mean Sunshine Hours')
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
plt.grid(True)
plt.show()


# ### Sunshine Hours England N 

# In[15]:


# URL to the dataset for Northern England
url_england_n = "https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Sunshine/date/England_N.txt"

# Fetch the data
response_england_n = requests.get(url_england_n)
data_str_england_n = response_england_n.text

# Read the data into a pandas DataFrame
data_england_n = pd.read_csv(StringIO(data_str_england_n), delim_whitespace=True, skiprows=6)

# Manually set the column names to ensure correctness
data_england_n.columns = ['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'win', 'spr', 'sum', 'aut', 'ann']

# Convert the 'year' column to integer
data_england_n['year'] = data_england_n['year'].astype(int)

# Convert all other columns to numeric and handle non-numeric values
for column in data_england_n.columns[1:]:
    data_england_n[column] = pd.to_numeric(data_england_n[column], errors='coerce')

# Fill NaN values with the mean of each column
data_england_n.fillna(data_england_n.mean(), inplace=True)

# Set the 'year' column as the index
data_england_n.set_index('year', inplace=True)

# Calculate the mean sunshine hours for each year
mean_sunshine_hours_england_n = data_england_n[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']].mean(axis=1)

# Convert the Series to a DataFrame for easier plotting
mean_sunshine_hours_england_n = mean_sunshine_hours_england_n.reset_index()
mean_sunshine_hours_england_n.columns = ['Year', 'Mean Sunshine Hours']

# Display the DataFrame as a table
print(mean_sunshine_hours_england_n)

# Plot as a bar chart
plt.figure(figsize=(12, 6))
plt.bar(mean_sunshine_hours_england_n['Year'], mean_sunshine_hours_england_n['Mean Sunshine Hours'], color='lightgreen')
plt.title('Average Sunshine Hours Each Year in Northern England')
plt.xlabel('Year')
plt.ylabel('Mean Sunshine Hours')
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
plt.grid(True)
plt.show()


# ### Sunshine Hours England S

# In[16]:


# URL to the dataset for Southern England
url_england_s = "https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Sunshine/date/England_S.txt"

# Fetch the data
response_england_s = requests.get(url_england_s)
data_str_england_s = response_england_s.text

# Read the data into a pandas DataFrame
data_england_s = pd.read_csv(StringIO(data_str_england_s), delim_whitespace=True, skiprows=6)

# Manually set the column names to ensure correctness
data_england_s.columns = ['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'win', 'spr', 'sum', 'aut', 'ann']

# Convert the 'year' column to integer
data_england_s['year'] = data_england_s['year'].astype(int)

# Convert all other columns to numeric and handle non-numeric values
for column in data_england_s.columns[1:]:
    data_england_s[column] = pd.to_numeric(data_england_s[column], errors='coerce')

# Fill NaN values with the mean of each column
data_england_s.fillna(data_england_s.mean(), inplace=True)

# Set the 'year' column as the index
data_england_s.set_index('year', inplace=True)

# Calculate the mean sunshine hours for each year
mean_sunshine_hours_england_s = data_england_s[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']].mean(axis=1)

# Convert the Series to a DataFrame for easier plotting
mean_sunshine_hours_england_s = mean_sunshine_hours_england_s.reset_index()
mean_sunshine_hours_england_s.columns = ['Year', 'Mean Sunshine Hours']

# Display the DataFrame as a table
print(mean_sunshine_hours_england_s)

# Plot as a bar chart
plt.figure(figsize=(12, 6))
plt.bar(mean_sunshine_hours_england_s['Year'], mean_sunshine_hours_england_s['Mean Sunshine Hours'], color='lightcoral')
plt.title('Average Sunshine Hours Each Year in Southern England')
plt.xlabel('Year')
plt.ylabel('Mean Sunshine Hours')
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
plt.grid(True)
plt.show()


# ### Sunshine Hours Northern Ireland

# In[17]:


# URL to the dataset for Northern Ireland
url_northern_ireland = "https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Sunshine/date/Northern_Ireland.txt"

# Fetch the data
response_northern_ireland = requests.get(url_northern_ireland)
data_str_northern_ireland = response_northern_ireland.text

# Read the data into a pandas DataFrame
data_northern_ireland = pd.read_csv(StringIO(data_str_northern_ireland), delim_whitespace=True, skiprows=6)

# Manually set the column names to ensure correctness
data_northern_ireland.columns = ['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'win', 'spr', 'sum', 'aut', 'ann']

# Convert the 'year' column to integer
data_northern_ireland['year'] = data_northern_ireland['year'].astype(int)

# Convert all other columns to numeric, forcing errors to NaN
for column in data_northern_ireland.columns[1:]:
    data_northern_ireland[column] = pd.to_numeric(data_northern_ireland[column], errors='coerce')

# Handle missing values (example: fill with the column mean)
data_northern_ireland.fillna(data_northern_ireland.mean(), inplace=True)

# Calculate the mean sunshine hours for each year
mean_sunshine_hours_northern_ireland = data_northern_ireland[['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']].set_index('year').mean(axis=1).reset_index()

# Rename the columns for clarity
mean_sunshine_hours_northern_ireland.columns = ['Year', 'Mean Sunshine Hours']

# Display the DataFrame as a table
display(mean_sunshine_hours_northern_ireland)

# Plot as a bar chart
plt.figure(figsize=(12, 6))
plt.bar(mean_sunshine_hours_northern_ireland['Year'], mean_sunshine_hours_northern_ireland['Mean Sunshine Hours'], color='pink')
plt.title('Average Sunshine Hours Each Year in Northern Ireland')
plt.xlabel('Year')
plt.ylabel('Mean Sunshine Hours')
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
plt.grid(True)
plt.show()


# ### Sunshine Hours Wales

# In[18]:


# URL to the dataset for Wales
url_wales = "https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Sunshine/date/Wales.txt"

# Fetch the data
response_wales = requests.get(url_wales)
data_str_wales = response_wales.text

# Read the data into a pandas DataFrame
data_wales = pd.read_csv(StringIO(data_str_wales), delim_whitespace=True, skiprows=6)

# Manually set the column names to ensure correctness
data_wales.columns = ['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'win', 'spr', 'sum', 'aut', 'ann']

# Convert the 'year' column to integer
data_wales['year'] = data_wales['year'].astype(int)

# Convert all other columns to numeric, forcing errors to NaN
for column in data_wales.columns[1:]:
    data_wales[column] = pd.to_numeric(data_wales[column], errors='coerce')

# Handle missing values (example: fill with the column mean)
data_wales.fillna(data_wales.mean(), inplace=True)

# Calculate the mean sunshine hours for each year
mean_sunshine_hours_wales = data_wales[['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']].set_index('year').mean(axis=1).reset_index()

# Rename the columns for clarity
mean_sunshine_hours_wales.columns = ['Year', 'Mean Sunshine Hours']

# Display the DataFrame as a table
display(mean_sunshine_hours_wales)

# Plot as a bar chart
plt.figure(figsize=(12, 6))
plt.bar(mean_sunshine_hours_wales['Year'], mean_sunshine_hours_wales['Mean Sunshine Hours'], color='purple')
plt.title('Average Sunshine Hours Each Year in Wales')
plt.xlabel('Year')
plt.ylabel('Mean Sunshine Hours')
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
plt.grid(True)
plt.show()


# ### Generate and display a Box and Whiskers Diagram showing the mean annual sunshine hours for Wales, Scotland, Midlands, England S, England N, Northern Ireland

# In[19]:


# Dictionary to hold the URLs and region names
urls = {
    "Wales": url_wales,
    "Scotland": url,
    "Midlands": url_midlands,
    "England S": url_england_s,
    "England N": url_england_n,
    "Northern Ireland": url_northern_ireland
}

# Function to fetch and process data
def fetch_and_process_data(url):
    response = requests.get(url)
    data_str = response.text
    data = pd.read_csv(StringIO(data_str), delim_whitespace=True, skiprows=6)
    data.columns = ['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'win', 'spr', 'sum', 'aut', 'ann']
    data['year'] = data['year'].astype(int)
    for column in data.columns[1:]:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    data.fillna(data.mean(), inplace=True)
    data.set_index('year', inplace=True)
    mean_sunshine_hours = data[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']].mean(axis=1)
    mean_sunshine_hours = mean_sunshine_hours.reset_index()
    mean_sunshine_hours.columns = ['Year', 'Mean Sunshine Hours']
    return mean_sunshine_hours

# Fetch and process data for each region and store the mean annual sunshine hours
regions_data = {}
for region, url in urls.items():
    mean_sunshine_hours = fetch_and_process_data(url)
    regions_data[region] = mean_sunshine_hours['Mean Sunshine Hours'].mean()

# Prepare data for plotting
df_mean_hours = pd.DataFrame(list(regions_data.items()), columns=['Region', 'Mean Sunshine Hours'])

# Plot the combined results as a bar chart
plt.figure(figsize=(12, 6))
plt.bar(df_mean_hours['Region'], df_mean_hours['Mean Sunshine Hours'], color=['blue', 'green', 'red', 'orange', 'purple', 'cyan'])
plt.title('Mean Annual Sunshine Hours per Region')
plt.xlabel('Region')
plt.ylabel('Mean Sunshine Hours')
plt.grid(True)
plt.show()


# In[84]:


# Fetch and process data for each region and store the mean annual sunshine hours
regions_data = {}
for region, url in urls.items():
    mean_sunshine_hours = fetch_and_process_data(url)
    regions_data[region] = mean_sunshine_hours['Mean Sunshine Hours'].mean()


# In[85]:


# Prepare the summary DataFrame
summary_df = pd.DataFrame(list(regions_data.items()), columns=['Region', 'Mean Sunshine Hours'])


# In[86]:


# Display the summary table
print(summary_df)


# In[20]:


# Fetch and process data for each region
regions_data = {region: fetch_and_process_data(url) for region, url in urls.items()}

# Prepare data for plotting
df_mean_hours = pd.concat([df.set_index('Year') for df in regions_data.values()], axis=1)
df_mean_hours.columns = regions_data.keys()

# Plot the combined results as a box-and-whisker plot
plt.figure(figsize=(12, 6))
df_mean_hours.plot(kind='box')
plt.title('Distribution of Annual Sunshine Hours per Region')
plt.xlabel('Region')
plt.ylabel('Sunshine Hours')
plt.grid(True)
plt.show()


# ### Obtain and Clean Solar Data set

# In[21]:


import pandas as pd


# In[22]:


# Path to the CSV file
file_path = '/Users/shaniquesmith/Desktop/CFG/Final Project/group_4_data/Data/Regional_spreadsheets__2003-2022__-_installed_capacity__MW_.csv'


# In[23]:


# Load the dataset
data_full = pd.read_csv(file_path, skiprows=5, header=None)


# In[24]:


print("Full dataset:")
display(data_full)


# In[25]:


# Find the start index
start_index = data_full[data_full.iloc[:, 0].str.contains('MW installed capacity - PV', na=False)].index[0]


# In[26]:


# Find the end index
end_index = data_full[data_full.iloc[:, 0].str.contains('UK Total', na=False)].index[-1]


# In[27]:


# Extract the relevant data
filtered_data = data_full.iloc[start_index:end_index + 1]


# In[28]:


filtered_data


# In[29]:


# Define new column names
new_column_names = ['Region'] + list(range(2003, 2023)) + list(filtered_data.columns[21:])


# In[30]:


# Assign new column names to the DataFrame
filtered_data.columns = new_column_names


# In[31]:


filtered_data


# In[32]:


# Drop rows with index 34 and 35
filtered_data = filtered_data.drop(index=[34, 35])


# In[33]:


# Set options to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[34]:


print (filtered_data.columns)


# In[35]:


# Drop columns named 21, 22, and 23
filtered_data = filtered_data.drop(columns=[21, 22, 23])


# In[36]:


# Drop rows from index 49 to 118
filtered_data = filtered_data.drop(index=range(49, 118))


# In[37]:


# Drop the row at index 118
filtered_data = filtered_data.drop(index=118)


# In[38]:


# Drop the row at index 36
filtered_data = filtered_data.drop(index=36)


# In[39]:


# Replace all '[y]' strings with NaN, preserving numeric values
filtered_data = filtered_data.applymap(lambda x: np.nan if x == '[y]' else x)


# In[40]:


# Convert all columns except the first one to numeric, coercing errors to NaN
filtered_data.iloc[:, 1:] = filtered_data.iloc[:, 1:].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')


# In[41]:


# Fill missing values from 2003 to 2009 using backfill method
filtered_data.loc[:, 2003:2009] = filtered_data.loc[:, 2003:2009].fillna(method='bfill', axis='columns')


# In[42]:


filtered_data['Region'] = filtered_data['Region'].astype(str)


# In[43]:


# Fill missing values from 2003 to 2009 using backfill method
filtered_data.loc[:, 2003:2009] = filtered_data.loc[:, 2003:2009].fillna(method='bfill', axis='columns')


# In[44]:


# Ensure columns 2003 to 2009 are treated as strings if they are not
columns = [str(year) for year in range(2003, 2010)]


# In[45]:


filtered_data


# In[46]:


# Print the column names to inspect them
print(filtered_data.columns)


# In[47]:


# Columns are integers, no need to convert them to strings
columns = list(range(2003, 2010))


# In[48]:


# Apply backfill for the specified rows and columns without the axis parameter
filtered_data.loc[40, columns] = filtered_data.loc[40, columns].fillna(method='bfill')
filtered_data.loc[45, columns] = filtered_data.loc[45, columns].fillna(method='bfill')


# In[49]:


filtered_data


# In[50]:


# Merge North East, North West, and Yorkshire and Humber into North of England
regions_to_merge = ['North East', 'North West', 'Yorkshire and The Humber']
merged_region_name = 'North of England'


# In[51]:


# Sum the values for the specified regions
merged_region_data = filtered_data[filtered_data['Region'].isin(regions_to_merge)].iloc[:, 1:].mean()


# In[52]:


# Create a new row for the merged region
merged_region_row = pd.Series([merged_region_name] + merged_region_data.tolist(), index=filtered_data.columns)


# In[53]:


# Append the new row to the DataFrame using pd.concat
filtered_data = pd.concat([filtered_data, pd.DataFrame([merged_region_row])], ignore_index=True)


# In[54]:


# Remove the original rows for the specified regions
filtered_data = filtered_data[~filtered_data['Region'].isin(regions_to_merge)]


# In[55]:


# Remove the original rows for the specified regions
filtered_data = filtered_data[~filtered_data['Region'].isin(regions_to_merge)]


# In[56]:


# Set options to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[57]:


# Correct the typo and merge the specified regions
filtered_data['Region'] = filtered_data['Region'].replace(
    ['North of England', 'Yorkshire & The Humber'], 'North of England')


# In[58]:


# Group by the new Region column and aggregate using mean as an example aggregation function
aggregated_data = filtered_data.groupby('Region').mean().reset_index()


# In[59]:


# Drop the row at index 8
filtered_data = filtered_data.drop(index=8)


# In[60]:


# Merge "West Midlands" and "East Midlands" into "Midlands"
filtered_data['Region'] = filtered_data['Region'].replace(
    ['West Midlands', 'East Midlands'], 'Midlands')


# In[61]:


# Group by the new Region column and aggregate
aggregated_data = filtered_data.groupby('Region').mean().reset_index()


# In[62]:


# Drop the row at index 0
filtered_data = filtered_data.drop(index=0)


# In[63]:


filtered_data


# In[64]:


# Merge "East of England," "London," "South East," and "South West" into "Southern England"
filtered_data['Region'] = filtered_data['Region'].replace(
    ['East of England', 'London', 'South East', 'South West'], 'Southern England')

# Group by the new Region column and aggregate
aggregated_data = filtered_data.groupby('Region').mean().reset_index()


# In[65]:


aggregated_data


# In[66]:


# Group by the new Region column and aggregate
solar_capacity_data = filtered_data.groupby('Region').mean().reset_index()


# In[67]:


solar_capacity_data


# In[68]:


# Merge "East of England," "London," "South East," and "South West" into "Southern England"
filtered_data['Region'] = filtered_data['Region'].replace(
    ['East of England', 'London', 'South East', 'South West'], 'Southern England')


# In[69]:


# Group by the new Region column and aggregate
solar_capacity_data = filtered_data.groupby('Region').mean().reset_index()


# In[70]:


# Melt the DataFrame for easier plotting
melted_data = filtered_data.melt(id_vars='Region', var_name='Year', value_name='Capacity')


# In[71]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create a boxplot
plt.figure(figsize=(14, 8))
sns.boxplot(x='Region', y='Capacity', data=melted_data)
plt.xticks(rotation=45)
plt.title('Box and Whiskers Plot of Solar Installation Capacity by Region')
plt.xlabel('Region')
plt.ylabel('Solar Installation Capacity (MW)')


# ## Scatter Diagram 

# In[72]:


solar_capacity_data


# In[77]:


solar_capacity_data = pd.DataFrame({
    'Region': ['Midlands', 'North of England', 'Northern Ireland', 'Scotland', 'Southern England', 'Wales'],
    '2003': [0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
    '2004': [0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
    '2005': [0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
    '2006': [0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
    '2007': [0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
    '2008': [0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
    '2009': [0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
    '2010': [4.0, 2.5, 1.0, 2.0, 10.25, 3.0],
    '2011': [62.0, 41.5, 2.0, 47.0, 132.75, 60.0],
    '2012': [118.0, 83.5, 6.0, 91.0, 221.5, 109.0],
    '2013': [175.0, 116.5, 27.0, 127.0, 415.0, 149.0],
    '2014': [285.0, 158.5, 62.0, 167.0, 848.25, 374.0],
    '2015': [549.0, 264.5, 105.0, 240.0, 1459.25, 696.0],
    '2016': [737.0, 346.5, 134.0, 292.0, 1758.75, 962.0],
    '2017': [830.0, 378.5, 252.0, 323.0, 1826.5, 1055.0],
    '2018': [841.0, 400.0, 322.0, 347.0, 1844.0, 1073.0],
    '2019': [873.0, 429.5, 334.0, 394.0, 1900.75, 1093.0],
    '2020': [883.0, 437.5, 336.0, 420.0, 1929.5, 1123.0],
    '2021': [908.0, 448.0, 340.0, 452.0, 1970.75, 1213.0],
    '2022': [992.0, 478.5, 352.0, 503.0, 2065.0, 1247.0]
})


# In[78]:


# Calculate the mean solar capacity for each region across all years
mean_solar_capacity = solar_capacity_data.set_index('Region').mean(axis=1).reset_index()
mean_solar_capacity.columns = ['Region', 'Mean_Solar_Capacity']


# In[79]:


mean_solar_capacity


# In[80]:


mean_solar_capacity = solar_capacity_data.set_index('Region').mean(axis=1).reset_index()
mean_solar_capacity.columns = ['Region', 'Mean_Solar_Capacity']


# In[87]:


# Read in DataFrames
summary_df = pd.DataFrame({
    'Region': ['Wales', 'Scotland', 'Midlands', 'England S', 'England N', 'Northern Ireland'],
    'Mean Sunshine Hours': [115.321194, 98.052218, 116.419336, 127.463023, 111.735659, 104.246645]
})

mean_solar_capacity = pd.DataFrame({
    'Region': ['Midlands', 'North of England', 'Northern Ireland', 'Scotland', 'Southern England', 'Wales'],
    'Mean_Solar_Capacity': [362.850, 179.275, 113.650, 170.250, 819.200, 457.850]
})


# In[88]:


# Merge the two DataFrames on the 'Region' column
merged_df = pd.merge(summary_df, mean_solar_capacity, left_on='Region', right_on='Region')


# In[93]:


# Ensure that region names match between the two DataFrames
mean_solar_capacity.replace({'Region': {'North of England': 'England N', 'Southern England': 'England S'}}, inplace=True)


# In[94]:


# Merge the two DataFrames on the 'Region' column
merged_df = pd.merge(summary_df, mean_solar_capacity, on='Region')


# In[95]:


# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(merged_df['Mean Sunshine Hours'], merged_df['Mean_Solar_Capacity'], color='blue')

# Add labels and title
for i, row in merged_df.iterrows():
    plt.text(row['Mean Sunshine Hours'], row['Mean_Solar_Capacity'], row['Region'], fontsize=9, ha='right')

plt.title('Scatter Plot of Mean Sunshine Hours vs. Mean Installed Solar Capacity')
plt.xlabel('Mean Sunshine Hours')
plt.ylabel('Mean Installed Solar Capacity')
plt.grid(True)

# Show plot
plt.show()


# In[ ]:




