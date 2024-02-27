'''import pandas as pd
def progress_bar(progress,total):
    percent=100*(progress/total)
    bar='█' * int(percent)+'-'*(100-int(percent))
    print(f"\r|{bar}|{percent:.2f}%",end="\r")



df=pd.read_csv("spliced_spotify_dataset.csv")
print(df.info())'''

'''import pandas as pd
from tqdm import tqdm

def progress_bar(progress, total):
    percent = 100 * (progress / total)
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    print(f"\r|{bar}|{percent:.2f}%", end="\r")

# Load the CSV file
df1 = pd.read_csv("spliced_spotify_dataset.csv")
print(type(df1))

# Show progress bar while loading the CSV file
with tqdm(total=628932, desc="Loading dependencies : ") as pbar:
    df = pd.read_csv("spliced_spotify_dataset.csv", iterator=True, chunksize=100)
    for chunk in df:
        pbar.update(len(chunk))

print(type(df))'''

'''import pandas as pd
from tqdm import tqdm

def progress_bar(progress, total):
    percent = 100 * (progress / total)
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    print(f"\r|{bar}|{percent:.2f}%", end="\r")

# Load the CSV file
chunksize = 1000
df_iter = pd.read_csv("spliced_spotify_dataset.csv", chunksize=chunksize)

# Get the total number of rows in the CSV file
num_rows = sum(1 for _ in pd.read_csv("spliced_spotify_dataset.csv", chunksize=chunksize))

# Show progress bar while loading the CSV file
with tqdm(total=num_rows, desc="Loading CSV") as pbar:
    for chunk in df_iter:
        pbar.update(len(chunk))
        # do some processing here if needed

print(df.info())'''

import pandas as pd
from tqdm import tqdm

'''def progress_bar(progress, total):
    percent = 100 * (progress / total)
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    print(f"\r|{bar}|{percent:.2f}%", end="\r")'''

# Load the CSV file
# df1 = pd.read_csv("spliced_spotify_dataset.csv")
# print(type(df1))

# Show progress bar while loading the CSV file
with tqdm(total=628932, desc="Loading dependencies : ") as pbar:
    df_list = []
    df = pd.read_csv("spliced_spotify_dataset.csv", iterator=True, chunksize=1000)
    for chunk in df:
        df_list.append(chunk)
        pbar.update(len(chunk))

# Concatenate the list of DataFrames into a single DataFrame
df = pd.concat(df_list)

print(type(df))
