import pandas as pd
import glob
import os

# Read in the data
total_df = pd.DataFrame()
for file in glob.glob("data/hyperparam_tune/*.csv"):
    if "total" in file:
        continue
    df = pd.read_csv(file)
    total_df = pd.concat([total_df, df])
total_df.drop_duplicates(subset=total_df.columns.difference(['current']), inplace=True)
total_df.sort_values(by="current", inplace=True, ascending=False)
total_df.to_csv("data/hyperparam_tune/total.csv", index=False)
# if current > 0.5: move corresponding plot to "best" folder
selected = total_df[total_df["current"] > 0.5]
for index, row in selected.iterrows():
    filename = f"example_model_{int(row['observation_distance'])}_{int(row['batch_size'])}_{row['gamma']}_{row['eps_start']}_{row['eps_end']}_{int(row['eps_decay'])}_{row['tau']}_{row['lr']}_{int(row['memory_size'])}.png"
    print(filename)
    file = glob.glob("plots/hyperparam_tune/" + filename)[0]
    os.rename(file, f"plots/hyperparam_tune/best/{filename}")
