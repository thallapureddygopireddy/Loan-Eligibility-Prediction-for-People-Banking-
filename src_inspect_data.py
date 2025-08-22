import argparse, pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True)
args = parser.parse_args()

df = pd.read_csv(args.path)
print('Rows:', len(df), 'Cols:', len(df.columns))
print('Columns:', list(df.columns))
print(df.head())
