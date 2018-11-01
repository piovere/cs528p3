import pandas as pd
from os.path import join


datadir = join("..", "..", "data", "processed")
datafile = join(datadir, "bc.csv")

df = pd.read_csv(datafile, index_col="Sample")
