import pandas as pd

class NLIDataset:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return tuple(row)