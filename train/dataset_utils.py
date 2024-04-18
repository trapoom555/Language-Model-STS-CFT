from datasets import load_from_disk

class NLIDataset:
    def __init__(self, path):
        self.ds = load_from_disk(path)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        row = self.ds.__getitem__(idx)
        sent0 = {
                    'input_ids': row['sent0_input_ids'],
                    'attention_mask': row['sent0_attention_mask']
                }
        sent1 = {
                    'input_ids': row['sent1_input_ids'],
                    'attention_mask': row['sent1_attention_mask']
                }
        hard_neg = {
                    'input_ids': row['hard_neg_input_ids'],
                    'attention_mask': row['hard_neg_attention_mask']
                }
        return (sent0, sent1, hard_neg)