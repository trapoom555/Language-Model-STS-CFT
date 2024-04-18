from datasets import load_dataset
from transformers import AutoTokenizer

class NLIPreprocess:
    def __init__(self, path):
        self.ds = load_dataset("csv", data_files=path)['train']

        tokenizer_path = '../pretrained/MiniCPM-2B-dpo-bf16/'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self._preprocess()
    
    def _tokenize(self, text, id):

        out = self.tokenizer(text, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=512)
        
        out[id + '_input_ids'] = out.pop('input_ids')
        out[id + '_attention_mask'] = out.pop('attention_mask')

        return out
    
    def _preprocess(self):
        self.ds = self.ds.map(
            lambda x: self._tokenize(x['sent0'], 'sent0'), batched=True)
        self.ds = self.ds.map(
            lambda x: self._tokenize(x['sent1'], 'sent1'), batched=True)
        self.ds = self.ds.map(
            lambda x: self._tokenize(x['hard_neg'], 'hard_neg'), batched=True)
        self.ds.set_format(
            type="torch", 
            columns=["sent0_input_ids", "sent0_attention_mask",
                     "sent1_input_ids", "sent1_attention_mask",
                     "hard_neg_input_ids", "hard_neg_attention_mask",]
        )

nlip = NLIPreprocess('nli_for_simcse.csv')
nlip.ds.save_to_disk("./processed/")