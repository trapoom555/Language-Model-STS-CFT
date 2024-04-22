from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
torch.manual_seed(0)


class MiniCPM:
    def __init__(self):
        model_path = '../../pretrained/MiniCPM-2B-dpo-bf16'
        adapter_path = '../../pretrained/adapter'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                          torch_dtype=torch.bfloat16,
                                                          device_map='cuda',
                                                          trust_remote_code=True)
        self.model.load_adapter(adapter_path)

    def get_last_hidden_state(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to('cuda')
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True).hidden_states[-1][0, -1, :]
        return out.squeeze().float().cpu().numpy()

    def encode(self, sentences: list[str], **kwargs) -> list[np.ndarray]:
        """
        Returns a list of embeddings for the given sentences.
        
        Args:
            sentences: List of sentences to encode

        Returns:
            List of embeddings for the given sentences
        """

        out = []

        # prompt = 'This sentence: "{}" means in one word: '
        prompt = '{}'

        for s in sentences:
            prompted_text = prompt.format(s)
            out.append(self.get_last_hidden_state(prompted_text))

        return out
