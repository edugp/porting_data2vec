import copy
import os
import tarfile
import urllib.request

import torch
from transformers import RobertaTokenizer

from data2vec_text import Data2VecTextModel


class Data2VecFairseqProxy():
    def __init__(self, module):
        self.module = module

    @classmethod
    def from_pretrained(cls, mname):
        ckpt = f"{mname}.pt"
        cls._download_weights(model=ckpt)
        return cls(Data2VecTextModel.from_pretrained("roberta/roberta.large", ckpt).models[0])
    
    @staticmethod
    def _download_weights(model: str="nlp_base.pt"):
        assert model in ("nlp_base.pt", "audio_base_ls.pt"), "Weights not found"
        root_url: str="https://dl.fbaipublicfiles.com/fairseq"

        if model == "nlp_base.pt":
            # Need download RoBERTa first to get the dictionary file
            if not os.path.isdir("roberta"):
                print("Downloading roberta")
                urllib.request.urlretrieve(f"{root_url}/models/roberta.large.tar.gz", "roberta.large.tar.gz")
                with tarfile.open("roberta.large.tar.gz") as f:
                    f.extractall("roberta")
                # Remove Roberta model weights and tar file
                os.remove(os.path.join("roberta", "roberta.large", "model.pt"))
                os.remove(os.path.join("roberta.large.tar.gz"))

        # Then download the actual data2vec weights
        model_url = f"{root_url}/data2vec/{model}"
        model_path = os.path.join("roberta", "roberta.large", model)
        if not os.path.isfile(model_path):
            print("Downloading model...")
            urllib.request.urlretrieve(model_url, model_path)



def predict_mask(model, tokenizer, sentence):
    assert "<mask>" in sentence, "Please, input a sentence containng <mask>"
    tokens = tokenizer(sentence, return_tensors="pt")["input_ids"]
    outputs = model.module(tokens)[0]
    masked_index = torch.nonzero(tokens == tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
    logits = outputs[0, masked_index[0][0], :]
    probs = logits.softmax(dim=-1)
    _, predictions = probs.topk(1)
    output_tokens = copy.deepcopy(tokens)
    output_tokens[0][masked_index[0][1]] = predictions[0]
    return tokenizer.decode(output_tokens[0])
    

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model = Data2VecFairseqProxy.from_pretrained("nlp_base")
model.module.eval()
predicted_mask = predict_mask(model, tokenizer, "The color of an <mask> is orange.")
print(predicted_mask)
