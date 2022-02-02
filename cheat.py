import copy
import os
import tarfile
import urllib.request

from transformers import RobertaTokenizer

from data2vec_text import Data2VecTextModel


class Data2VecFairseqProxy():
    def __init__(self, module):
        self.module = module

    @classmethod
    def from_pretrained(cls, mname):
        ckpt = f"{mname}.pt"
        cls._download_weights(model=ckpt)
        return cls(Data2VecTextModel.from_pretrained("roberta/roberta.large", checkpoint_file=ckpt).models[0].encoder.sentence_encoder)
    
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


def get_outputs(model, tokenizer, sentence):
    assert "<mask>" in sentence, "Please, input a sentence containng <mask>"
    tokens = tokenizer(sentence, return_tensors="pt")["input_ids"]
    return model.module(tokens)["encoder_out"][0]


tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model = Data2VecFairseqProxy.from_pretrained("nlp_base")
model.module.eval()
outputs = get_outputs(model, tokenizer, "The color of an <mask> is orange.")
print(outputs)
