from transformers import Data2VecModel


# TODO: Change to data2vec checkpoint
checkpoint = "facebook/wav2vec2-base"
model = Data2VecModel.from_pretrained(checkpoint)
