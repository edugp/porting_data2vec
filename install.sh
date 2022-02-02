cd transformers
git clone https://github.com/edugp/transformers/
git checkout add-data2vec
pip install -e .[dev]
cd ..

# install fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install -e .
# install mosesdecoder under fairseq
git clone https://github.com/moses-smt/mosesdecoder
# install fastBPE under fairseq
git clone https://github.com/glample/fastBPE.git
cd fastBPE; g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast; cd ../..

pip install setuptools==59.5.0