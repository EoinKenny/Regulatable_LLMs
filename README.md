# Regulating LLMs with Human-Centered Concepts

Download reviews.260k.train.txt and annotations.json from

https://people.csail.mit.edu/taolei/beer/

run

```
conda create sdfs
conda activate adfasf

pip install pandas
pip install matplotlib
pip install seaborn
pip install nltk
pip install textblob
pip install scikit-learn
pip install matplotlib
pip install textaugment
pip install translators
pip install tqdm
pip install nlpaug
pip install transformers
pip install sacremoses
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
nltk.download('punkt')
```

Then 

```
python preprocess_data.py
python collect_data.py
python preprocess_concept_data.py
python run_tests.py
```

The results will be printed off in the terminal.
