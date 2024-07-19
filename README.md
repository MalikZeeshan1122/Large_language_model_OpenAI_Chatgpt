# Large_language_model_OpenAI_Chatgpt
# Introduction 

This repository contains the code for training a large language model using OpenAI's GPT architecture. The model is trained on the [Wikitext-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset, which is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. The model is trained using the [Hugging Face Transformers]

# Requirements
- Python 3.6+        - PyTorch 1.0+  - Hugging Face Transformers 2.0+  - tqdm 4.0+  - numpy 1.0+  - tensorboard 2.0+  - torchtext 0.0+  - spacy 2.0+  - pandas 1.0+  - matplotlib 3.0+  - seaborn 0.0+  - scikit-learn 0.0+  - nltk 3.0+  - sentencepiece 0.0+  - sacremoses 0.0+  - tokenizers 0.0+  - transformers 2.0+  - datasets 1.0+  - torchmetrics 0.0+  - torchsummary 1.0+  - torchviz 0.0+  - torchtext 0.0+  - torchinfo 0.0+  - torchcontrib 0.0+  - torchbearer 0.0+  - torchsparse  0.0+  - torchsparseattn 

# Installation
To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

# Training
To train the model, run the following command:

```bash
python train.py
```

# Inference
To generate text using the trained model, run the following command:

```bash
python generate.py
```
