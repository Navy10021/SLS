#! pip install datasets
#! pip install transformers
#! pip install sentencepiece

import pandas as pd
from datasets import *
from transformers import BertTokenizerFast, DistilBertTokenizerFast, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from tokenizers import *
import torch
import os
import json
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

###############################################
### 1. Loda CKLC(Clean Korean Legal Corpus) ###
###############################################
# Train - Validation Dataset Split
files = [
         './data/legal_data.txt',
         ]
dataset = load_dataset("text", data_files = files, split = "train")

data = dataset.train_test_split(test_size=0.08, shuffle=True)

# Legal data check!
for t in data['train']['text'][:5]:
    print(t)
    print("===="*50)


###################################
###  2. Tokenizer & Legal Vocab ###
###################################
def dataset_to_text(dataset, output_filename='data.txt'):
    with open(output_filename, "w") as f:
        for t in dataset['text']:
            print(t, file=f)

# Save train / text dataset -> txt 
dataset_to_text(data["train"], "./data/train.txt")
dataset_to_text(data["test"], "./data/test.txt")

# Parameters
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
files = ["./data/train.txt", "./data/test.txt"]
vocab_size = 30522
max_length = 512
truncate_longer_samples = False  # No cut!

# Initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()

# 2-4. Train the tokenizer
tokenizer.train(
    files = files,
    vocab_size = vocab_size,
    special_tokens = special_tokens)

# Enable truncation up to the maximum "512 tokens"
tokenizer.enable_truncation(max_length = max_length)

model_path = "mlm-krlawbert"
# Make the directory if not already there
if not os.path.isdir(model_path):
  os.mkdir(model_path)

# Save the tokenizer in model_path
tokenizer.save_model(model_path)

# Dumping some of the tokenizer config to config file, including special tokens, whether to lower case and the maximum sequence length
with open(os.path.join(model_path, "config.json"), "w") as f:
  tokenizer_cfg = {
      "do_lower_case": True,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "model_max_length": max_length,
      "max_len": max_length,
  }
  json.dump(tokenizer_cfg, f)

# When the tokenizer is trained and configured, load it as BertTokenizerFast
model_path ="mlm-krlawbert"
tokenizer = BertTokenizerFast.from_pretrained(model_path)


#######################################
### 3. Tokenizing the Legal Dataset ###
#######################################
def encode_with_truncation(examples):
    """Mapping function to tokenize the sentences passed with truncation"""
    return tokenizer(examples["text"],
                     truncation=True,
                     padding="max_length",
                     max_length=max_length,
                     return_special_tokens_mask=True)

def encode_without_truncation(examples):
  """Mapping function to tokenize the sentences passed without truncation"""
  return tokenizer(examples["text"], return_special_tokens_mask=True)


# 1.Encoding : The encode function will depend on the truncate_longer_samples variable
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation


# 2. Tokenizing the Train & Test Dataset 
train_dataset = data['train'].map(encode, batched=True)
test_dataset = data['test'].map(encode, batched=True)

if truncate_longer_samples:
  # remove other columns and set input_ids and attention_mask as 
  train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
  test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
  test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
  train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])


# 3. Main data processing function that will concatenate all texts from our dataset and generate chunks of max_seq_length.
def group_texts(examples):
    # 1.Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 2. Drop the small remainder
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # 3. Split by Chunk of Max_len
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

# 'batched=True' : This map processes 1,000 Texts together, so group_texts THROWS AWAY a remainder for each of those groups of 1,000 texts. 
if not truncate_longer_samples:
  train_dataset = train_dataset.map(group_texts, batched=True, batch_size=2000,
                                    desc=f"Grouping texts in chunks of {max_length}")
  test_dataset = test_dataset.map(group_texts,  batched=True, batch_size=2000,
                                  desc=f"Grouping texts in chunks of {max_length}")
  
  
# Load BERT model
# initialize the model with the config
model_config = BertConfig(vocab_size=vocab_size)

#model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-4000"))
model = BertForMaskedLM(config=model_config)

###########################################
### 4. Pre-Training(BERT Statistic MLM) ###
###########################################
# 1. MLM : Randomly Masking 20% of the tokens For the MLM Task
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                mlm=True,
                                                mlm_probability=0.2)
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=10,            # number of training epochs
    per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,              # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)

# 2. Initialize the trainer and pass everything to it
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 3. Train & Evaluation the model(N Steps)
trainer.train()
