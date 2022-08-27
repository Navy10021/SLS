# ! pip install transformers
from transformers import AutoTokenizer, RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(">> We use {}".format(device))

##################################
## 1. Load PLMs for Re-Training ##
##################################
model_name = "klue/roberta-base" # your PLMs
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)

#########################
## 2. Legal DataLoader ##
#########################
# The block_size argument gives the largest token length supported by the LM to be trained.
# RoBERTa model supports sequences of length 512 (including special tokens like <s> (start of sequence) and </s> (end of sequence)
dataset = LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = "./data/cklc-small.txt",
    block_size = 512,
    )

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm = True,
    mlm_probability = 0.20,
    )

##############
## 3. Train ##
##############
# Setting for Train
args = TrainingArguments(
    output_dir = "./output/mlm-roberta",
    overwrite_output_dir = True,
    per_device_train_batch_size = 8,
    num_train_epochs = 12,
    seed = 1,
    logging_steps = 1000,
    save_steps = 3000,
    save_total_limit = 5,
    )

trainer = Trainer(model = model,
                  args = args,
                  data_collator = data_collator,
                  train_dataset = dataset
                  )

# Train
trainer.train()

# Save final model
trainer.save_model("./output/mlm-retrained-roberta")
