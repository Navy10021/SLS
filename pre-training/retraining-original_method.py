# ! pip install transformers
from transformers import BertTokenizer, BertForMaskedLM
import torch
from tqdm import tqdm
from transformers import TrainingArguments, Trainer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(">> We use {}".format(device))

##################################
## 1. Load PLMs for Re-Training ##
##################################
model_name = "klue/bert-base"  # your PLMs
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Clean Legal Dataset
with open('./data/cklc-small.txt', 'r') as fp:
    data = fp.read().split('\n')
print(">> Total Number of Corpus : {}".format(len(data)))


###############################################
## 2. Tokenization, Create Label and Masking ##
###############################################
# 1. Tokenization : input_ids [sent_length, 512]
inputs = tokenizer(data, return_tensors = 'pt', max_length = 512, truncation = True, padding = 'max_length')

# 2. Create Labels : input_ids.detach().clone()
inputs['labels'] = inputs.input_ids.detach().clone()

# 3. Masking : Random selection of Tokens in our input_ids tensor
# create random array of floats in equal dimension to input_ids where the random array is less than 20%, we set true except [CLS] or [SEP] tokens(101, 102)
rand = torch.rand(inputs.input_ids.shape)
mask_arr = (rand < 0.20) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)

selection = []
for i in range(inputs.input_ids.shape[0]):
    selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103

#########################
## 3. Legal DataLoader ##
#########################
class LegalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

# Make DataLoader
dataset = LegalDataset(inputs)

# Initialize DataLoader
loader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle = True)

##############
## 4. Train ##
##############
model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 10

for epoch in range(epochs):
    iter = tqdm(loader, leave = True)
    for batch in iter:
        # Initialize calculated gradients
        optim.zero_grad()
        # Pull tensors
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # LM output
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels) # outputs = {'loss', 'logits'}
        loss = outputs.loss
        # Calculate loss
        loss.backward()
        # Update parameters
        optim.step()
        iter.set_description(f' >> Epoch {epoch + 1}')
        iter.set_postfix(loss=loss.item())

# Trained model save
torch.save(model, 'output/mlm-retrained-bert')


########################
## 4-2. Using Trainer ##
########################
#from transformers import TrainingArguments, Trainer
# 1. Initialize
#args = TrainingArguments(output_dir='./output/mlm-bert', per_device_train_batch_size = 8, num_train_epochs = 10, seed = 1, logging_steps = 1000, save_steps = 1000, save_total_limit = 3)

# 2. Train with Trainer
#trainer = Trainer(model = model, args = args, train_dataset = dataset)
#trainer.train()
# Save final model
#trainer.save_model("./output/mlm-retrained-bert")
