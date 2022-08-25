#! pip install transformers
#! pip install -U sentence-transformers
#! pip install konlpy

##########################################
### 1. Load Clean Korean Legal Dataset ###
##########################################
import konlpy
from konlpy.tag import Mecab

file_path = "./data/cklc-small.txt"
with open(file_path) as f:
    lines = f.readlines()
    sent_list = [line.rstrip('\n') for line in lines]

print(">> Total Number of Corpus : {}".format(len(sent_list)))


#######################################
## 2. Denoising Auto-Encoder Dataset ##
#######################################
from torch.utils.data import Dataset
from sentence_transformers.readers.InputExample import InputExample
import numpy as np

class DenoisingAutoEncoderDataset(Dataset):
    def __init__(self, sentences, noise_fn = lambda sent :  DenoisingAutoEncoderDataset.delete(sent)):
        self.sentences = sentences
        self.noise_fn = noise_fn
        self.mecab = Mecab()

    def __getitem__(self, item):
        sent = self.sentences[item]
        return InputExample(texts=[self.noise_fn(sent), sent])  # label : Similar[0], texts : {Original text; Noised text}

    def __len__(self):
        return len(self.sentences)
    
    # Noise function
    @staticmethod
    def delete(text, del_ratio = 0.55): # ratio 60 % is best performance
        mecab = Mecab()
        words_tok = mecab.morphs(text)
        n = len(words_tok)
        if n == 0:
            return text
        
        keep_or_not = np.random.rand(n) > del_ratio  # [False, Fasle, True, True, False, True]
        if sum(keep_or_not) == 0:                    # number of [True]
            keep_or_not[np.random.choice(n)] = True
        words_processed = " ".join(np.array(words_tok)[keep_or_not])
        return words_processed


#############################################
## 3. Denoising Auto-Encoder Loss Function ##
#############################################
import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer, models
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel

class DenoisingAutoEncoderLoss(nn.Module):
    def __init__(self, model, decoder_name = None, tie_encoder_decoder = True):
        super(DenoisingAutoEncoderLoss, self).__init__()
        self.encoder = model
        self.tokenizer_encoder = model.tokenizer # model tokenizer

        encoder_name = model[0].auto_model.config._name_or_path       # model[0] : Transformer, model[0].auto_model : 'bert-base-uncased' / model[1] : Pooling
        
        if decoder_name == None:
            assert tie_encoder_decoder, "Must indicate the decoder_name argument when tie_encoder_decoder = False"
        if tie_encoder_decoder:
            decoder_name = encoder_name
        
        self.tokenizer_decoder = AutoTokenizer.from_pretrained(decoder_name)
        self.need_retokenization = not (type(self.tokenizer_encoder) == type(self.tokenizer_decoder))

        # Bert Config with Decoder & Cross attentions
        decoder_config = AutoConfig.from_pretrained(decoder_name)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        kwargs_decoder = {'config': decoder_config}
        # Make Decoder : LM Cross Attentions
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name, **kwargs_decoder)
        
        # if you use GPT-2 
        if self.tokenizer_decoder.pad_token is None:
            self.tokenizer_decoder.pad_token = self.tokenizer_decoder.eos_token
            self.decoder.config.pad_token_id = self.decoder.config.eos_token_id
        
        if tie_encoder_decoder:
            if len(self.tokenizer_encoder) != len(self.tokenizer_decoder):
                self.tokenizer_decoder = self.tokenizer_encoder
                self.decoder.resize_token_embeddings(len(self.tokenizer_decoder))  # Vocab_size(30522)
            decoder_base_model_prefix = self.decoder.base_model_prefix
            PreTrainedModel._tie_encoder_decoder_weights(
                model[0].auto_model,
                self.decoder._modules[decoder_base_model_prefix],
                self.decoder.base_model_prefix
            )

    def retokenize(self, sentence_features):
        input_ids = sentence_features['input_ids']
        device = input_ids.device
        sentences_decoded = self.tokenizer_encoder.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        retokenized = self.tokenizer_decoder(
            sentences_decoded,
            padding=True,
            truncation='longest_first',
            return_tensors="pt",
            max_length=None).to(device)
        return retokenized

    def forward(self, sentence_features, labels):
        source_features, target_features = tuple(sentence_features)  # (noised text, orginal text)
        if self.need_retokenization:
            target_features = self.retokenize(target_features)

        # 1. Sentence Embedding from Encoder
        reps = self.encoder(source_features)['sentence_embedding']  # [batch_size, hidden_dim]

        target_length = target_features['input_ids'].shape[1]
        decoder_input_ids = target_features['input_ids'].clone()[:, :target_length - 1]     # Decoder Input : input - [102] token
        label_ids = target_features['input_ids'][:, 1:]                                     # Label : input - [102] token

        # 2. Sentence Embedding from Decoder : output is CausalLMOutput with Cross Attentions
        decoder_outputs = self.decoder(
            input_ids = decoder_input_ids,
            inputs_embeds = None,
            attention_mask = None,
            encoder_hidden_states = reps[:, None],  # (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
            encoder_attention_mask = source_features['attention_mask'][:, 0:1],
            labels = None,
            return_dict = None,
            use_cache = False)  # decoder_outputs : [loss = None, logits]
        
        # 3. Calculate Loss
        lm_logits = decoder_outputs[0]  # logits : [batch_size, seq_length, vocab_size]
        ce_loss_fct = nn.CrossEntropyLoss(ignore_index = self.tokenizer_decoder.pad_token_id)
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label_ids.reshape(-1)) # CE_Loss([seq_length, vocab_size], [seq_length])
        return loss


################################################
## 4. TSDAE unsupervised-embeddings training  ##
################################################
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models

# dataset with noise function
train_data = DenoisingAutoEncoderDataset(sent_list)

# dataloader
loader = DataLoader(
    train_data,
    batch_size = 8,
    shuffle = True, 
    drop_last = True)

# Transformers models
model_name = 'klue/bert-base' 
bert = models.Transformer(model_name)

# Sentence Embedding using [CLS] token or Mean/Max Pooling
pooling = models.Pooling(bert.get_word_embedding_dimension(), 'mean') # cls, mean, max
model = SentenceTransformer(modules = [bert, pooling])

# Use Loss function
loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder = True)

# Train
epochs = 3
warmup_steps = int(len(loader) * epochs * 0.10) # Warmup 10 %

model.fit(
    train_objectives=[(loader, loss)],
    epochs = epochs,
    warmup_steps = warmup_steps,
    checkpoint_path = './output/tsdae-bert',
    checkpoint_save_steps= 3000,
    weight_decay = 0,
    scheduler = 'constantlr',
    optimizer_params = {'lr': 3e-5},
    show_progress_bar = True
)
# Save final model
model.save('output/tsdae-bert')