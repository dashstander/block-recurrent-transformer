from datasets import load_dataset
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, DataLoader
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer
from x_transformers.x_transformers import TokenEmbedding
import wandb

from block_recurrent_transformer import BlockRecurrentAttention, long_sequence_splitter


class WikiDataset:
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, i: int):
        record = self.data[i]
        title = record['title']
        text = record['text']
        return f'{title}\n\n{text}'
    
    def __len__(self):
        return len(self.data)


class BlockRecurrentDecoder(nn.Module):
    """As simple as I can make the model.
    """
    
    def __init__(self, num_tokens, dim):
        super().__init__()
        self.embed = TokenEmbedding(dim, num_tokens)
        self.attn = BlockRecurrentAttention(dim, dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, state=None):
        x, state = self.attn(self.embed(x), state)
        x = self.to_logits(self.norm(x))
        return x, state



def setup_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer



def train( data, tokenizer, config):
    model = BlockRecurrentDecoder(len(tokenizer), 512)
    model.to(device)
    opt = Adam(model.parameters())
    train_data = WikiDataset(data['train'])
    data_loader = DataLoader(train_data,  batch_size = config.batch_size, sampler = RandomSampler(train_data), pin_memory=True)
    i = 0
    for raw_batch in tqdm(data_loader):
        state = None
        article_batch = tokenizer(raw_batch, return_tensors='pt', padding=True)['input_ids']
        for text in tqdm(long_sequence_splitter(article_batch, config.window_len)):
            inputs = text[:, :-1]
            targets = text[:, 1:]
            preds, state = model(inputs, state)
            loss = cross_entropy(preds, targets)
            loss.backward()
            opt.step()
            preds, state = preds.detach(), state.detach()
            preds.to('cpu')
            i += 1
        



if __name__ == '__main__':
    device = 'cuda:0'
    data = load_dataset("wikipedia", "20220301.en")
    tokenizer = setup_tokenizer()
    config = OmegaConf.load('configs/base.yaml')
    train(data, tokenizer, config)
    