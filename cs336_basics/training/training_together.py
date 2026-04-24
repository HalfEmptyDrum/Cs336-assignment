import torch
from cs336_basics.tokenizer.tokenizer import train_bpe
from cs336_basics.tokenizer.encoder import Tokenizer
from cs336_basics.tokenizer.save_bpe import save_bpe
from cs336_basics.model.transformer_lm import TransformerLanguageModel
from cs336_basics.training.adamw import AdamW

from cs336_basics.training.data_loading import data_loading

from cs336_basics.training.cross_entropy import cross_entropy

from einops import rearrange

from cs336_basics.training.learning_rate_schedule import lr_cosine_schedule

from cs336_basics.training.checkpointing import save_checkpoint



def train():
    print("train()")
    # TODO: make these args command-line inputs.
    
    d_model = 512
    d_ff = 1344
    rope_theta = 10000
    context_length = 256
    vocab_size = 10000
    
    device = 'cuda'
    
    num_heads = 16
    
    num_layers = 4
    
    batch_size = 32
    
    training_steps = 1000
    
    special_tokens = ['<|endoftext|>']
    
    ckpt_path = "checkpoints/"
    
    vocab_path = "vocab/vocab_full.json"
    merges_path = "merges/merges_full.json"
    
    text_filepath = "TinyStoriesV2-GPT4-train.txt"
    
    
    # print("training the tokenizer...")
    # vocab, merges = train_bpe(text_filepath, vocab_size, special_tokens)
    
    # print("done training the tokenizer!")
    # print("saving vocab and merges...")
    # save_bpe(vocab, merges, vocab_path, merges_path)
    # print("done saving vocab and merges")
    
    print("creating the tokenizer...")
    
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    with open(text_filepath, "r", encoding="utf-8") as f:
        token_ids = list(tokenizer.encode_iterable(f))   # or iterate lazily
        
    
    
    language_model = TransformerLanguageModel(vocab_size=vocab_size, context_length=context_length, num_layers=num_layers, d_model=d_model, d_ff=d_ff, num_heads=num_heads, rope_theta=rope_theta, device=device)
    
    language_model = torch.compile(language_model, backend="aot_eager")
    
    optimizer = AdamW(language_model.parameters(), lr=0.01, weight_decay=0.9, betas=(0.99, 0.999))
    
    
    print("starting training...")
    # Let's train
    
    for it in range(training_steps):
        # generate batch:
        x, y = data_loading(token_ids, batch_size, context_length, device)
        
        # zero out gradients from the previous step.
        optimizer.zero_grad()
        
        # forward pass
        y_pred = language_model(x)
        
        
        y_pred = rearrange(y_pred, "B T V -> (B T) V")
        y = rearrange(y, "B T -> (B T)")
        
        loss = cross_entropy(y_pred, y)
        
        loss.backward()
        
        optimizer.step()
        
        if it % 50 == 0:
            print(f"iteration {it}, loss = {loss.item()}")
            
        if it % 100 == 0:
            save_checkpoint(language_model, optimizer, it, ckpt_path)
        
  
    print(f"final loss = {loss.item()}")
        
        
        
        
        
    
    
if __name__ == "__main__":
    train()
    
    
    
    
    
    
    
    
    
    
    
    
    
    