import torch
from tqdm import tqdm

from .models import gpt
from .tokenizers import toy_tokenizers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = toy_tokenizers.CharLevelTokenizer()

with open("data/gpt_data.txt", "r", encoding="utf-8") as file:
    text = file.read()

tokenizer.train(text)


# we don't have much text, so lets encode the whole thing
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# we want to split the data so that we have a training set and a validation set
# we'll train on the first bit and validate on the second bit to avoid leakage
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

# now lets get the model config
model_config = gpt.Config()
model_config.vocab_size = tokenizer.vocab_size

model = gpt.GPT(model_config)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # note WD=1 here!!!

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix  = torch.randint(len(data) - model_config.block_size, (batch_size,)) # starting index for each sequence
    x = torch.stack([data[i:i+model_config.block_size] for i in ix]) # we're just indexing the data tensor
    y = torch.stack([data[i+1:i+model_config.block_size+1] for i in ix]) # y is the same but one ahead
    return x, y


batch_size = 32
for steps in tqdm(range(10_000)):
    # sample a batch of data
    xb, yb = get_batch('train')

    xb, yb = xb.to(device), yb.to(device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # TIL this existed
    loss.backward()
    optimizer.step()

print(loss.item())

# now let's generate some text
print(
    tokenizer.decode(
        model.generate(
            torch.zeros((1,1), dtype=torch.long, device=device), # idxs batch of 1, time of 1 holding a 0 - this is a "kick off" (0 = new line char)
            max_new_tokens=100
        )[0].tolist() # 0 for batches, we only have a bs of 1 anyway
    )
)