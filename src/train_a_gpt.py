import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .models import gpt
from .tokenizers import toy_tokenizers

accelerator = Accelerator()
device = accelerator.device


class TRAININGCONFIG:
    lr = 1e-3
    weight_decay = 0.01
    warmup_ratio = 0.06  # from the roberta paper
    total_steps = 0
    batch_size = 128
    max_grad = 1  # not mentioned but useful
    epochs = 5


training_config = TRAININGCONFIG()

# with open("data/gpt_data.txt", "r", encoding="utf-8") as file:
#     text = file.read()

tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

dataset = load_dataset("HuggingFaceH4/no_robots")

df = dataset["train"].to_pandas()
df = df.sample(frac=0.1)
df["flattened_content"] = df.apply(lambda x: " ".join([y["content"] for y in x.messages]), axis=1)

text = tokenizer.eos_token.join(df["flattened_content"].tolist())

print(text[:300])

# tokenizer = toy_tokenizers.CharLevelTokenizer()
# tokenizer.train(text)

# now lets get the model config
model_config = gpt.Config()
model_config.vocab_size = tokenizer.vocab_size


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def create_dataloaders(text, tokenizer, block_size, batch_size, split_ratio=0.9):
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Split data into training and validation sets
    n = int(split_ratio * len(data))
    train_data, val_data = data[:n], data[n:]

    # Create dataset objects
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    return train_loader, val_loader


train_data, val_data = create_dataloaders(
    text, tokenizer, model_config.block_size, batch_size=training_config.batch_size
)


model = gpt.GPT(model_config)

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias"]
optimizer_parameters = [
    {
        "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        "weight_decay": training_config.weight_decay,
    },
    {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]


optimizer = torch.optim.AdamW(optimizer_parameters, lr=training_config.lr, weight_decay=training_config.weight_decay)


scheduler = transformers.get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(training_config.warmup_ratio * len(train_data) * training_config.epochs),
    num_training_steps=(
        training_config.total_steps if training_config.total_steps > 0 else len(train_data) * training_config.epochs
    ),
)


model, optimizer, scheduler, train_data, val_data = accelerator.prepare(
    model, optimizer, scheduler, train_data, val_data
)


for epoch in range(training_config.epochs):
    for xb, yb in tqdm(train_data):

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)  # TIL this existed
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), training_config.max_grad)
        optimizer.step()
        scheduler.step()

    print(loss.item())


print(loss.item())


# save the model
accelerator.save(model.state_dict(), "model.pth")

# now let's generate some text

kickoff = (
    torch.tensor(
        tokenizer.encode("write me the python code for Einstein's equation of general relativity"), dtype=torch.long
    )
    .to(device)
    .unsqueeze(0)
)
# torch.zeros((1,1), dtype=torch.long, device=device), # idxs batch of 1, time of 1 holding a 0 - this is a "kick off" (0 = new line char)

print(kickoff)

print(
    tokenizer.decode(
        model.generate(
            kickoff,  # idxs batch of 1, time of 1 holding a 0 - this is a "kick off" (0 = new line char)
            max_new_tokens=500,
        )[
            0
        ].tolist()  # 0 for batches, we only have a bs of 1 anyway
    )
)
