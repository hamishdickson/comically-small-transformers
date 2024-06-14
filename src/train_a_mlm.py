import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .models import bert
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
    epochs = 10
    max_length = 32


training_config = TRAININGCONFIG()

with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()

# tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize a tokenizer with the BPE model
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Instantiate a pre-tokenizer that splits the input on whitespace
tokenizer.pre_tokenizer = Whitespace()

# Create a trainer object with desired training configuration
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=5000,  # Specify the desired vocabulary size
    min_frequency=2    # Minimum frequency for a token to be included in the vocabulary
)

files = ["data/tiny_shakespeare.txt"]

# Train the tokenizer on the provided files
tokenizer.train(files, trainer)


# Save the tokenizer to a directory
tokenizer.save("src/tokenizers/bpe/tokenizer.json")

tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})

if tokenizer.cls_token is None:
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})

if tokenizer.sep_token is None:
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})


print(text[:300])

print(tokenizer.decode(tokenizer.encode(text[:300])))

# tokenizer = toy_tokenizers.CharLevelTokenizer()
# tokenizer.train(text)

# now lets get the model config
model_config = bert.Config()
model_config.vocab_size = tokenizer.vocab_size

print(f"Vocab size: {model_config.vocab_size}")


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # raise Exception(inputs)
        input_ids = inputs["input_ids"]
        output_ids = input_ids.clone()

        # mask 15% of the input ids
        mask_token = self.tokenizer.mask_token_id
        mask_indices = torch.rand(input_ids.shape) < 0.15
        input_ids[mask_indices] = mask_token  

        return input_ids, output_ids


def create_dataloaders(text, tokenizer, block_size, batch_size, split_ratio=0.9):
    data = text.split("\n")

    # remove empty strings
    data = [d for d in data if len(d) > 0]

    # Split data into training and validation sets
    n = int(split_ratio * len(data))
    train_data, val_data = data[:n], data[n:]

    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # Create dataset objects
    train_dataset = TextDataset(train_data, tokenizer, training_config.max_length)
    val_dataset = TextDataset(val_data, tokenizer, training_config.max_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    return train_loader, val_loader


train_data, val_data = create_dataloaders(
    text, tokenizer, model_config.block_size, batch_size=training_config.batch_size
)


model = bert.BERT(model_config)

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
    for input_ids, output_ids in tqdm(train_data):

        # raise Exception(input_ids)

        # evaluate the loss
        logits, loss = model(input_ids.squeeze(1), output_ids.squeeze(1))
        optimizer.zero_grad(set_to_none=True)  # TIL this existed
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), training_config.max_grad)
        optimizer.step()
        scheduler.step()

    print(loss.item())


print(loss.item())


input_string = "hello my name is Hamish and I'm an ML something, scientist?"
input_ids = tokenizer(input_string, return_tensors="pt")["input_ids"]

# mask 15% of the input ids
mask_token = tokenizer.mask_token_id
mask_indices = torch.rand(input_ids.shape) < 0.15
input_ids[mask_indices] = mask_token

print(f"mask token: {mask_token}")
print(input_ids)

model.eval()
logits, _ = model(input_ids.cuda())
predicted_ids = torch.argmax(logits, dim=-1)
predicted_string = tokenizer.decode(predicted_ids[0])

print(predicted_string)