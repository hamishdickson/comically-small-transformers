from abc import ABC


class CharLevelTokenizer(ABC):
    def __init__(self):
        super().__init__()

        self.vocab_size = None
        self.stoi = None
        self.itos = None

    def train(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        print("Number of unique characters: ", self.vocab_size)
        print(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, x):
        return [self.stoi[ch] for ch in x]

    def decode(self, x):
        return "".join([self.itos[i] for i in x])
