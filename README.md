# Comically Small Transformers

This is a learning repo. In this repo I implement various transformer models from scratch. The goal is to understand the transformer architecture and its working in depth.

This repo is HEAVILY based on the [excellent video](https://youtu.be/kCc8FmEb1nY?si=ciY6QYc7iLsRguEv) from Karpathy and his [nanoGPT repo](https://github.com/karpathy/nanoGPT)

I have made no attempt to optimise these models for production or so that you can load them into HuggingFace. This is purely for learning purposes.
'

## Models Implemented

### GPT (sort of 2?)

This is very much based upon the nanoGPT model. The big changes are in the training code

```shell
python -m src.train_a_gpt
```

## Data

This repo includes the tiny Shakespeare dataset from the nanoGPT repo. You can download it from the nanoGPT repo or use your own data.