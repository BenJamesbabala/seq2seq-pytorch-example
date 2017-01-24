# A toy example of sequence to sequence (seq2seq) model with PyTorch.

## Toy Dataset
* __Source Sequence__: Consecutive numbers in 0-9 with length 10
* __Target Sequence__: Consecutive numbers which add 1 to each source sequence number
* __Example__:
```
Source: 3 4 5 6 7 8 9 0 1 2
Target: 4 5 6 7 8 9 0 1 2 3
```

## Train & Run
`python translate.py`

Trained model will be persisted in `model.dat`.  Remove the file to re-train.
