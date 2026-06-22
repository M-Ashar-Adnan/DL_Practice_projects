import torch;
import torch.nn as nn;
import torch.nn.attention as F;
import math;


vocab = {
    "<pad>": 0,
    "i" : 1,
    "will" : 2,
    "read" : 3,
    "a" : 4,
    "book" : 5,
    "yesterday" : 6,
    "tomorrow" : 7
}

vocab_size = len(vocab);

#1. "i will read a book" -> [1, 2, 3, 4, 5]
#2. "i read a book yesterday" -> [1, 3, 4, 5, 6]

batch_of_sentences = [
    [1, 2, 3, 4, 5],
    [1, 3, 4, 5, 6]
]   

sentence_1 = [vocab["i"], vocab["will"], vocab["read"], vocab["a"], vocab["book"]]
sentence_2 = [vocab["i"], vocab["read"], vocab["a"], vocab["book"], vocab["yesterday"]]
sentence_3 = [vocab["i"], vocab["read"]]


batch_tensor = torch.tensor([
    [1, 2, 3, 4, 5],
    [1, 3, 4, 5, 6],
    [1, 3, 0, 0, 0]
])

print(f"Batch Shape: {batch_tensor.shape} -> (Batch Size: 3, Sequence Length: 5)\n")

#EMBEDDING
embedding_dim = 4
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
embedded_batch = embedding_layer(batch_tensor)

print(f"Embedded Batch Shape: {embedded_batch.shape} -> (Batch Size: 3, Sequence Length: 5, Embedding Dimension: {embedding_dim})\n")
print(f"Vector for the word 'read' in the first sentence: {embedded_batch[0] [2]}")