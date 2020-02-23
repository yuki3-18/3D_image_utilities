import gensim
import torch
from tensorboardX import SummaryWriter

vec_path = "entity_vector.model.bin"

writer = SummaryWriter()
model = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=True)
weights = model.vectors
labels = model.index2word

# DEBUG: visualize vectors up to 1000
weights = weights[:1000]
labels = labels[:1000]

writer.add_embedding(torch.FloatTensor(weights), metadata=labels)