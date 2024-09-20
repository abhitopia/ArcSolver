#%%
import os
import sys

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
#%%
import torch

from src.dataset import GridTokenizer, ProgramTokenizer
from src.interpreter import Interpreter, InterpreterConfig
#%%

def load_checkpoint(checkpoint_path: str, ref=False):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    prog_tokenizer = ProgramTokenizer.from_dict(state_dict['tokenizers']['program_tokenizer'])
    grid_tokenizer = GridTokenizer.from_dict(state_dict['tokenizers']['grid_tokenizer'])
    model_config = InterpreterConfig.from_dict(state_dict['model_config'])
    checkpoint_model = Interpreter(model_config, prog_tokenizer, grid_tokenizer)
    checkpoint_model.load_state_dict(state_dict['model_state_dict'])
    return checkpoint_model

model_256 = '/Users/abhishekaggarwal/synced_repos/ArcSolver/runs/V5_11Sept/A2D5M256H8B4L8_v15/checkpoint_420211.pth'
model_512 = '/Users/abhishekaggarwal/synced_repos/ArcSolver/lightning_runs/V6_18Sept/A2D5M512H16B5L8.v2/checkpoints/checkpoint_172031.pth'

paths = [model_256, model_512]

models = {}
for path in paths:
    model = load_checkpoint(path)
    models[model.n_dim] = model
    print(model.n_dim)
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Assuming you have your embeddings and token names loaded into variables `embeddings` and `token_names`.
# embeddings: numpy array of shape (num_tokens, embedding_dim)
# token_names: list of length (num_tokens)

#%%
model = models[512]
model.pte.weight.shape

# len(tokens) 
vocab_size  = len(model.prog_tokenizer)
embeddings = model.pte.weight.detach().cpu().numpy()[:vocab_size, :]
token_names = [model.prog_tokenizer.idx2token[i] for i in range(vocab_size)]

assert embeddings.shape[0] == len(token_names)

dataset_prefix = 'ARC_TRAIN_'

chosen_indices = []
chosen_tokens = []
for token in token_names:
    if token.startswith(dataset_prefix):
        chosen_indices.append(token_names.index(token))
        task_id = token[len(dataset_prefix):].split('_')[0]
        chosen_tokens.append(task_id)


embeddings = embeddings[chosen_indices, :]
token_names = chosen_tokens

print(len(token_names))

token_names
#%%
# num_samples = 1000
# # Generate random indices
# random_indices = np.random.choice(len(token_names), size=num_samples, replace=False)

# # Select a subset of embeddings and token_names using the random indices
# embeddings = embeddings[random_indices, :]
# token_names = [token_names[i] for i in random_indices]

#%%
import numpy as np
random_token_idx = np.random.choice(len(token_names))
random_token = token_names[random_token_idx]
random_embedding = embeddings[random_token_idx, :]

## Find 5 closest tokens to the random token
from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between the random token and all other tokens
cosine_similarities = cosine_similarity(random_embedding.reshape(1, -1), embeddings).flatten()

# Find the indices of the 5 most similar tokens
closest_token_indices = np.argsort(cosine_similarities)[::-1][1:6]

# Get the names of the 5 most similar tokens
closest_tokens = [token_names[i] for i in closest_token_indices]

print(f"Random token: {random_token}")
print(f"5 closest tokens: {closest_tokens}")
#%%
# %%
import plotly.express as px
from sklearn.manifold import TSNE

# Assuming 'embeddings' and 'token_names' are already defined in your environment
# Reduce dimensionality to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, verbose=1)
embeddings_2d = tsne.fit_transform(embeddings)

# Create a Plotly scatter plot
fig = px.scatter(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    text=token_names,  # Add the token names as hover text
    title="2D Visualization of Token Embeddings",
    labels={'x': 'Component 1', 'y': 'Component 2'}
)

# Update traces to show text on hover
fig.update_traces(
    mode='markers+text',
    marker=dict(size=5, opacity=0.7),
    textposition='top center',
    hoverinfo='text'
)

# Update layout to allow for zooming and adjust other layout properties
fig.update_layout(
    hovermode='closest',
    plot_bgcolor='white',
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True)
)

# Show the plot
fig.show()
# %%

task_id_1 = '57aa92db'
task_id_2 = '7df24a62'

# %%
