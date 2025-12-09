import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_pca_inputs(batch_trials, model, episode):
    # batch_trials shape batch_size x num_trials x 2*item_size
    model.eval()
    batch_trial_embeddings = model.fc1(batch_trials) # shape batch_size x num_trials x hidden_size
    model.train()
    batch_trial_embeddings = batch_trial_embeddings.detach().cpu().numpy()


    _, _, embed_dim = batch_trial_embeddings.shape
    item_dim = embed_dim // 2

    # Split into item1 and item2
    item1 = batch_trial_embeddings[:, :, :item_dim]  # (batch, trials, item_dim)
    item2 = batch_trial_embeddings[:, :, item_dim:]  # (batch, trials, item_dim)

    # Flatten batch and trials dimensions
    item1_flat = item1.reshape(-1, item_dim)  # (batch*trials, item_dim)
    item2_flat = item2.reshape(-1, item_dim)  # (batch*trials, item_dim)

    # Stack item1 on top of item2
    batch_trial_items_embeddings = np.concatenate([item1_flat, item2_flat], axis=0)
    pca = PCA(n_components=2)
    pca.fit(batch_trial_items_embeddings)
    split_batch_trial_pca = pca.transform(batch_trial_items_embeddings)

    half = len(split_batch_trial_pca) // 2
    plt.scatter(split_batch_trial_pca[:half, 0], split_batch_trial_pca[:half, 1], c='blue', label='Item 1')
    plt.scatter(split_batch_trial_pca[half:, 0], split_batch_trial_pca[half:, 1], c='red', label='Item 2')
    plt.title(f'PCA embeddings of item dimensions Episode {episode}')
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    return fig