import scanpy as sc
import matplotlib.pyplot as plt
import pickle
import gc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import os
def evaluate_cell_embedding(adata, embedding, label_key='cell_type', resolution=1.0, output_dir='./'):
    """
    Evaluate cell embeddings using clustering metrics and visualization
    
    Parameters:
        adata: AnnData object
        embedding: 2D array of cell embeddings [n_cells, n_features]
        label_key: obs column name for true cell type labels
        resolution: Leiden clustering resolution parameter
        output_dir: path to save output files
    
    Returns:
        Dictionary of clustering metrics
    """
    # Store embeddings in AnnData
    adata.obsm['X_emb'] = embedding
    
    # Calculate neighbors and clustering
    sc.pp.neighbors(adata, use_rep="X_emb")
    sc.tl.leiden(adata, resolution=resolution)
    
    # Get cluster assignments and true labels
    y_pred = adata.obs["leiden"].astype(int).to_numpy()
    y_true = adata.obs[label_key]
    
    # Calculate clustering metrics
    scores = cluster_metrics(adata.obsm['X_emb'], y_pred=y_pred, y_true=y_true)
    
    # Visualization
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # # Compute UMAP coordinates
    # sc.tl.umap(adata, min_dist=0.3)
    
    # # Plot clustering results
    # sc.pl.umap(adata, color="leiden", ax=axes[0], show=False, legend_loc="on data")
    # axes[0].set_title("Leiden Clustering")
    
    # # Plot true cell types
    # sc.pl.umap(adata, color=label_key, ax=axes[1], show=False, legend_loc="right margin")
    # axes[1].set_title("True Cell Types")
    
    # # Save and close figure
    # plt.tight_layout()
    # plt.savefig(f'{output_dir}/cluster_umap_{resolution}.pdf')
    # plt.close()
    
    # # Cleanup
    # del adata
    # gc.collect()
    
    return scores



def cluster_metrics(X, y_pred, y_true):
    """
    embedding 是 X，聚类标签是 y_pred，真实细胞类型是 y_true
    """

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    return {'ari': ari, 'nmi': nmi}