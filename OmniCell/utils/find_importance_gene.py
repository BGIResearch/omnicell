from scipy.stats import wasserstein_distance
from tqdm import tqdm
import numpy as np
from typing import List

def find_diff_genes_one_vs_rest(
    X: np.ndarray,                   
    cell_types: list,                
    gene_names: list,                
    target_cell_type: str,           
    top_k: int = 50
):

    assert len(gene_names) == X.shape[1], "The length of the gene name list must match the number of genes"
    assert len(cell_types) == X.shape[0], "The length of the cell type list must match the number of cells"

    unique_types = set(cell_types)
    assert target_cell_type in unique_types, \
        f"Target cell type '{target_cell_type}' does not exist. Available types: {list(unique_types)}"

    
       
    target_indices = [i for i, ct in enumerate(cell_types) if ct == target_cell_type]
    background_indices = [i for i, ct in enumerate(cell_types) if ct != target_cell_type]
    
       
    background_types = list({cell_types[i] for i in background_indices})
    
    num_genes = X.shape[1]
    gene_scores = []
    
       
    for gene in tqdm(range(num_genes), desc=f"Processing genes for {target_cell_type}"):
        target_emb = X[target_indices, gene, :]
        background_emb = X[background_indices, gene, :]
        
        dim_scores = [
            wasserstein_distance(target_emb[:, dim], background_emb[:, dim])
            for dim in range(X.shape[2])
        ]
        gene_scores.append(np.mean(dim_scores))
    
       
    ranked_indices = sorted(range(num_genes), key=lambda x: -gene_scores[x])[:top_k]
    top_genes = [gene_names[i] for i in ranked_indices]
    top_scores = [gene_scores[i] for i in ranked_indices]
    
    return top_genes

def find_higher_expressed_genes(
    adata, 
    target_cell_type: str, 
    gene_list: List[str], 
    cell_type_key: str = 'cell_type'
) -> List[str]:
       
    genes_to_test = [gene for gene in gene_list if gene in adata.var_names]
    
    if not genes_to_test:
        return []
    
       
    target_mask = adata.obs[cell_type_key] == target_cell_type
    other_mask = ~target_mask
    
       
    if isinstance(adata.X, np.ndarray):
        expr_matrix = adata.X
    else:
        expr_matrix = adata.X.toarray()     
    
    higher_expr_genes = []
    
    for gene in genes_to_test:
        gene_idx = adata.var_names.tolist().index(gene)
        
           
        target_mean = expr_matrix[target_mask, gene_idx].mean()
        other_mean = expr_matrix[other_mask, gene_idx].mean()
        
           
        if target_mean > other_mean:
            higher_expr_genes.append(gene)
    
    return higher_expr_genes

from typing import Dict, List, Tuple

def find_celltype_specific_genes(
    adata,
    cell_type_list,
    gene_list: List[str],
    cell_type_key: str = 'cell_type',
    X: np.ndarray = None,
    top_k: int = 100,
    **kwargs
) -> Dict[str, List[str]]:
       
    cell_types = adata.obs[cell_type_key].unique().tolist()
    
    result_dict = {}
    
    for cell_type in cell_types:
           
        diff_genes = find_diff_genes_one_vs_rest(
            X=X,
            cell_types=cell_type_list,
            gene_names=gene_list,
            target_cell_type=cell_type,
            top_k=top_k,
            **kwargs
        )
        
           
        higher_expr_genes = find_higher_expressed_genes(
            adata,
            target_cell_type=cell_type,
            gene_list=diff_genes,
            cell_type_key=cell_type_key,
            **kwargs
        )
        
        result_dict[cell_type] = higher_expr_genes
    
    return result_dict


