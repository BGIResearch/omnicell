import numpy as np
import scanpy as sc
from sklearn.neighbors import BallTree
from tqdm import tqdm
import scipy.sparse as sp
import json
import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix

class DataProcessor:

    def __init__(self, vocab_path, mode='sc', n_genes=2000, 
                 n_neighbors=15, batch_key=None):
        
        self.vocab = self._load_vocab(vocab_path)
        self.mode = mode
        self.n_genes = n_genes
        self.n_neighbors = n_neighbors
        self.spatial_tree = None    
        self.batch_key = batch_key

    def _load_vocab(self, path):
        with open(path) as f:
            return json.load(f)

    def process_single(self, adata, selected_genes=None):
          
        adata = self._preprocess_data(adata)
        if adata is None:
            raise ValueError(
                "Missing required AnnData input. "
                "Please provide a valid AnnData object containing gene expression data."
            )
        
          
        if selected_genes is not None:
              
            gene_names = [g for g in selected_genes if g in self.vocab]
            if len(gene_names) < self.n_genes:
                print(f"Warning: Only {len(gene_names)} genes found in vocabulary")
        else:
            gene_names = self._select_hvg_genes(adata)
        
          
        gene_ids = [self.vocab[g] for g in gene_names]
        
          
        if self.mode == 'sc':
            return self._process_sc(adata, gene_names, gene_ids)
        elif self.mode == 'spatial':
            return self._process_spatial(adata, gene_names, gene_ids)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _preprocess_data(self, adata):
        _, unique_index = np.unique(adata.var_names, return_index=True)
        adata = adata[:, unique_index].copy()
        
          
        valid_gene_mask = np.array([g in self.vocab for g in adata.var_names])
        valid_gene_indices = np.where(valid_gene_mask)[0]
        
        if len(valid_gene_indices) == 0:
            raise ValueError("No gene names in ref gene, please check the adata.var_names")
        
        adata = adata[:, valid_gene_indices]
        print(f"Number of matched genes: {len(valid_gene_indices)}/{len(valid_gene_mask)}")
        
          
        if np.max(adata.X) <= 20: 
            print("Warning: Data appears to be already normalized. This model requires raw expression counts.")
        
        return adata

    def _select_hvg_genes(self, adata):
          
        if self.mode == 'spatial' or self.batch_key is None:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=self.n_genes,
                flavor="seurat_v3"
            )
        else:    
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=self.n_genes,
                batch_key=self.batch_key,
                flavor="seurat_v3"
            )
        
          
        return adata.var_names[adata.var["highly_variable"]].tolist()

    def _process_sc(self, adata, gene_names, gene_ids):
          
        expr_matrix = adata[:, gene_names].X.toarray()
        
          
        result_dict = {}
        for i in tqdm(range(expr_matrix.shape[0]), desc="Processing cells"):
            result_dict[str(i)] = {
                "gene_ids": gene_ids,
                "expressions": expr_matrix[i].astype(float).tolist()
            }
        
        return result_dict

    def _process_spatial(self, adata, gene_names, gene_ids):
          
        self.spatial_tree = BallTree(
            adata.obs[['x', 'y']].values.astype(np.float32),
            metric='euclidean'
        )
        
          
        expr_matrix = adata[:, gene_names].X.toarray()
        
          
        result_dict = {}
        for i in tqdm(range(adata.n_obs), desc="Processing cells"):
              
            expr_values = expr_matrix[i]
            
              
            neighbors = self._get_neighbors(adata, i)
            
              
            result_dict[str(i)] = {
                "gene_ids": gene_ids,
                "expressions": expr_values.astype(float).tolist(),
                "neighbors": neighbors,
                "x": float(adata.obs.iloc[i]['x']),
                "y": float(adata.obs.iloc[i]['y'])
            }
        
        return result_dict

    def _get_neighbors(self, adata, cell_idx):
        if self.spatial_tree is None:
            raise RuntimeError("Spatial tree not initialized")
        
          
        distances, indices = self.spatial_tree.query(
            [adata.obs[["x", "y"]].values[cell_idx, :]], 
            k=self.n_neighbors + 1
        )
        
          
        return [str(int(i)) for i in indices[0][1:]]


      
      
      
      
      
      
      