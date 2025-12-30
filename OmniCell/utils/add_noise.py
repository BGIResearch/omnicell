import numpy as np
import scanpy as sc
from scipy.sparse import issparse, csr_matrix
from sklearn.preprocessing import LabelEncoder

def add_scRNA_noise(
    adata,
    pcr_bias=0.0,
    dropout_rate=0.0,
    batch_effect=0.0,
    random_seed=42,
    inplace=False
):
    np.random.seed(random_seed)
    
    if not inplace:
        adata = adata.copy()
    
    X = adata.X.toarray().astype(np.float64) if issparse(adata.X) else adata.X.astype(np.float64).copy()
    
       
    if pcr_bias > 0:
        bias = 1 + pcr_bias * np.random.lognormal(mean=0, sigma=0.3, size=X.shape[1])
        X = X * bias
        
        error_mask = np.random.random(X.shape) < (pcr_bias * 0.05)
        X[error_mask] = X[error_mask] * np.random.uniform(0.7, 1.3, size=np.sum(error_mask))
    
       
    if dropout_rate > 0:
           
        rows, cols = np.where(X > 0)
        non_zero_indices = list(zip(rows, cols))     
        
           
        num_to_drop = int(dropout_rate * len(non_zero_indices))
        
        if num_to_drop > 0:
               
            drop_indices = np.random.choice(
                len(non_zero_indices), 
                size=num_to_drop, 
                replace=False
            )
            
               
            for idx in drop_indices:
                r, c = non_zero_indices[idx]
                X[r, c] = 0
    
       
    if batch_effect > 0:
        if 'batch' not in adata.obs:
            adata.obs['batch'] = np.random.choice(2, size=X.shape[0])
        
        le = LabelEncoder()
        batch_ids = le.fit_transform(adata.obs['batch'])
        batch_factors = np.exp(
            batch_effect * np.random.normal(size=(len(le.classes_), X.shape[1]))
        )
        
        for i, bid in enumerate(batch_ids):
            X[i] = X[i] * batch_factors[bid]
    
    adata.X = csr_matrix(X) if issparse(adata.X) else X
    adata.uns['noise_params'] = {
        'pcr_bias': pcr_bias,
        'dropout_rate': dropout_rate,
        'batch_effect': batch_effect,
        'random_seed': random_seed
    }
    
    return adata if not inplace else None

