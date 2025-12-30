import scanpy as sc
import numpy as np
from typing import Optional, Dict, Union

def simple_proportional_sampling(
    adata: sc.AnnData,
    ratio: Union[float, Dict[str, float]],
    sample_key: str = "cell_type",
    random_state: Optional[int] = 42,
    min_samples: int = 1,
) -> sc.AnnData:
       
    if sample_key not in adata.obs:
        raise ValueError(f"'{sample_key}' not found in adata.obs")
    
    if isinstance(ratio, dict):
        missing_groups = set(adata.obs[sample_key].unique()) - set(ratio.keys())
        if missing_groups:
            raise ValueError(f"Missing ratios for groups: {missing_groups}")
    elif not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1] when a single value")

       
    rng = np.random.RandomState(random_state)

       
    sampled_indices = []
    groups = adata.obs[sample_key].unique()
    
    for group in groups:
        group_indices = adata.obs[adata.obs[sample_key] == group].index
        group_size = len(group_indices)
        
           
        if isinstance(ratio, dict):
            n_samples = max(min_samples, int(group_size * ratio[group]))
        else:
            n_samples = max(min_samples, int(group_size * ratio))
        
           
        n_samples = min(n_samples, group_size)
        
           
        sampled_indices.extend(
            group_indices[rng.choice(len(group_indices), size=n_samples, replace=False)]
        )

    return adata[sampled_indices, :].copy()