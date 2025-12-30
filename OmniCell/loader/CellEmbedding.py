import sys
from pathlib import Path
current_file_path = Path(__file__).absolute() if "__file__" in globals() else Path.cwd()
current_dir = current_file_path.parent if "__file__" in globals() else current_file_path
parent_parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_parent_dir))
from OmniCell.utils.checkpoint_loader import CheckpointLoader
from OmniCell.utils.h5ad_to_dict import DataProcessor
from OmniCell.utils.dataset import RNADataset, collate_fn
from OmniCell.utils.projector import OrthogonalProjector
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union
from torch.utils.data import DataLoader
import anndata as ad
import json


class CellFormer:
    def __init__(
        self,
        checkpoint_dir: str,
        dtype: torch.dtype = torch.half,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_genes: int = 2000,
        mode: str = "sc",
        n_neighbors: Optional[int] = None,
        batch_key: Optional[str] = None,
        selected_genes: Optional[List[str]] = None,
        threshold: float = 0.9,
        use_smooth_rank: bool = True,
        use_project: bool = True,
        vocab_path = None
    ):
        self.dtype = dtype
        self.checkpoint_dir = Path(checkpoint_dir)
        self.batch_size = batch_size
        self.device = device
        self.n_genes = n_genes
        self.mode = mode
        self.n_neighbors = n_neighbors
        self.batch_key = batch_key
        self.selected_genes = selected_genes
        self.use_project = use_project
          
        if self.mode == "spatial":
            if self.n_neighbors is None:
                raise ValueError("Space pattern must specify n_neighbors")
            self.cell_token_len = self.n_genes + 2  
        
        loader = CheckpointLoader(self.checkpoint_dir)
        self.model, self.W = loader.load()
        self.d_model = loader.d_model
        self.model.eval()    
        
          
        if self.dtype is not None:
            self.model = self.model.to(dtype=self.dtype)
            self.W = self.W
        self.model = self.model.to(device=device)
        self.W = self.W.to(device=device)
        self.projector = OrthogonalProjector(W = self.W,
                                        threshold = threshold)
        self.projected_dim = self.projector.basis.shape[1]   
        print(f"✅ Model initialization complete, using device: {self.device}, mode: {mode}")
        print(f"🔧 Projected dimension: {self.projected_dim}")
        self.dataloader: Optional[DataLoader] = None
        self.total_cell: int = 0
        self.total_spots: int = 0    
        self.use_smooth_rank = use_smooth_rank
        
        self.vocab_path = vocab_path
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.vocab_dict = json.load(f)
    
    def _prepare_data(self, adata: ad.AnnData):
        processor = DataProcessor(
            vocab_path=self.vocab_path,
            mode=self.mode,
            n_genes=self.n_genes,
            n_neighbors=self.n_neighbors,
            batch_key=self.batch_key
        )
        result = processor.process_single(adata=adata, selected_genes=self.selected_genes)
        dataset = RNADataset(data_dict=result, vocab_path=self.vocab_path, n_neighbors=self.n_neighbors, use_smooth_rank=self.use_smooth_rank)
        
          
        if self.mode == "spatial":
            self.total_spots = len(dataset)    
            print(f"📊 Spatial dataset: {self.total_spots} spots")
        else:
            self.total_cell = len(dataset)    
            print(f"📊 Single Cell dataset: {self.total_cell} cells")
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False
        )

    @torch.no_grad()
    def _run_inference(self) -> Tuple[torch.Tensor, int]:
        all_embeddings = []    
        all_values = []
        for batch in tqdm(self.dataloader, 
                          desc="Inferencing",
                          unit="batch",
                          dynamic_ncols=True):
              
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            
              
            outputs = self.model(
                gene=inputs['tokens'],
                value=inputs['original_values'],
                index=inputs['positions']
            )
            
              
            if self.mode == "spatial":
                cell_embeddings, values = self._process_spatial_batch(
                    outputs=outputs,
                    nonzero_mask=inputs["nonzero_mask"],
                    cell_token_len=self.cell_token_len,
                    n_neighbors=self.n_neighbors,
                    values = inputs['original_values']
                )   
            else:
                cell_embeddings, values = self._process_batch_cell(
                    outputs=outputs,
                    nonzero_mask=inputs["nonzero_mask"],
                    values = inputs['original_values']
                )   
            
            cell_embeddings = self._normalize_embeddings(cell_embeddings)
            if self.use_project:
                cell_embeddings = self.projector.project(cell_embeddings)
              
            all_embeddings.append(cell_embeddings.float().cpu().numpy())
            all_values.append(values.float().cpu().numpy())
        
          
        full_embeddings = np.concatenate(all_embeddings, axis=0)
        full_values = np.concatenate(all_values, axis=0)
        return full_embeddings, full_values
    
    def _process_spatial_batch(
        self,
        outputs: torch.Tensor,
        nonzero_mask: torch.Tensor,
        cell_token_len: int,
        n_neighbors: int,
        values: torch.Tensor
    ) -> torch.Tensor:

        batch_size, seq_len, d_model = outputs.shape
        total_cells_per_sample = 1 + n_neighbors
        
          
        expected_len = total_cells_per_sample * cell_token_len
        if seq_len != expected_len:
            raise ValueError(
                f"Sequence length {seq_len} does not match the expected value {expected_len} "
                f"(total cells: {total_cells_per_sample}, cell token length: {cell_token_len})"
            )

        
          
        outputs = outputs.view(batch_size, total_cells_per_sample, cell_token_len, d_model)
        nonzero_mask = nonzero_mask.view(batch_size, total_cells_per_sample, cell_token_len)
        values = values.view(batch_size, total_cells_per_sample, cell_token_len)
        values = values[:, 0, 1:-1]
        
        
          
        cell_embeddings = self._non_zero_mean(outputs, nonzero_mask)
        
        return cell_embeddings, values   
    
    def _non_zero_mean(
        self, 
        outputs: torch.Tensor, 
        nonzero_mask: torch.Tensor
    ) -> torch.Tensor:
        
        assert outputs.shape[:-1] == nonzero_mask.shape, \
        f"Mask shape {nonzero_mask.shape} does not match output shape {outputs.shape[:-1]}"

        
          
        mask_expanded = nonzero_mask.unsqueeze(-1)
        
          
        weighted_outputs = outputs * mask_expanded
        sum_features = weighted_outputs.sum(dim=-2)    
        
          
        nonzero_counts = mask_expanded.sum(dim=-2).clamp(min=1)
        
          
        return sum_features / nonzero_counts
    
    def _process_batch_cell(
        self, 
        outputs: torch.Tensor, 
        nonzero_mask: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
          
        return self._non_zero_mean(outputs, nonzero_mask), values[:,1:-1:]
    
    def _normalize_embeddings(
        self, 
        embeddings: torch.Tensor
    ) -> torch.Tensor:
   
        norms = torch.norm(embeddings, dim=-1, keepdim=True)
    
          
        return embeddings / torch.clamp(norms, min=1e-7)
    
    def _reshape_spatial_embeddings(
        self, 
        embeddings: torch.Tensor
    ) -> torch.Tensor:
          
        total_cells = embeddings.shape[0]
        expected_cells = self.total_spots * (1 + self.n_neighbors)
        
        if total_cells != expected_cells:
           raise ValueError(
    f"Number of embeddings {total_cells} does not match the expected value {expected_cells} "
    f"(number of spots: {self.total_spots}, number of neighbors: {self.n_neighbors})"
)

        
          
        return embeddings.view(self.total_spots, 1 + self.n_neighbors, self.d_model)
    
    @torch.no_grad()
    def infer(self, adata: ad.AnnData) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
          
        self._prepare_data(adata)
        
          
        embeddings, values = self._run_inference()
          
          
        print(f"✅ Inference complete, total embedding dimension: {embeddings.shape}, expression value dimension: {values.shape}")
            
        return embeddings, values
