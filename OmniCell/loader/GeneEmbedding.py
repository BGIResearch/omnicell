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
from typing import Optional, List, Tuple, Dict
from torch.utils.data import DataLoader
import anndata as ad
import json
import os


class GeneFormer:
    """Single-cell gene embedding inference system"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        dtype: torch.dtype = torch.half,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_genes: int = 2000,
        threshold: float = 0.9,
        batch_key: Optional[str] = None,
        selected_genes: Optional[List[str]] = None,
        save_dir_path: str = None,
        vocab_path: str = None
    ):
        """
        Initialize inference system
        
        Parameters:
        checkpoint_dir: Directory containing config files and model checkpoints
        dtype: Inference precision (default: float16)
        batch_size: Inference batch size (default: 1)
        device: Inference device (default: cuda or cpu)
        """
        self.dtype = dtype
        self.checkpoint_dir = Path(checkpoint_dir)
        self.batch_size = batch_size
        self.device = device
        self.n_genes = n_genes
        self.token_len = n_genes + 2  # gene tokens + start/end special tokens
        self.batch_key = batch_key
        self.selected_genes = selected_genes
        
        # Load model and projection matrix
        loader = CheckpointLoader(self.checkpoint_dir)
        self.model, self.W = loader.load()
        self.d_model = loader.d_model
        self.model.eval()  # Set to inference mode
        
        # Precision and device handling
        if self.dtype is not None:
            self.model = self.model.to(dtype=self.dtype)
        self.model = self.model.to(device=device)
        self.W = self.W.to(device=device)
        self.projector = OrthogonalProjector(W=self.W, threshold=threshold)
        
        # Get projected dimension
        self.projected_dim = self.projector.basis.shape[1]
            
        print(f"✅ Model initialized, using device: {self.device}, number of genes: {n_genes}")
        print(f"🔧 Projected dimension: {self.projected_dim}")
        
        self.dataloader: Optional[DataLoader] = None
        self.total_cells: int = 0
        self.gene_names: List[str] = []  # Store gene names list
        
        # Load gene vocabulary
        self.vocab_path = vocab_path
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.vocab_dict = json.load(f)
            # Create reverse mapping: token id -> gene name
            self.id_to_gene = {v: k for k, v in self.vocab_dict.items()}
        
        # Memory mapped files
        self.save_dir_path = save_dir_path
        self.emb_mmap_path: Optional[str] = None
        self.val_mmap_path: Optional[str] = None
        self.emb_mmap: Optional[np.memmap] = None
        self.val_mmap: Optional[np.memmap] = None
    
    def _prepare_mmap_files(self):
        """Prepare memory-mapped files for long-term storage of gene embeddings and expression values"""
        # Ensure save directory exists
        os.makedirs(self.save_dir_path, exist_ok=True)
        
        # Set file paths
        self.gene_name_path = os.path.join(self.save_dir_path, "gene_name.json")
        self.emb_mmap_path = os.path.join(self.save_dir_path, "gene_emb.dat")
        self.val_mmap_path = os.path.join(self.save_dir_path, "gene_vals.dat")

        with open(self.gene_name_path, 'w') as f:
            json.dump(self.gene_names, f, indent=4)
        print(f"✅ Gene names saved to: {self.gene_name_path}")
        
        # Calculate tensor shapes
        emb_shape = (self.total_cells, self.n_genes, self.projected_dim)
        val_shape = (self.total_cells, self.n_genes)
        
        # Calculate disk usage and print
        emb_size_gb = np.prod(emb_shape) * 4 / (1024**3)  # float32 takes 4 bytes
        val_size_gb = np.prod(val_shape) * 4 / (1024**3)
        total_size_gb = emb_size_gb + val_size_gb
        
        print(f"💾 Estimated disk usage:")
        print(f"  Embedding file: {emb_size_gb:.2f} GB ({emb_shape})")
        print(f"  Expression value file: {val_size_gb:.2f} GB ({val_shape})")
        print(f"  Total: {total_size_gb:.2f} GB")
        print(f"  Save path: {self.save_dir_path}")
        
        self.emb_mmap = np.memmap(
            self.emb_mmap_path, 
            dtype='float32', 
            mode='w+', 
            shape=emb_shape
        )
        
        self.val_mmap = np.memmap(
            self.val_mmap_path,
            dtype='float32',
            mode='w+',
            shape=val_shape
        )

    def _prepare_data(self, adata: ad.AnnData):
        """Prepare data loader"""
        processor = DataProcessor(
            vocab_path=self.vocab_path,
            n_genes=self.n_genes,
            mode="sc",
            batch_key=self.batch_key
        )
        result = processor.process_single(adata=adata, selected_genes=self.selected_genes)
        dataset = RNADataset(data_dict=result, vocab_path=self.vocab_path)
        
        # Record total number of cells
        self.total_cells = len(dataset)
        print(f"📊 Loaded single-cell dataset: {self.total_cells} cells")
        
        # Get gene names list (save only once, same for all cells)
        test_data = dataset[0]
        gene_tokens = np.array(test_data['tokens'])  # Get token sequence of first sample
        # Extract gene names (skip start and end special tokens)
        self.gene_names = [self.id_to_gene.get(int(token), "UNKNOWN") for token in gene_tokens[1:-1]]
        print(f"🧬 Number of genes: {len(self.gene_names)}, examples: {self.gene_names[:3]}, ...")
        
        # Create memory mapped files
        self._prepare_mmap_files()
        
        # Initialize data loader
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False
        )

    def _process_batch(
        self, 
        outputs: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process batch output
        Returns:
          gene_embeddings: [batch_size, n_genes, projected_dim] gene-level embeddings
          gene_values: [batch_size, n_genes] gene expression values
        """
        # Validate sequence length
        batch_size, seq_len, d_model = outputs.shape
        if seq_len != self.token_len:
            raise ValueError(f"Sequence length {seq_len} does not match expected {self.token_len}")
        
        # Extract gene embeddings (skip start and end tokens)
        gene_embeddings = outputs[:, 1:-1, :]  # [batch, n_genes, d_model]
        gene_values = values[:, 1:-1]  # [batch, n_genes]
        
        return gene_embeddings, gene_values

    def _normalize_embeddings(
        self, 
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Gene-level L2 normalization"""
        # Calculate L2 norm along feature dimension
        norms = torch.norm(embeddings, dim=-1, keepdim=True)
        # Avoid division by zero
        norms = torch.clamp(norms, min=1e-7)
        return embeddings / norms

    @torch.no_grad()
    def _run_inference(self) -> None:
        """Execute main inference loop and write to memory-mapped files"""
        current_idx = 0  # Current write position
        
        for batch_idx, batch in enumerate(tqdm(self.dataloader, 
                          desc="Inference in progress",
                          unit="batch",
                          dynamic_ncols=True)):
            # Move data to device
            tokens = batch['tokens'].to(self.device)
            values = batch['original_values'].to(self.device)
            positions = batch['positions'].to(self.device)
            
            # Model forward pass
            outputs = self.model(gene=tokens, value=values, index=positions)
            
            # Process batch output (keep gene-level embeddings)
            gene_embeddings, gene_values = self._process_batch(outputs, values)
            
            # Gene-level normalization
            # gene_embeddings = self._normalize_embeddings(gene_embeddings).to(self.W.dtype)
            
            # Orthogonal projection (using projected dimension)
            gene_embeddings = self.projector.project(gene_embeddings.to(self.W.dtype))
            
            # Convert to CPU and numpy arrays
            batch_size = tokens.size(0)
            emb_cpu = gene_embeddings.float().cpu().numpy()
            val_cpu = gene_values.float().cpu().numpy()
            
            # Write to memory-mapped files
            self.emb_mmap[current_idx:current_idx + batch_size] = emb_cpu
            self.val_mmap[current_idx:current_idx + batch_size] = val_cpu
            
            # Update index
            current_idx += batch_size

    def infer(self, adata: ad.AnnData) -> Tuple[np.memmap, np.memmap, List[str]]:
        """
        Execute complete inference pipeline
        Returns:
          emb_mmap: Memory map of gene embeddings (cells × genes × projected_dim)
          val_mmap: Memory map of gene expression values (cells × genes)
          gene_names: List of gene names
        """
        # Prepare data
        self._prepare_data(adata)
        
        # Execute inference and write to memory-mapped files
        self._run_inference()
        
        print(f"✅ Inference completed, results stored in memory-mapped files")
        print(f"   Embedding file: {self.emb_mmap_path}")
        print(f"   Expression value file: {self.val_mmap_path}")
        
        # Ensure all data is written to disk
        self.emb_mmap.flush()
        self.val_mmap.flush()
        
        return self.emb_mmap, self.val_mmap, self.gene_names