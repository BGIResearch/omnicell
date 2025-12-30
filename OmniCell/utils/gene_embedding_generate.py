import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
import random


class DataLoader:
    """Data Loader"""

    def __init__(self,
                 gene_embeddings: np.ndarray,
                 cell_embeddings: np.ndarray,
                 ensemble_ids: np.ndarray,
                 adata: sc.AnnData):
        """
        Initialize data

        Args:
            gene_embeddings: Gene embeddings shape (N_cells, N_genes, embedding_dim)
            cell_embeddings: Cell embeddings shape (N_cells, cell_embedding_dim)
            ensemble_ids: Array of gene ensemble IDs shape (N_genes,)
            adata: AnnData object containing gene expression data
        """
        self.gene_embeddings = gene_embeddings
        self.cell_embeddings = cell_embeddings
        self.ensemble_ids = ensemble_ids
        self.adata = adata

        # Get data dimensions
        self.n_cells = gene_embeddings.shape[0]
        self.n_genes = gene_embeddings.shape[1]
        self.embedding_dim = gene_embeddings.shape[2]

        print(f"Data dimensions: {self.n_cells} cells × {self.n_genes} genes")
        print(f"Gene embedding dimension: {self.embedding_dim}")

    def load_cell_embeddings(self) -> torch.Tensor:
        """Load cell embeddings"""
        print(f"Cell embedding shape: {self.cell_embeddings.shape}")
        return torch.from_numpy(self.cell_embeddings).float()

    def load_gene_embeddings(self) -> torch.Tensor:
        """Load gene embeddings"""
        print(f"Gene embedding shape: {self.gene_embeddings.shape}")
        return torch.from_numpy(self.gene_embeddings).float()

    def build_expression_matrix(self) -> torch.Tensor:
        """Build gene expression matrix Y"""
        print("Building gene expression matrix...")

        n_cells = self.adata.n_obs
        print(f"AnnData dimensions: {n_cells} cells × {self.adata.n_vars} genes")

        # Ensure cell counts match
        if n_cells != self.n_cells:
            raise ValueError(f"Cell count mismatch: {self.n_cells} cells in embeddings, but {n_cells} in AnnData")

        # Convert expression matrix to CSR sparse format
        print("Converting expression matrix to CSR format...")
        if hasattr(self.adata.X, 'tocsr'):
            expression_matrix = self.adata.X.tocsr()
        else:
            expression_matrix = csr_matrix(self.adata.X)

        # Build gene name to index mapping
        var_name_to_idx = {name: idx for idx, name in enumerate(self.adata.var_names)}
        print(f"AnnData contains {len(var_name_to_idx)} genes")

        # Initialize expression matrix
        Y = np.zeros((self.n_cells, self.n_genes), dtype=np.float32)

        # Match genes
        valid_genes = []
        missing_genes = []

        print("Matching genes...")
        for i, ensemble_id in enumerate(self.ensemble_ids):
            if ensemble_id in var_name_to_idx:
                adata_idx = var_name_to_idx[ensemble_id]
                valid_genes.append((i, adata_idx))
            else:
                missing_genes.append((i, ensemble_id))

        print(f"Found {len(valid_genes)} valid genes")
        print(f"Missing {len(missing_genes)} genes")

        if valid_genes:
            # Extract all valid gene indices
            gene_indices, adata_indices = zip(*valid_genes)
            adata_indices = list(adata_indices)

            print("Batch extracting gene expression values...")
            # Extract all required gene columns at once
            selected_expression = expression_matrix[:, adata_indices]

            # Convert to dense matrix
            if hasattr(selected_expression, 'toarray'):
                selected_expression = selected_expression.toarray()

            # Place extracted data into the correct positions
            for i, gene_idx in enumerate(gene_indices):
                Y[:, gene_idx] = selected_expression[:, i]

        # Report missing genes
        if missing_genes:
            print("\nMissing genes:")
            for gene_idx, ensemble_id in missing_genes[:10]:
                print(f"  Position {gene_idx}: Ensemble ID: {ensemble_id}")
            if len(missing_genes) > 10:
                print(f"  ... and {len(missing_genes) - 10} more missing genes")

        # Compute nonzero statistics
        nonzero_count = np.count_nonzero(Y)
        total_values = Y.shape[0] * Y.shape[1]
        sparsity = 1 - (nonzero_count / total_values)
        print(f"\nExpression matrix statistics:")
        print(f"  Shape: {Y.shape}")
        print(f"  Nonzero values: {nonzero_count:,} / {total_values:,}")
        print(f"  Sparsity: {sparsity:.2%}")
        print(f"  Mean expression: {Y.mean():.4f}")
        print(f"  Max expression: {Y.max():.4f}")

        return torch.from_numpy(Y).float()



class CellGraphBuilder:
    """Build a cell similarity graph based on shared genes"""

    def __init__(self, k=15, alpha=0.95):
        """
        :param k: Number of nearest neighbors
        :param alpha: Decay parameter
        """
        self.k = k
        self.alpha = alpha

    def build_graph(self, cell_embeddings):
        """
        :param cell_embeddings: Cell embeddings shape (N, embedding)
        :return: Edge indices and edge weights
        """
        N = cell_embeddings.shape[0]
        device = cell_embeddings.device

        # Compute Euclidean distance matrix between cells
        distances = torch.cdist(cell_embeddings, cell_embeddings, p=2)
        distances_np = distances.cpu().numpy()

        # Find k nearest neighbors for each cell
        indices = np.zeros((N, self.k), dtype=np.int64)
        neighbor_distances = np.zeros((N, self.k))

        for i in range(N):
            # Get the distance from the i-th cell to all others
            dists = distances_np[i].copy()
            dists[i] = np.inf
            # Find the k nearest ones
            k_nearest = np.argpartition(dists, self.k)[:self.k]
            k_nearest = k_nearest[np.argsort(dists[k_nearest])]

            indices[i] = k_nearest
            neighbor_distances[i] = dists[k_nearest]

        # Compute distance to the k-th nearest neighbor
        epsilon_k = neighbor_distances[:, -1]

        # Build edges and weights
        edges = []
        weights = []

        for i in range(N):
            for j, neighbor_idx in enumerate(indices[i]):
                dist_ij = neighbor_distances[i, j]
                eps_i = epsilon_k[i]
                eps_j = epsilon_k[neighbor_idx]

                weight = 0.5 * np.exp(-(dist_ij / eps_i) ** self.alpha) + \
                         0.5 * np.exp(-(dist_ij / eps_j) ** self.alpha)
                if weight > 1e-4:
                    edges.append([i, neighbor_idx])
                    weights.append(weight)

        edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)
        edge_weight = torch.tensor(weights, dtype=torch.float32).to(device)
        return edge_index, edge_weight


class GPR_prop(MessagePassing):
    """
    Propagation class for GPR_GNN
    """

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like; note that in this case, alpha has to be an integer.
            # It means where the peak is when initializing GPR weights.
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Use specified Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'SGC':
            self.temp.data[self.alpha] = 1.0
        elif self.Init == 'PPR':
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
            self.temp.data[-1] = (1 - self.alpha) ** self.K
        elif self.Init == 'NPPR':
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha ** k
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'Random':
            bound = np.sqrt(3 / (self.K + 1))
            torch.nn.init.uniform_(self.temp, -bound, bound)
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'WS':
            self.temp.data = self.Gamma

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(nn.Module):
    """
    GPR-GNN (Generalized PageRank Graph Neural Network)
    """

    def __init__(self, input_dim, hidden_dim, K=10, alpha=0.1, dropout=0.2, dprate=0.0):
        """
        :param input_dim: Input dimension
        :param hidden_dim: Hidden layer dimension
        :param K: Number of propagation steps
        :param alpha: PPR restart probability
        :param dropout: Dropout rate
        :param dprate: Dropout rate before propagation
        """
        super(GPRGNN, self).__init__()

        # MLP feature transformation
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, input_dim)  # Output keeps input_dim

        # GPR propagation layer (initialized with PPR)
        self.prop = GPR_prop(K=K, alpha=alpha, Init='PPR')

        self.dropout = dropout
        self.dprate = dprate

    def forward(self, x, edge_index, edge_weight=None):
        """
        :param x: Node features (N, input_dim)
        :param edge_index: Edge indices
        :param edge_weight: Edge weights
        :return: Processed node features (N, input_dim)
        """
        # MLP feature transformation
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        # GPR propagation
        if self.dprate == 0.0:
            x = self.prop(x, edge_index, edge_weight)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop(x, edge_index, edge_weight)

        return x  # Return embeddings, not classifications


class GeneEmbeddingGNN(nn.Module):
    """GNN model for generating gene embeddings - uses modified GPR-GNN"""

    def __init__(self, input_dim, hidden_dim, K=10, alpha=0.1, dropout=0.2):
        """
        :param input_dim: Input dimension
        :param hidden_dim: Hidden layer dimension
        :param K: Number of GPR-GNN propagation steps
        :param alpha: PPR restart probability
        :param dropout: Dropout rate
        """
        super(GeneEmbeddingGNN, self).__init__()
        self.gpr_gnn = GPRGNN(input_dim, hidden_dim, K=K, alpha=alpha, dropout=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        """
        :param x: Node features (N, input_dim)
        :param edge_index: Edge indices
        :param edge_weight: Edge weights
        :return: Processed node features (N, input_dim)
        """
        return self.gpr_gnn(x, edge_index, edge_weight)


class GeneEmbeddingModel(nn.Module):
    """Complete gene embedding model"""

    def __init__(self, input_dim, hidden_dim=128, K=10,
                 k_neighbors=15, alpha_graph=0.95, alpha_gpr=0.1, dropout=0.1, n_cells=None):
        """
        :param input_dim: Input embedding dimension d
        :param hidden_dim: GNN hidden layer dimension
        :param K: Number of GPR-GNN propagation steps
        :param k_neighbors: Number of nearest neighbors
        :param alpha_graph: Decay parameter for graph construction
        :param alpha_gpr: Restart probability for GPR
        :param dropout: Dropout rate
        :param n_cells: Number of cells
        """
        super(GeneEmbeddingModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_cells = n_cells

        # Graph builder
        self.graph_builder = CellGraphBuilder(k=k_neighbors, alpha=alpha_graph)

        # GPR-GNN (outputs input_dim directly)
        self.gnn = GeneEmbeddingGNN(input_dim, hidden_dim, K=K, alpha=alpha_gpr, dropout=dropout)

        # Linear layer to predict gene expression (from input_dim directly)
        if n_cells is not None:
            self.expression_predictor = nn.Linear(input_dim, n_cells)
        else:
            self.expression_predictor = None

    def forward_single_gene(self, gene_embeddings, edge_index, edge_weight):
        """
        Process a single gene

        :param gene_embeddings: Embeddings of a single gene across all cells (N, input_dim)
        :param edge_index: Edge indices
        :param edge_weight: Edge weights
        :return: (Gene embedding, predicted expression)
        """
        # Process through GNN (outputs input_dim directly)
        gnn_output = self.gnn(gene_embeddings, edge_index, edge_weight)  # (N, input_dim)

        # Average across cells to get gene representation
        gene_representation = gnn_output.mean(dim=0)  # (input_dim,)

        # Predict expression directly
        predicted_expression = self.expression_predictor(gene_representation)  # (n_cells,)

        return gene_representation, predicted_expression

    def forward(self, gene_data_batch, cell_embeddings, gene_indices):
        """
        Process multiple genes in batch

        :param gene_data_batch: Batch gene data (batch_size, N, input_dim)
        :param cell_embeddings: Cell embeddings (N, cell_embedding_dim)
        :param gene_indices: List of gene indices
        :return: (Gene embeddings, predicted expressions)
        """
        batch_size = gene_data_batch.shape[0]
        device = gene_data_batch.device

        # Build cell graph
        edge_index, edge_weight = self.graph_builder.build_graph(cell_embeddings)

        # Initialize outputs
        gene_embeddings = torch.zeros(batch_size, self.input_dim, device=device)
        predicted_expressions = torch.zeros(batch_size, self.n_cells, device=device)

        # Process each gene
        for i in range(batch_size):
            gene_emb, pred_expr = self.forward_single_gene(
                gene_data_batch[i], edge_index, edge_weight
            )
            gene_embeddings[i] = gene_emb
            predicted_expressions[i] = pred_expr

        return gene_embeddings, predicted_expressions



class Trainer:
    """Trainer"""

    def __init__(self, model, data_loader, learning_rate=1e-3, device='cuda'):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Load data
        print("Loading data...")
        self.cell_embeddings = data_loader.load_cell_embeddings().to(device)
        self.gene_embeddings = data_loader.load_gene_embeddings().to(device)
        self.Y = data_loader.build_expression_matrix().to(device)

    def train_epoch(self, batch_size=10):
        """Train one epoch"""
        self.model.train()
        n_genes = self.data_loader.n_genes
        n_batches = (n_genes + batch_size - 1) // batch_size

        # Shuffle gene indices at the start of each epoch
        gene_indices = list(range(n_genes))
        random.shuffle(gene_indices)

        total_loss = 0
        pbar = tqdm(range(n_batches), desc="Training")

        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_genes)
            # Use shuffled gene indices
            batch_gene_indices = gene_indices[start_idx:end_idx]
            actual_batch_size = len(batch_gene_indices)

            # Extract batch data from gene embeddings
            gene_batch = self.gene_embeddings[:, batch_gene_indices, :].permute(1, 0, 2)  # (batch_size, N, d)
            y_batch = self.Y[:, batch_gene_indices].t()  # (batch_size, n_cells)

            # Forward pass
            self.optimizer.zero_grad()
            gene_embeddings, predicted_expressions = self.model(
                gene_batch, self.cell_embeddings, batch_gene_indices
            )

            # Compute loss
            loss = self.criterion(predicted_expressions, y_batch)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / n_batches

    def extract_all_gene_embeddings(self, batch_size=50):
        """Extract embeddings for all genes"""
        self.model.eval()
        n_genes = self.data_loader.n_genes
        n_batches = (n_genes + batch_size - 1) // batch_size

        all_embeddings = []

        with torch.no_grad():
            for batch_idx in tqdm(range(n_batches), desc="Extracting embeddings"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_genes)
                batch_gene_indices = list(range(start_idx, end_idx))

                # Extract batch data from gene embeddings
                gene_batch = self.gene_embeddings[:, batch_gene_indices, :].permute(1, 0, 2)

                # Get embeddings
                gene_embeddings, _ = self.model(
                    gene_batch, self.cell_embeddings, batch_gene_indices
                )

                all_embeddings.append(gene_embeddings.cpu())

        # Combine all embeddings
        final_embeddings = torch.cat(all_embeddings, dim=0)
        return final_embeddings


def train_gene_embedding_model(gene_embeddings: np.ndarray,
                               cell_embeddings: np.ndarray,
                               ensemble_ids: np.ndarray,
                               adata: sc.AnnData,
                               n_epochs: int = 10,
                               batch_size: int = 20,
                               learning_rate: float = 1e-3,
                               hidden_dim: int = 128,
                               K: int = 10,
                               k_neighbors: int = 15,
                               alpha_graph: float = 0.95,
                               alpha_gpr: float = 0.1,
                               dropout: float = 0.1,
                               device: str = None) -> np.ndarray:
    """
    Train the gene embedding model

    Args:
        gene_embeddings: Gene embeddings shape (N_cells, N_genes, embedding_dim)
        cell_embeddings: Cell embeddings shape (N_cells, cell_embedding_dim)
        ensemble_ids: Array of gene ensemble IDs shape (N_genes,)
        adata: AnnData object containing gene expression data
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_dim: Hidden layer dimension
        K: Number of GPR-GNN propagation steps
        k_neighbors: Number of nearest neighbors
        alpha_graph: Decay parameter for graph construction
        alpha_gpr: Restart probability for GPR
        dropout: Dropout rate
        device: Device ('cuda' or 'cpu')

    Returns:
        gene_embeddings: Trained gene embeddings shape (N_genes, embedding_dim)
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize data loader
    data_loader = DataLoader(
        gene_embeddings=gene_embeddings,
        cell_embeddings=cell_embeddings,
        ensemble_ids=ensemble_ids,
        adata=adata
    )

    # Get data dimensions
    n_cells = data_loader.n_cells
    input_dim = data_loader.embedding_dim

    # Create model
    model = GeneEmbeddingModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        K=K,
        k_neighbors=k_neighbors,
        alpha_graph=alpha_graph,
        alpha_gpr=alpha_gpr,
        dropout=dropout,
        n_cells=n_cells
    )

    # Create trainer
    trainer = Trainer(model, data_loader, learning_rate=learning_rate, device=device)

    # Train model
    for epoch in range(n_epochs):
        avg_loss = trainer.train_epoch(batch_size=batch_size)
        print(f"Epoch {epoch + 1}/{n_epochs}, Average Loss: {avg_loss:.4f}")

    # Extract all gene embeddings
    print("\nExtracting final gene embeddings...")
    gene_embeddings_result = trainer.extract_all_gene_embeddings(batch_size=50)
    print(f"Gene embedding shape: {gene_embeddings_result.shape}")

    # Convert to numpy array and return
    return gene_embeddings_result.numpy()
