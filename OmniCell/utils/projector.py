import torch

class OrthogonalProjector:
    def __init__(self, W, threshold=0.95, device=None):
        self.W = W
        self.device = device or W.device
        self.dtype = W.dtype
        self.threshold = threshold
        self.sym_W = (W + W.t()) / 2
        self._compute_eigen()
        self._select_basis()
    
    def _compute_eigen(self):
        eigenvalues, eigenvectors = torch.linalg.eigh(self.sym_W)
        abs_eigen = torch.abs(eigenvalues)
        sorted_idx = torch.argsort(abs_eigen, descending=True)
        self.eigenvalues = eigenvalues[sorted_idx]
        self.eigenvectors = eigenvectors[:, sorted_idx]
    
    def _select_basis(self):
        variances = torch.abs(self.eigenvalues)
        total_variance = torch.sum(variances)
        explained_ratios = variances / total_variance
        cumulative_ratios = torch.cumsum(explained_ratios, dim=0)
        n_components = max(1, torch.sum(cumulative_ratios < self.threshold).item() + 1)
        self.basis = self.eigenvectors[:, :n_components]
        self.basis = self.basis / torch.norm(self.basis, dim=0)
        inner_product = self.basis.t() @ self.basis
        self.orthonormal = torch.allclose(
            inner_product, 
            torch.eye(n_components, device=self.device, dtype=self.dtype),
            atol=1e-5
        )
    
    def project(self, embedding):
        if embedding.size(-1) != self.basis.size(0):
            raise ValueError(
            f"Embedding dimension {embedding.size(-1)} does not match the base dimension {self.basis.size(0)}"
            )
        return embedding @ self.basis