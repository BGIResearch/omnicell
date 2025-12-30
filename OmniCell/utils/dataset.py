import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Union

class RNADataset(Dataset):
    def __init__(
        self,
        data_dict: Dict[str, dict],
        vocab_path: str,
        use_smooth_rank: bool = True,
        smooth_rank_range: Tuple[float, float] = (0, 5.0),
        rna_start: str = "<RNA_START>",
        rna_end: str = "<RNA_END>",
        n_neighbors: int = 9,
        include_batch_info: bool = False    
    ):
        self.data_dict = data_dict
        self.total_cells = len(data_dict)
        self.use_smooth_rank = use_smooth_rank
        self.smooth_rank_range = smooth_rank_range
        self.n_neighbors = n_neighbors
        self.include_batch_info = include_batch_info    

          
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        
          
        self.special_tokens = {
            'start': self._validate_token(rna_start),
            'end': self._validate_token(rna_end)
        }

          
        sample0 = next(iter(data_dict.values()))
        self.is_spatial = all(key in sample0 for key in ['neighbors', 'x', 'y'])
        
          
        if self.include_batch_info:
            self.has_batch_info = 'batch_indices' in sample0
            if self.has_batch_info:
                print(f"🔧 The dataset contains batch information")
            else:
                print("⚠️ Warning: The request includes batch information, but none was found in the data")
        else:
            self.has_batch_info = False

    def _validate_token(self, token: str) -> int:
        if token not in self.vocab:
            raise ValueError(f"Special token '{token}' does not exist in the vocabulary")
        return self.vocab[token]

    def __len__(self):
        return self.total_cells

    def _smooth_rank(self, expr: np.ndarray) -> List:
        unique_values = np.unique(expr)
        if 0 not in unique_values:
            unique_values = np.append(unique_values, 0)
        unique_sorted = np.sort(unique_values)
        n = len(unique_sorted)
        
        if n == 0:
            return expr
        
        left, right = self.smooth_rank_range
        if n == 1:
            scaled_ranks = np.full(n, left, dtype=np.float32)
        else:
            scaled_ranks = left + (np.arange(n) / (n - 1)) * (right - left)
            scaled_ranks = scaled_ranks.astype(np.float32)
        
        indices = np.searchsorted(unique_sorted, expr, side='left')
        return scaled_ranks[indices].tolist()

    def _process_cell(
        self,
        gene_ids: List[int],
        expressions: List[float],
    ) -> Tuple[List[int], List[float], List[int]]:
          
        if self.use_smooth_rank:
            expressions = self._smooth_rank(np.array(expressions))
        
          
        tokens = [self.special_tokens['start']] + gene_ids + [self.special_tokens['end']]
        original_values = [0] + expressions + [0]
          
        mask = [0] + [1 if x > 0 else 0 for x in expressions] + [0]
          
        
        return tokens, original_values, mask

    def __getitem__(self, index: int) -> Dict[str, Union[List, float, str, int]]:
        cell_data = self.data_dict[str(index)]
        
          
        batch_index = cell_data.get('batch_indices', 0) if self.include_batch_info else 0
        
        if self.is_spatial:
              
            center_cell = cell_data
            
              
            center_tokens, center_original, center_mask = self._process_cell(
                center_cell['gene_ids'],
                center_cell['expressions']
            )
            
              
            neighbor_indices = center_cell['neighbors'][:self.n_neighbors]
            neighbors = [self.data_dict[str(idx)] for idx in neighbor_indices 
                        if str(idx) in self.data_dict]
            
              
            all_tokens = center_tokens
            all_original = center_original
            all_positions = [[center_cell['x'], center_cell['y']]] * len(center_tokens)
            nonzero_mask = center_mask
            
              
            if neighbors:
                for i, neighbor in enumerate(neighbors):
                    n_tokens, n_original, n_mask = self._process_cell(
                        neighbor['gene_ids'],
                        neighbor['expressions']
                    )
                    all_tokens.extend(n_tokens)
                    all_original.extend(n_original)
                    all_positions.extend([[neighbor['x'], neighbor['y']]] * len(n_tokens))
                    nonzero_mask.extend(n_mask)
            
              
            result = {
                'tokens': all_tokens,
                'original_values': all_original,
                'pos': all_positions,
                'nonzero_mask': nonzero_mask,
                'batch_indices': batch_index    
            }
        
        else:
              
            tokens, original, mask = self._process_cell(
                cell_data['gene_ids'],
                cell_data['expressions']
            )
            
              
            result = {
                'tokens': tokens,
                'original_values': original,
                'pos': [[0, 0]] * len(tokens),    
                'nonzero_mask': mask,
                'batch_indices': batch_index    
            }
                
        return result

def collate_fn(batch):
      
    collated = {
        'tokens': torch.tensor([item['tokens'] for item in batch], dtype=torch.long),
        'original_values': torch.tensor([item['original_values'] for item in batch], dtype=torch.float),
        'positions': torch.tensor([item['pos'] for item in batch], dtype=torch.float),
        'nonzero_mask': torch.tensor([item["nonzero_mask"] for item in batch], dtype=torch.float)
    }
    
      
    if 'batch_indices' in batch[0]:
        collated['batch_indices'] = torch.tensor(
            [item['batch_indices'] for item in batch], 
            dtype=torch.long
        )
    
    return collated
