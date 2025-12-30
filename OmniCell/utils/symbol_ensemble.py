import os
import pandas as pd

def create_gene_mapping_dicts():   
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        os.path.dirname(current_dir),     
        "vocab",     
        "new_genes_homo_sapiens.csv"     
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Gene mapping file not found: {file_path}")
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        raise ValueError(f"Failed to read gene mapping file: {str(e)}")   
    ensembl_to_symbol = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    symbol_to_ensemble = dict(zip(df.iloc[:, 1], df.iloc[:, 0]))
    
    return ensembl_to_symbol, symbol_to_ensemble


import scanpy as sc
import pandas as pd
from typing import Union

def convert_var_names_to_ensembl(
    adata: sc.AnnData,
    gene_mapping_dicts: tuple = None,
    symbol_col: str = "symbol",
    inplace: bool = True
) -> Union[sc.AnnData, None]:
    if gene_mapping_dicts is None:
        ensembl_to_symbol, symbol_to_ensemble = create_gene_mapping_dicts()
    
       
    adata.var[symbol_col] = adata.var_names
    
       
    total_genes = len(adata.var_names)
    converted = 0
    failed_genes = []
    
       
    new_var_names = []
    for gene in adata.var_names:
        if gene in symbol_to_ensemble:
            new_var_names.append(symbol_to_ensemble[gene])
            converted += 1
        else:
            new_var_names.append(gene)
            failed_genes.append(gene)
    
       
    conversion_rate = converted / total_genes * 100
    print(f"Gene ID conversion statistics:")
    print(f"  Successfully converted: {converted}/{total_genes} ({conversion_rate:.2f}%)")
    print(f"  Unconverted gene examples: {failed_genes[:5]}{'...' if len(failed_genes) > 5 else ''}")

    
       
    if inplace:
        adata.var_names = new_var_names
        return None
    else:
        new_adata = adata.copy()
        new_adata.var_names = new_var_names
        return new_adata

