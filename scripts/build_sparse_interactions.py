"""Build and audit the Day-03 train-only sparse interaction representation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from recomx.evaluation.relevance import RelevancePolicy
from recomx.models.collaborative_protocol import DEFAULT_COLLABORATIVE_PROTOCOL
from recomx.models.sparse_interactions import build_sparse_interaction_matrix

TRAIN_PATH=Path("data/processed/ratings_train.parquet")
OUTPUT_DIR=Path("reports/baselines/collaborative/sparse")
def main():
    p=DEFAULT_COLLABORATIVE_PROTOCOL; p.validate()
    if not TRAIN_PATH.is_file(): raise FileNotFoundError(f"Missing train partition: {TRAIN_PATH}")
    print("RecomX Day-03 CP-02 Sparse Interaction Matrix"); print("="*72)
    print("validation=NOT READ"); print("locked_test=NOT READ")
    train=pd.read_parquet(TRAIN_PATH,columns=["userId","movieId","rating"])
    r=build_sparse_interaction_matrix(train,RelevancePolicy(minimum_rating=p.relevance_threshold))
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    sparse.save_npz(OUTPUT_DIR/"positive_user_item_csr.npz",r.matrix,compressed=True)
    np.save(OUTPUT_DIR/"user_ids.npy",r.user_ids,allow_pickle=False); np.save(OUTPUT_DIR/"item_ids.npy",r.item_ids,allow_pickle=False)
    possible=r.n_users*r.n_items
    sparse_bytes=r.matrix.data.nbytes+r.matrix.indices.nbytes+r.matrix.indptr.nbytes
    summary={"training_rows":int(len(train)),"positive_users":r.n_users,"positive_items":r.n_items,"positive_user_item_pairs":r.n_interactions,"possible_positive_matrix_cells":int(possible),"density":r.density,"sparsity":1-r.density,"matrix_format":"csr","dtype":str(r.matrix.dtype),"sparse_storage_bytes":int(sparse_bytes),"dense_float32_bytes_avoided":int(possible*np.dtype(np.float32).itemsize),"locked_test_used":False}
    with (OUTPUT_DIR/"sparse_matrix_summary.json").open("w",encoding="utf-8") as h: json.dump(summary,h,indent=2,sort_keys=True); h.write("\n")
    for k,v in summary.items(): print(f"{k}={v}")
    print("DAY-03 CP-02 SPARSE INTERACTION MATRIX: PASS")
if __name__=="__main__": main()
