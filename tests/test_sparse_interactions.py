"""Tests for the Day-03 sparse interaction representation."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from recomx.models.sparse_interactions import build_sparse_interaction_matrix

@pytest.fixture
def train():
    return pd.DataFrame({"userId":[20,10,20,10,30,30,20],"movieId":[300,100,100,300,200,400,300],"rating":[5.,4.5,3.,4.,2.,5.,4.5]})

def test_builds_csr_float32_binary_matrix(train):
    r=build_sparse_interaction_matrix(train)
    assert sparse.isspmatrix_csr(r.matrix) and r.matrix.dtype==np.float32
    assert np.all(r.matrix.data==1.0)

def test_only_relevant_feedback_is_encoded(train):
    r=build_sparse_interaction_matrix(train)
    assert r.n_interactions==4
    assert set(r.positive_items_for_user(20))=={300}
    assert set(r.positive_items_for_user(30))=={400}

def test_mappings_are_deterministic(train):
    a=build_sparse_interaction_matrix(train); b=build_sparse_interaction_matrix(train.sample(frac=1,random_state=7))
    assert a.user_ids.tolist()==[10,20,30] and a.item_ids.tolist()==[100,300,400]
    assert a.user_to_index==b.user_to_index and a.item_to_index==b.item_to_index
    assert (a.matrix!=b.matrix).nnz==0

def test_unknown_user_has_empty_history(train):
    assert build_sparse_interaction_matrix(train).positive_items_for_user(999).size==0

def test_density(train):
    assert build_sparse_interaction_matrix(train).density==pytest.approx(4/9)

def test_missing_columns_fail():
    with pytest.raises(ValueError,match="missing required columns"):
        build_sparse_interaction_matrix(pd.DataFrame({"userId":[1],"movieId":[2]}))

def test_empty_fails():
    with pytest.raises(ValueError,match="cannot be empty"):
        build_sparse_interaction_matrix(pd.DataFrame(columns=["userId","movieId","rating"]))

def test_no_relevant_fails():
    with pytest.raises(ValueError,match="No relevant"):
        build_sparse_interaction_matrix(pd.DataFrame({"userId":[1],"movieId":[2],"rating":[2.]}))
