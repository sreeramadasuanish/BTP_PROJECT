"""Unit tests for partitioning strategies, coverage, and duplicate detection."""

import numpy as np
import pytest
from src.data.partition.uniform import partition_uniform
from src.data.partition.clustered import partition_clustered
from src.data.partition.category import partition_category
from src.data.partition.dirichlet import partition_dirichlet
from src.data.partition.label_skew import partition_label_skew
from src.data.partition import create_partitions


def test_uniform_partition_coverage_and_uniqueness():
    n_samples = 1000
    n_clients = 20
    parts = partition_uniform(n_samples, n_clients, seed=42)

    assert len(parts) == n_clients
    all_indices = [idx for c_inds in parts.values() for idx in c_inds]

    assert len(all_indices) == n_samples
    assert len(set(all_indices)) == n_samples, "Duplicate indices detected in uniform partition"


def test_clustered_partition_no_duplicates():
    targets = np.random.randint(0, 10, size=1200)
    n_clients = 10
    parts = partition_clustered(targets, n_clients, shards_per_client=2, seed=42)

    all_indices = [idx for c_inds in parts.values() for idx in c_inds]
    assert len(all_indices) == len(targets)
    assert len(set(all_indices)) == len(targets)


def test_dirichlet_partition_no_duplicates():
    targets = np.random.randint(0, 10, size=1000)
    n_clients = 15
    parts = partition_dirichlet(targets, n_clients, alpha=0.1, seed=42)

    all_indices = [idx for c_inds in parts.values() for idx in c_inds]
    assert len(all_indices) == len(targets)
    assert len(set(all_indices)) == len(targets)
    # Every client should have samples
    for cid, inds in parts.items():
        assert len(inds) > 0


def test_label_skew_partition_no_duplicates():
    targets = np.random.randint(0, 2, size=800)
    n_clients = 10
    parts = partition_label_skew(targets, n_clients, skew_ratio=0.9, seed=42)

    all_indices = [idx for c_inds in parts.values() for idx in c_inds]
    assert len(all_indices) == len(targets)
    assert len(set(all_indices)) == len(targets)
