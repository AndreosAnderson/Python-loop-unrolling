import numpy as np
import pytest
from matrix_operations import matrix_multiply, track_memory, track_cpu

@pytest.fixture
def setup_matrices():
    size = 512
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    return A, B

@pytest.mark.benchmark(min_rounds=5)
def test_matrix_multiply(benchmark, setup_matrices):
    A, B = setup_matrices

    result = benchmark(matrix_multiply, A, B)

    track_memory(matrix_multiply, A, B)
    track_cpu(matrix_multiply, A, B)

    assert result is not None
