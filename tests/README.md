# Test Files Directory

This directory contains test cases for the SDK.

## Organization

- `test_main.c`: Test for hello function
- `test_cholesky.c`: Comprehensive test suite for Cholesky decomposition

## Key Files

- `test_cholesky.c`: Test suite with 7 test cases covering:
  - Identity matrix decomposition
  - Lower triangular property verification
  - Correctness (L × L^T = A)
  - Positive diagonal elements
  - Error handling for non-symmetric matrices
  - Error handling for non-positive definite matrices
  - Numerical stability

