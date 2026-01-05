# Cholesky Decomposition

## Overview

Cholesky decomposition is a matrix factorization method that decomposes a symmetric positive definite matrix **A** into the product of a lower triangular matrix **L** and its transpose:

**A = L × L^T**

where **L** is a lower triangular matrix with positive diagonal elements.

This decomposition is widely used in numerical linear algebra for:
- Solving systems of linear equations
- Computing matrix inverses
- Optimization problems
- Statistical computations (e.g., covariance matrices)
- Monte Carlo simulations

## API Reference

### Function Signature

```c
int cholesky_decompose_32x32(double A[32][32], double L[32][32]);
```

### Parameters

- **A** (input): A 32×32 symmetric positive definite matrix to decompose
- **L** (output): A 32×32 lower triangular matrix such that L × L^T = A

### Return Values

- **0**: Success - decomposition completed successfully
- **1**: Error - Matrix is not symmetric
- **2**: Error - Matrix is not positive definite
- **3**: Error - Numerical error during computation (e.g., negative value under square root)

### Usage Example

```c
#include "cholesky.h"
#include <stdio.h>

int main(void) {
    double A[32][32];
    double L[32][32];
    
    // Initialize A as a symmetric positive definite matrix
    // ... (fill A with your matrix data)
    
    int result = cholesky_decompose_32x32(A, L);
    
    if (result == 0) {
        printf("Cholesky decomposition successful\n");
        // Use L for further computations
    } else {
        printf("Error: Cholesky decomposition failed with code %d\n", result);
        return 1;
    }
    
    return 0;
}
```

## Mathematical Background

### Algorithm

The Cholesky decomposition uses the Cholesky-Crout algorithm (column-oriented):

For i = 0 to 31:
1. Compute diagonal element:
   ```
   L[i][i] = sqrt(A[i][i] - Σ(k=0 to i-1) L[i][k]²)
   ```

2. Compute off-diagonal elements in column i:
   ```
   L[j][i] = (A[j][i] - Σ(k=0 to i-1) L[j][k] × L[i][k]) / L[i][i]
   ```
   for j = i+1 to 31

### Complexity

- **Time Complexity**: O(n³/3) where n = 32, approximately 10,923 operations
- **Space Complexity**: O(n²) = O(1024) for storing the result matrix

### Requirements

The input matrix **A** must satisfy:
1. **Symmetric**: A[i][j] = A[j][i] for all i, j
2. **Positive Definite**: All eigenvalues are positive, which implies:
   - All leading principal minors are positive
   - All diagonal elements of L are positive during decomposition

## Error Conditions

### Error Code 1: Non-Symmetric Matrix

The function checks that A[i][j] ≈ A[j][i] (within floating-point tolerance) for all i, j before decomposition.

### Error Code 2: Non-Positive Definite Matrix

If during decomposition, the value under the square root becomes negative or zero (within tolerance), the matrix is not positive definite.

### Error Code 3: Numerical Error

Occurs when:
- Division by zero (should not happen if matrix is positive definite)
- Overflow/underflow during computation
- Invalid floating-point operations

## Numerical Considerations

- Uses `double` precision (64-bit floating-point) for better numerical stability
- Includes epsilon tolerance (typically 1e-10) for floating-point comparisons
- Handles near-zero values appropriately to avoid division by zero
- Optimized memory access patterns for cache performance on 32×32 matrices

## See Also

- Test cases: `tests/test_cholesky.c`
- Implementation: `src/cholesky.c`
- Header file: `include/cholesky.h`

