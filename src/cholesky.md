# cholesky.c

Implementation of Cholesky decomposition for 32×32 positive definite matrices.

## External Interface

### cholesky_decompose_32x32()

```c
int cholesky_decompose_32x32(double A[32][32], double L[32][32]);
```

Performs Cholesky decomposition of a symmetric positive definite matrix.

**Parameters:**
- `A` (input): 32×32 symmetric positive definite matrix to decompose
- `L` (output): 32×32 lower triangular matrix such that L × L^T = A

**Return values:**
- `0`: Success - decomposition completed successfully
- `1`: Error - Matrix is not symmetric (A[i][j] ≠ A[j][i] within tolerance)
- `2`: Error - Matrix is not positive definite (diagonal value ≤ 0 during decomposition)
- `3`: Error - Numerical error during computation (division by zero or invalid operation)

**Algorithm:**
Uses Cholesky-Crout algorithm (column-oriented):
- For each column i from 0 to 31:
  1. Compute diagonal element: L[i][i] = sqrt(A[i][i] - Σ(k=0 to i-1) L[i][k]²)
  2. Compute off-diagonal elements: L[j][i] = (A[j][i] - Σ(k=0 to i-1) L[j][k] × L[i][k]) / L[i][i]

**Time complexity:** O(n³/3) where n=32  
**Space complexity:** O(n²) for result matrix

## Internal Helpers

### Symmetry Validation

The function validates matrix symmetry before decomposition by checking:
```c
fabs(A[i][j] - A[j][i]) <= EPSILON
```
for all i, j where i < j. Uses `EPSILON = 1e-10` for floating-point tolerance.

### Positive Definiteness Check

During decomposition, the function checks that:
```c
diag_value = A[i][i] - sum(L[i][k]² for k < i) > EPSILON
```
If this condition fails, the matrix is not positive definite and the function returns error code 2.

### Numerical Error Handling

The function checks for division by zero when computing off-diagonal elements:
```c
if (fabs(L[i][i]) < EPSILON) {
    return 3; /* Numerical error */
}
```
This should not occur if the matrix is positive definite, but provides safety against numerical instability.

## Constants

- `N`: Matrix dimension (32)
- `EPSILON`: Floating-point tolerance (1e-10)

## Dependencies

- `math.h`: For `sqrt()`, `fabs()`

