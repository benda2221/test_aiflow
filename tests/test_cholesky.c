#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "cholesky.h"

#define N 32
#define EPSILON 1e-10

/* Test matrix constants */
#define TEST_DIAGONAL_VALUE 2.0
#define TEST_OFF_DIAGONAL_DECAY 0.1
#define TEST_STABILITY_DIAGONAL 10.0
#define TEST_STABILITY_OFF_DIAGONAL 0.01

/* Helper function to check if two matrices are approximately equal */
static bool matrix_equal(double A[N][N], double B[N][N], double tol) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(A[i][j] - B[i][j]) > tol) {
                return false;
            }
        }
    }
    return true;
}

/* Helper function to check if matrix is lower triangular */
static bool is_lower_triangular(double L[N][N], double tol) {
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            if (fabs(L[i][j]) > tol) {
                return false;
            }
        }
    }
    return true;
}

/* Helper function to compute L × L^T */
static void multiply_llt(double L[N][N], double result[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                result[i][j] += L[i][k] * L[j][k];
            }
        }
    }
}

/* Helper function to create identity matrix */
static void create_identity(double A[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

/* Helper function to create a simple positive definite matrix */
static void create_simple_pd_matrix(double A[N][N]) {
    /* Create A = I + small random-like values to ensure positive definiteness */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                A[i][j] = TEST_DIAGONAL_VALUE; /* Strong diagonal */
            } else {
                int diff = (i > j) ? (i - j) : (j - i);
                A[i][j] = TEST_OFF_DIAGONAL_DECAY / (diff + 1); /* Decay off-diagonal */
            }
        }
    }
    /* Make symmetric */
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            A[i][j] = A[j][i];
        }
    }
}

/* Test 1: Identity matrix decomposition */
static int test_identity_matrix(void) {
    double A[N][N];
    double L[N][N];
    double reconstructed[N][N];
    
    create_identity(A);
    
    int result = cholesky_decompose_32x32(A, L);
    if (result != 0) {
        printf("Test 1 FAILED: Identity matrix decomposition returned error %d\n", result);
        return 1;
    }
    
    /* Verify L is identity (lower triangular with ones on diagonal) */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(L[i][j] - expected) > EPSILON) {
                printf("Test 1 FAILED: L[%d][%d] = %f, expected %f\n", i, j, L[i][j], expected);
                return 1;
            }
        }
    }
    
    /* Verify L × L^T = A */
    multiply_llt(L, reconstructed);
    if (!matrix_equal(A, reconstructed, EPSILON)) {
        printf("Test 1 FAILED: L × L^T does not equal A\n");
        return 1;
    }
    
    printf("Test 1 PASSED: Identity matrix decomposition\n");
    return 0;
}

/* Test 2: Lower triangular property */
static int test_lower_triangular_property(void) {
    double A[N][N];
    double L[N][N];
    
    create_simple_pd_matrix(A);
    
    int result = cholesky_decompose_32x32(A, L);
    if (result != 0) {
        printf("Test 2 FAILED: Decomposition returned error %d\n", result);
        return 1;
    }
    
    if (!is_lower_triangular(L, EPSILON)) {
        printf("Test 2 FAILED: Result matrix is not lower triangular\n");
        return 1;
    }
    
    printf("Test 2 PASSED: Lower triangular property verified\n");
    return 0;
}

/* Test 3: Correctness - L × L^T = A */
static int test_correctness(void) {
    double A[N][N];
    double L[N][N];
    double reconstructed[N][N];
    
    create_simple_pd_matrix(A);
    
    int result = cholesky_decompose_32x32(A, L);
    if (result != 0) {
        printf("Test 3 FAILED: Decomposition returned error %d\n", result);
        return 1;
    }
    
    multiply_llt(L, reconstructed);
    if (!matrix_equal(A, reconstructed, EPSILON * 100)) {
        printf("Test 3 FAILED: L × L^T does not equal A\n");
        return 1;
    }
    
    printf("Test 3 PASSED: Correctness verified (L × L^T = A)\n");
    return 0;
}

/* Test 4: Positive diagonal elements */
static int test_positive_diagonal(void) {
    double A[N][N];
    double L[N][N];
    
    create_simple_pd_matrix(A);
    
    int result = cholesky_decompose_32x32(A, L);
    if (result != 0) {
        printf("Test 4 FAILED: Decomposition returned error %d\n", result);
        return 1;
    }
    
    for (int i = 0; i < N; i++) {
        if (L[i][i] <= 0) {
            printf("Test 4 FAILED: L[%d][%d] = %f is not positive\n", i, i, L[i][i]);
            return 1;
        }
    }
    
    printf("Test 4 PASSED: All diagonal elements are positive\n");
    return 0;
}

/* Test 5: Non-symmetric matrix error handling */
static int test_non_symmetric_error(void) {
    double A[N][N];
    double L[N][N];
    
    create_identity(A);
    A[0][1] = 1.0;
    A[1][0] = 2.0; /* Make it non-symmetric */
    
    int result = cholesky_decompose_32x32(A, L);
    if (result == 0) {
        printf("Test 5 FAILED: Should have returned error for non-symmetric matrix\n");
        return 1;
    }
    
    if (result != 1) {
        printf("Test 5 FAILED: Expected error code 1 (non-symmetric), got %d\n", result);
        return 1;
    }
    
    printf("Test 5 PASSED: Non-symmetric matrix correctly rejected\n");
    return 0;
}

/* Test 6: Non-positive definite matrix error handling */
static int test_non_positive_definite_error(void) {
    double A[N][N];
    double L[N][N];
    
    /* Create a matrix that is symmetric but not positive definite */
    create_identity(A);
    for (int i = 0; i < N; i++) {
        A[i][i] = -1.0; /* Negative diagonal makes it not positive definite */
    }
    
    int result = cholesky_decompose_32x32(A, L);
    if (result == 0) {
        printf("Test 6 FAILED: Should have returned error for non-positive definite matrix\n");
        return 1;
    }
    
    if (result != 2) {
        printf("Test 6 FAILED: Expected error code 2 (non-positive definite), got %d\n", result);
        return 1;
    }
    
    printf("Test 6 PASSED: Non-positive definite matrix correctly rejected\n");
    return 0;
}

/* Test 7: Numerical stability with well-conditioned matrix */
static int test_numerical_stability(void) {
    double A[N][N];
    double L[N][N];
    double reconstructed[N][N];
    
    /* Create a well-conditioned positive definite matrix */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                A[i][j] = TEST_STABILITY_DIAGONAL;
            } else {
                int diff = (i > j) ? (i - j) : (j - i);
                A[i][j] = TEST_STABILITY_OFF_DIAGONAL / (diff + 1);
            }
        }
    }
    /* Make symmetric */
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            A[i][j] = A[j][i];
        }
    }
    
    int result = cholesky_decompose_32x32(A, L);
    if (result != 0) {
        printf("Test 7 FAILED: Decomposition returned error %d\n", result);
        return 1;
    }
    
    multiply_llt(L, reconstructed);
    if (!matrix_equal(A, reconstructed, EPSILON * 1000)) {
        printf("Test 7 FAILED: Numerical stability issue - reconstruction error too large\n");
        return 1;
    }
    
    printf("Test 7 PASSED: Numerical stability verified\n");
    return 0;
}

int main(void) {
    int failures = 0;
    
    printf("Running Cholesky decomposition tests...\n\n");
    
    failures += test_identity_matrix();
    failures += test_lower_triangular_property();
    failures += test_correctness();
    failures += test_positive_diagonal();
    failures += test_non_symmetric_error();
    failures += test_non_positive_definite_error();
    failures += test_numerical_stability();
    
    printf("\n");
    if (failures == 0) {
        printf("All tests PASSED (7/7)\n");
        return 0;
    } else {
        printf("Tests FAILED: %d out of 7 tests failed\n", failures);
        return 1;
    }
}

