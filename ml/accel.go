//go:build darwin

package ml

/*
#cgo LDFLAGS: -framework Accelerate
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

// sgemv_bias computes y = A @ x + y, where y is pre-loaded with the bias vector.
// A is row-major [M x N], x is [N], y is [M].
static inline void sgemv_bias(int M, int N, const float* A, const float* x, float* y) {
	cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0f, A, N, x, 1, 1.0f, y, 1);
}

// sgemv_trans computes y = A^T @ x + y.
// A is row-major [M x N], x is [M], y is [N].
static inline void sgemv_trans(int M, int N, const float* A, const float* x, float* y) {
	cblas_sgemv(CblasRowMajor, CblasTrans, M, N, 1.0f, A, N, x, 1, 1.0f, y, 1);
}

// sger_update computes A = alpha * (x @ y^T) + A.
// A is row-major [M x N], x is [M], y is [N].
static inline void sger_update(int M, int N, float alpha, const float* x, const float* y, float* A) {
	cblas_sger(CblasRowMajor, M, N, alpha, x, 1, y, 1, A, N);
}

// sgemm_flexible computes C = alpha * (op(A) @ op(B)) + beta * C.
static inline void sgemm_flexible(int transA, int transB, int M, int N, int K, const float* A, const float* B, float* C) {
	enum CBLAS_TRANSPOSE tA = transA ? CblasTrans : CblasNoTrans;
	enum CBLAS_TRANSPOSE tB = transB ? CblasTrans : CblasNoTrans;
	int lda = transA ? M : K;
	int ldb = transB ? K : N;
	cblas_sgemm(CblasRowMajor, tA, tB, M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, N);
}
*/
import "C"
import "unsafe"

// ... (previous functions)

// AccelGEMM computes C = op(A) @ op(B).
func AccelGEMM(transA, transB bool, M, N, K int, A, B, C_out []float32) {
	tA, tB := 0, 0
	if transA { tA = 1 }
	if transB { tB = 1 }
	C.sgemm_flexible(
		C.int(tA), C.int(tB),
		C.int(M), C.int(N), C.int(K),
		(*C.float)(unsafe.Pointer(&A[0])),
		(*C.float)(unsafe.Pointer(&B[0])),
		(*C.float)(unsafe.Pointer(&C_out[0])),
	)
}

// accelGEMV computes y = A @ x + y (y pre-filled with bias).
// A is row-major [M x N].
func accelGEMV(M, N int, A []float32, x []float32, y []float32) {
	C.sgemv_bias(
		C.int(M), C.int(N),
		(*C.float)(unsafe.Pointer(&A[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&y[0])),
	)
}

// accelGEMVTrans computes y = A^T @ x + y.
// A is row-major [M x N], x is [M], y is [N].
func accelGEMVTrans(M, N int, A []float32, x []float32, y []float32) {
	C.sgemv_trans(
		C.int(M), C.int(N),
		(*C.float)(unsafe.Pointer(&A[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&y[0])),
	)
}

// accelGER computes A = alpha * (x @ y^T) + A.
// A is row-major [M x N], x is [M], y is [N].
func accelGER(M, N int, alpha float32, x []float32, y []float32, A []float32) {
	C.sger_update(
		C.int(M), C.int(N), C.float(alpha),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&y[0])),
		(*C.float)(unsafe.Pointer(&A[0])),
	)
}
