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
*/
import "C"
import "unsafe"

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
