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

// mlp_forward_batch runs a complete MLP forward pass (2 or 3 hidden layers) for n inputs.
// All weight matrices are row-major with transB layout (W[neuron][input]).
// h3_buf/w3h/b3h/w_out3/b_out3 are only used when H3 > 0 (3-layer model).
// h1_buf and h2_buf are scratch buffers [n*H1] and [n*H2].
static void mlp_forward_batch(
	int n, int nFeat, int H1, int H2, int H3,
	const float* feat,
	const float* w1, const float* b1,
	const float* w2, const float* b2,
	const float* w3h, const float* b3h,
	const float* w_out, float b_out,
	float yStd, float yMean,
	float* h1_buf, float* h2_buf, float* h3_buf,
	float* out
) {
	// Layer 1: h1 = relu(feat @ w1^T + b1)
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, H1, nFeat, 1.0f, feat, nFeat, w1, nFeat, 0.0f, h1_buf, H1);
	for (int i = 0; i < n; i++) {
		float* row = h1_buf + i * H1;
		for (int j = 0; j < H1; j++) {
			float v = row[j] + b1[j];
			row[j] = v > 0.0f ? v : 0.0f;
		}
	}

	// Layer 2: h2 = relu(h1 @ w2^T + b2)
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, H2, H1, 1.0f, h1_buf, H1, w2, H1, 0.0f, h2_buf, H2);
	for (int i = 0; i < n; i++) {
		float* row = h2_buf + i * H2;
		for (int j = 0; j < H2; j++) {
			float v = row[j] + b2[j];
			row[j] = v > 0.0f ? v : 0.0f;
		}
	}

	if (H3 > 0) {
		// Layer 3: h3 = relu(h2 @ w3h^T + b3h)
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, H3, H2, 1.0f, h2_buf, H2, w3h, H2, 0.0f, h3_buf, H3);
		for (int i = 0; i < n; i++) {
			float* row = h3_buf + i * H3;
			for (int j = 0; j < H3; j++) {
				float v = row[j] + b3h[j];
				row[j] = v > 0.0f ? v : 0.0f;
			}
		}
		for (int i = 0; i < n; i++) {
			float* row = h3_buf + i * H3;
			float s = b_out;
			for (int j = 0; j < H3; j++) s += w_out[j] * row[j];
			out[i] = s * yStd + yMean;
		}
	} else {
		for (int i = 0; i < n; i++) {
			float* row = h2_buf + i * H2;
			float s = b_out;
			for (int j = 0; j < H2; j++) s += w_out[j] * row[j];
			out[i] = s * yStd + yMean;
		}
	}
}
*/
import "C"
import "unsafe"

// AccelMLPForwardBatch runs a fused MLP forward pass in a single cgo call.
// Eliminates per-layer cgo overhead by doing all layers in C.
func AccelMLPForwardBatch(
	n, nFeat, H1, H2, H3 int,
	feat, w1, b1, w2, b2 []float32,
	w3h, b3h []float32, // nil for 2-layer models
	wOut []float32, bOut float32,
	yStd, yMean float32,
	h1Buf, h2Buf, h3Buf []float32, // h3Buf nil for 2-layer
	out []float32,
) {
	var pw3h, pb3h, ph3Buf *C.float
	if H3 > 0 {
		pw3h = (*C.float)(unsafe.Pointer(&w3h[0]))
		pb3h = (*C.float)(unsafe.Pointer(&b3h[0]))
		ph3Buf = (*C.float)(unsafe.Pointer(&h3Buf[0]))
	}
	C.mlp_forward_batch(
		C.int(n), C.int(nFeat), C.int(H1), C.int(H2), C.int(H3),
		(*C.float)(unsafe.Pointer(&feat[0])),
		(*C.float)(unsafe.Pointer(&w1[0])), (*C.float)(unsafe.Pointer(&b1[0])),
		(*C.float)(unsafe.Pointer(&w2[0])), (*C.float)(unsafe.Pointer(&b2[0])),
		pw3h, pb3h,
		(*C.float)(unsafe.Pointer(&wOut[0])), C.float(bOut),
		C.float(yStd), C.float(yMean),
		(*C.float)(unsafe.Pointer(&h1Buf[0])),
		(*C.float)(unsafe.Pointer(&h2Buf[0])),
		ph3Buf,
		(*C.float)(unsafe.Pointer(&out[0])),
	)
}

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
