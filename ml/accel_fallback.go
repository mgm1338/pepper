//go:build !darwin

package ml

// accelGEMV computes y = A @ x + y (y pre-filled with bias).
// A is row-major [M x N].
// Pure-Go fallback for non-darwin builds (e.g. Linux Docker containers)
// where Apple's Accelerate framework is unavailable.
func accelGEMV(M, N int, A []float32, x []float32, y []float32) {
	for i := 0; i < M; i++ {
		off := i * N
		v := y[i]
		for j := 0; j < N; j++ {
			v += A[off+j] * x[j]
		}
		y[i] = v
	}
}

// accelGEMVTrans computes y = A^T @ x + y.
// A is row-major [M x N], x is [M], y is [N].
func accelGEMVTrans(M, N int, A []float32, x []float32, y []float32) {
	for i := 0; i < M; i++ {
		off := i * N
		xi := x[i]
		for j := 0; j < N; j++ {
			y[j] += A[off+j] * xi
		}
	}
}

// accelGER computes A = alpha * (x @ y^T) + A.
// A is row-major [M x N], x is [M], y is [N].
func accelGER(M, N int, alpha float32, x []float32, y []float32, A []float32) {
	for i := 0; i < M; i++ {
		off := i * N
		axi := alpha * x[i]
		row := A[off : off+N]
		for j := 0; j < N; j++ {
			row[j] += axi * y[j]
		}
	}
}

// AccelGEMM computes C = op(A) @ op(B).
func AccelGEMM(transA, transB bool, M, N, K int, A, B, C_out []float32) {
	for i := 0; i < M; i++ {
		cRow := C_out[i*N : (i+1)*N]
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				var av, bv float32
				if transA { av = A[k*M+i] } else { av = A[i*K+k] }
				if transB { bv = B[j*K+k] } else { bv = B[k*N+j] }
				sum += av * bv
			}
			cRow[j] = sum
		}
	}
}
