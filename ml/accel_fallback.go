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
