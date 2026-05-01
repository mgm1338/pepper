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

// AccelMLPForwardBatch runs a fused MLP forward pass in pure Go.
// Equivalent to the darwin cgo version; used on Linux builds.
func AccelMLPForwardBatch(
	n, nFeat, H1, H2, H3 int,
	feat, w1, b1, w2, b2 []float32,
	w3h, b3h []float32,
	wOut []float32, bOut float32,
	yStd, yMean float32,
	h1Buf, h2Buf, h3Buf []float32,
	out []float32,
) {
	relu := func(v float32) float32 {
		if v < 0 {
			return 0
		}
		return v
	}

	for i := 0; i < n; i++ {
		x := feat[i*nFeat : (i+1)*nFeat]

		// Layer 1: h1 = ReLU(w1 @ x + b1)
		h1 := h1Buf[i*H1 : (i+1)*H1]
		for j := 0; j < H1; j++ {
			v := b1[j]
			row := w1[j*nFeat : (j+1)*nFeat]
			for k, xk := range x {
				v += row[k] * xk
			}
			h1[j] = relu(v)
		}

		// Layer 2: h2 = ReLU(w2 @ h1 + b2)
		h2 := h2Buf[i*H2 : (i+1)*H2]
		for j := 0; j < H2; j++ {
			v := b2[j]
			row := w2[j*H1 : (j+1)*H1]
			for k, hk := range h1 {
				v += row[k] * hk
			}
			h2[j] = relu(v)
		}

		// Layer 3 (optional): h3 = ReLU(w3h @ h2 + b3h)
		var hLast []float32
		hLastLen := H2
		if H3 > 0 {
			h3 := h3Buf[i*H3 : (i+1)*H3]
			for j := 0; j < H3; j++ {
				v := b3h[j]
				row := w3h[j*H2 : (j+1)*H2]
				for k, hk := range h2 {
					v += row[k] * hk
				}
				h3[j] = relu(v)
			}
			hLast = h3
			hLastLen = H3
		} else {
			hLast = h2
		}

		// Output layer: scalar dot product
		v := bOut
		for j := 0; j < hLastLen; j++ {
			v += wOut[j] * hLast[j]
		}
		out[i] = v*yStd + yMean
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
