package ml

// goGEMMBatchT computes C = A @ B^T without cgo.
// A: [M × K] row-major (M samples, K features each)
// B: [N × K] row-major (N neurons, K weights each — accessed transposed)
// C: [M × N] row-major output (overwritten, not accumulated)
//
// Used by ScoreBatch for inference with small M (≤ MaxCardBatch).
// 8-wide unrolling in N matches scoreGoFallback's register strategy.
func goGEMMBatchT(M, N, K int, A, B, C []float32) {
	for i := 0; i < M; i++ {
		aRow := A[i*K : i*K+K]
		cRow := C[i*N : i*N+N]
		j := 0
		for ; j+7 < N; j += 8 {
			b0 := B[(j+0)*K : (j+0)*K+K]
			b1 := B[(j+1)*K : (j+1)*K+K]
			b2 := B[(j+2)*K : (j+2)*K+K]
			b3 := B[(j+3)*K : (j+3)*K+K]
			b4 := B[(j+4)*K : (j+4)*K+K]
			b5 := B[(j+5)*K : (j+5)*K+K]
			b6 := B[(j+6)*K : (j+6)*K+K]
			b7 := B[(j+7)*K : (j+7)*K+K]
			var v0, v1, v2, v3, v4, v5, v6, v7 float32
			for k, ak := range aRow {
				v0 += b0[k] * ak
				v1 += b1[k] * ak
				v2 += b2[k] * ak
				v3 += b3[k] * ak
				v4 += b4[k] * ak
				v5 += b5[k] * ak
				v6 += b6[k] * ak
				v7 += b7[k] * ak
			}
			cRow[j+0] = v0
			cRow[j+1] = v1
			cRow[j+2] = v2
			cRow[j+3] = v3
			cRow[j+4] = v4
			cRow[j+5] = v5
			cRow[j+6] = v6
			cRow[j+7] = v7
		}
		for ; j < N; j++ {
			bRow := B[j*K : j*K+K]
			var v float32
			for k, ak := range aRow {
				v += bRow[k] * ak
			}
			cRow[j] = v
		}
	}
}
