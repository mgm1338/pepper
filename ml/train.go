package ml

import (
	"math"
	"math/rand"
	"sync"
)

// MLPTrainer implements training logic for an MLP.
type MLPTrainer struct {
	W MLPWeights
	// Contiguous weights
	F1 []float32
	F2 []float32
	F3 []float32 // output weights for 2-layer or W3H for 3-layer
	F4 []float32 // output weights for 3-layer

	// First moment (Adam)
	M1_F1 []float32
	M1_B1 []float32
	M1_F2 []float32
	M1_B2 []float32
	M1_F3 []float32
	M1_B3 []float32
	M1_F4 []float32 // for 3-layer output
	M1_B4 float32   // for 3-layer output

	// Second moment (Adam)
	V1_F1 []float32
	V1_B1 []float32
	V1_F2 []float32
	V1_B2 []float32
	V1_F3 []float32
	V1_B3 []float32
	V1_F4 []float32
	V1_B4 float32

	// Gradients (accumulated over batch)
	G_F1 []float32
	G_B1 []float32
	G_F2 []float32
	G_B2 []float32
	G_F3 []float32
	G_B3 []float32
	G_F4 []float32
	G_B4 float32

	// Scratch buffers for backward pass
	H1 []float32
	H2 []float32
	H3 []float32
	D1 []float32 // deltas (pre-activation gradients)
	D2 []float32
	D3 []float32

	// Batch buffers (for GEMM)
	BatchX  []float32 // [BatchSize * NFeatures]
	BatchH1 []float32 // [BatchSize * Hidden1]
	BatchH2 []float32 // [BatchSize * Hidden2]
	BatchH3 []float32 // [BatchSize * Hidden3]
	BatchD1 []float32
	BatchD2 []float32
	BatchD3 []float32
	BatchY  []float32 // targets [BatchSize]

	T int64 // Adam step counter
}

func NewMLPTrainer(nFeat, h1, h2, h3 int, rng *rand.Rand) *MLPTrainer {
	t := &MLPTrainer{
		W: MLPWeights{
			NFeatures: nFeat,
			Hidden1:  h1,
			Hidden2:  h2,
			Hidden3:  h3,
		},
	}

	// Initialize weights (Xavier/Glorot)
	initW := func(rows, cols int) ([]float32, []float32) {
		limit := float32(math.Sqrt(6.0 / float64(rows+cols)))
		w := make([]float32, rows*cols)
		for i := range w {
			w[i] = rng.Float32()*2*limit - limit
		}
		return w, make([]float32, rows)
	}

	t.F1, t.W.B1 = initW(h1, nFeat)
	t.G_B1 = make([]float32, h1)
	t.M1_F1, t.V1_F1 = make([]float32, h1*nFeat), make([]float32, h1*nFeat)
	t.M1_B1, t.V1_B1 = make([]float32, h1), make([]float32, h1)
	t.G_F1 = make([]float32, h1*nFeat)

	t.F2, t.W.B2 = initW(h2, h1)
	t.G_B2 = make([]float32, h2)
	t.M1_F2, t.V1_F2 = make([]float32, h2*h1), make([]float32, h2*h1)
	t.M1_B2, t.V1_B2 = make([]float32, h2), make([]float32, h2)
	t.G_F2 = make([]float32, h2*h1)

	if h3 > 0 {
		t.F3, t.W.B3H = initW(h3, h2)
		t.G_B3 = make([]float32, h3)
		t.M1_F3, t.V1_F3 = make([]float32, h3*h2), make([]float32, h3*h2)
		t.M1_B3, t.V1_B3 = make([]float32, h3), make([]float32, h3)
		t.G_F3 = make([]float32, h3*h2)

		limit := float32(math.Sqrt(6.0 / float64(h3+1)))
		t.F4 = make([]float32, h3)
		for i := range t.F4 {
			t.F4[i] = rng.Float32()*2*limit - limit
		}
		t.M1_F4, t.V1_F4 = make([]float32, h3), make([]float32, h3)
		t.G_F4 = make([]float32, h3)
	} else {
		limitW := float32(math.Sqrt(6.0 / float64(h2+1)))
		t.F3 = make([]float32, h2)
		for i := range t.F3 {
			t.F3[i] = rng.Float32()*2*limitW - limitW
		}
		t.M1_F3, t.V1_F3 = make([]float32, h2), make([]float32, h2)
		t.G_F3 = make([]float32, h2)
		t.G_B3 = make([]float32, 1) // scalar bias for output
		t.M1_B3, t.V1_B3 = make([]float32, 1), make([]float32, 1)
	}

	t.H1, t.D1 = make([]float32, h1), make([]float32, h1)
	t.H2, t.D2 = make([]float32, h2), make([]float32, h2)
	if h3 > 0 {
		t.H3, t.D3 = make([]float32, h3), make([]float32, h3)
	}

	t.ResizeBatch(4096) // Default batch size

	return t
}

func (t *MLPTrainer) ResizeBatch(n int) {
	t.BatchX = make([]float32, n*t.W.NFeatures)
	t.BatchH1 = make([]float32, n*t.W.Hidden1)
	t.BatchH2 = make([]float32, n*t.W.Hidden2)
	if t.W.Hidden3 > 0 {
		t.BatchH3 = make([]float32, n*t.W.Hidden3)
		t.BatchD3 = make([]float32, n*t.W.Hidden3)
	}
	t.BatchD1 = make([]float32, n*t.W.Hidden1)
	t.BatchD2 = make([]float32, n*t.W.Hidden2)
	t.BatchY = make([]float32, n)
}

// ForwardBatch processes n samples at once using optimized GEMM and threading.
func (t *MLPTrainer) ForwardBatch(n int, results []float32) {
	// L1: BatchH1 = BatchX @ F1^T
	AccelGEMM(false, true, n, t.W.Hidden1, t.W.NFeatures, t.BatchX, t.F1, t.BatchH1)
	
	var wg sync.WaitGroup
	numWorkers := 8
	chunkSize := (n + numWorkers - 1) / numWorkers
	for w := 0; numWorkers > 0 && w < numWorkers; w++ {
		start, end := w*chunkSize, (w+1)*chunkSize
		if start >= n { break }
		if end > n { end = n }
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				row := t.BatchH1[i*t.W.Hidden1 : (i+1)*t.W.Hidden1]
				b1 := t.W.B1
				for j := 0; j < len(row); j++ {
					v := row[j] + b1[j]
					if v < 0 { v = 0 }
					row[j] = v
				}
			}
		}(start, end)
	}
	wg.Wait()

	// L2: BatchH2 = BatchH1 @ F2^T
	AccelGEMM(false, true, n, t.W.Hidden2, t.W.Hidden1, t.BatchH1, t.F2, t.BatchH2)
	for w := 0; numWorkers > 0 && w < numWorkers; w++ {
		start, end := w*chunkSize, (w+1)*chunkSize
		if start >= n { break }
		if end > n { end = n }
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				row := t.BatchH2[i*t.W.Hidden2 : (i+1)*t.W.Hidden2]
				b2 := t.W.B2
				for j := 0; j < len(row); j++ {
					v := row[j] + b2[j]
					if v < 0 { v = 0 }
					row[j] = v
				}
			}
		}(start, end)
	}
	wg.Wait()

	if t.W.Hidden3 > 0 {
		// L3: BatchH3 = BatchH2 @ F3^T
		AccelGEMM(false, true, n, t.W.Hidden3, t.W.Hidden2, t.BatchH2, t.F3, t.BatchH3)
		for w := 0; numWorkers > 0 && w < numWorkers; w++ {
			start, end := w*chunkSize, (w+1)*chunkSize
			if start >= n { break }
			if end > n { end = n }
			wg.Add(1)
			go func(s, e int) {
				defer wg.Done()
				for i := s; i < e; i++ {
					row := t.BatchH3[i*t.W.Hidden3 : (i+1)*t.W.Hidden3]
					for j, b := range t.W.B3H {
						v := row[j] + b
						if v < 0 { v = 0 }
						row[j] = v
					}
					// Output
					out := t.W.B4
					for j, v := range row { out += t.F4[j] * v }
					results[i] = out
				}
			}(start, end)
		}
		wg.Wait()
	} else {
		// Output (2-layer)
		for w := 0; numWorkers > 0 && w < numWorkers; w++ {
			start, end := w*chunkSize, (w+1)*chunkSize
			if start >= n { break }
			if end > n { end = n }
			wg.Add(1)
			go func(s, e int) {
				defer wg.Done()
				for i := s; i < e; i++ {
					row := t.BatchH2[i*t.W.Hidden2 : (i+1)*t.W.Hidden2]
					out := t.W.B3
					for j, v := range row { out += t.F3[j] * v }
					results[i] = out
				}
			}(start, end)
		}
		wg.Wait()
	}
}

// BackwardBatch computes gradients for n samples using block GEMM.
func (t *MLPTrainer) BackwardBatch(n int, preds []float32, targets []float32) {
	// 1. Output error
	// For 2-layer: output is results = BatchH2 @ F3 + B3
	// dLoss/dPred = preds - targets (for MSE)
	// dLoss/dF3 = (preds - targets)^T @ BatchH2  -> [1 x BatchSize] @ [BatchSize x Hidden2] = [1 x Hidden2]
	// dLoss/dBatchH2 = (preds - targets) @ F3^T  -> [BatchSize x 1] @ [1 x Hidden2] = [BatchSize x Hidden2]
	
	if t.W.Hidden3 > 0 {
		// 3-layer path
		for i := 0; i < n; i++ { t.BatchD3[i] = preds[i] - targets[i] }
		// G_F4 = BatchD3^T @ BatchH3
		for i := 0; i < n; i++ {
			grad := t.BatchD3[i]
			t.G_B4 += grad
			row := t.BatchH3[i*t.W.Hidden3 : (i+1)*t.W.Hidden3]
			for j, v := range row { t.G_F4[j] += grad * v }
		}
		// dBatchH3 = BatchD3 @ F4^T
		// BatchD3 is [n x 1], F4 is [1 x Hidden3] (wait, F4 is [Hidden3])
		// So dBatchH3[i, j] = BatchD3[i] * t.F4[j]
		// Then apply ReLU derivative
		for i := 0; i < n; i++ {
			grad := t.BatchD3[i]
			hRow := t.BatchH3[i*t.W.Hidden3 : (i+1)*t.W.Hidden3]
			dRow := t.BatchD3[i*t.W.Hidden3 : (i+1)*t.W.Hidden3]
			for j, hv := range hRow {
				if hv > 0 { dRow[j] = grad * t.F4[j] } else { dRow[j] = 0 }
			}
		}
		// G_F3 = BatchD3^T @ BatchH2 -> [Hidden3 x n] @ [n x Hidden2]
		AccelGEMM(true, false, t.W.Hidden3, t.W.Hidden2, n, t.BatchD3, t.BatchH2, t.G_F3)
		// Bias G_B3: sum columns of BatchD3
		for i := 0; i < n; i++ {
			row := t.BatchD3[i*t.W.Hidden3 : (i+1)*t.W.Hidden3]
			for j, v := range row { t.G_B3[j] += v }
		}
		// BatchD2 = BatchD3 @ F3
		AccelGEMM(false, false, n, t.W.Hidden2, t.W.Hidden3, t.BatchD3, t.F3, t.BatchD2)
		// Apply ReLU'(H2)
		for i := 0; i < n * t.W.Hidden2; i++ {
			if t.BatchH2[i] <= 0 { t.BatchD2[i] = 0 }
		}
	} else {
		// 2-layer path
		// Compute output error deltas in BatchD2
		for i := 0; i < n; i++ {
			grad := preds[i] - targets[i]
			t.G_B3[0] += grad
			// Accumulate G_F3: [1 x Hidden2] = [1 x n] @ [n x Hidden2]
			row := t.BatchH2[i*t.W.Hidden2 : (i+1)*t.W.Hidden2]
			for j, v := range row { t.G_F3[j] += grad * v }
			
			// Compute BatchD2: [n x Hidden2] = [n x 1] @ [1 x Hidden2]
			dRow := t.BatchD2[i*t.W.Hidden2 : (i+1)*t.W.Hidden2]
			for j, fv := range t.F3 {
				if row[j] > 0 { dRow[j] = grad * fv } else { dRow[j] = 0 }
			}
		}
	}

	// Grad wrt L1: G_F2 = BatchD2^T @ BatchH1 -> [Hidden2 x n] @ [n x Hidden1]
	AccelGEMM(true, false, t.W.Hidden2, t.W.Hidden1, n, t.BatchD2, t.BatchH1, t.G_F2)
	for i := 0; i < n; i++ {
		row := t.BatchD2[i*t.W.Hidden2 : (i+1)*t.W.Hidden2]
		for j, v := range row { t.G_B2[j] += v }
	}
	
	// BatchD1 = BatchD2 @ F2 -> [n x Hidden1] = [n x Hidden2] @ [Hidden2 x Hidden1]
	AccelGEMM(false, false, n, t.W.Hidden1, t.W.Hidden2, t.BatchD2, t.F2, t.BatchD1)
	for i := 0; i < n * t.W.Hidden1; i++ {
		if t.BatchH1[i] <= 0 { t.BatchD1[i] = 0 }
	}

	// Grad wrt Input: G_F1 = BatchD1^T @ BatchX -> [Hidden1 x n] @ [n x NFeatures]
	AccelGEMM(true, false, t.W.Hidden1, t.W.NFeatures, n, t.BatchD1, t.BatchX, t.G_F1)
	for i := 0; i < n; i++ {
		row := t.BatchD1[i*t.W.Hidden1 : (i+1)*t.W.Hidden1]
		for j, v := range row { t.G_B1[j] += v }
	}
}

// Forward computes the output and fills scratch buffers with activations.
func (t *MLPTrainer) Forward(x []float32) float32 {
	h1, h2, h3 := t.H1, t.H2, t.H3
	w1, w2, w3 := t.F1, t.F2, t.F3
	b1, b2 := t.W.B1, t.W.B2

	// L1
	copy(h1, b1)
	accelGEMV(len(h1), len(x), w1, x, h1)
	for i, v := range h1 {
		if v < 0 { h1[i] = 0 }
	}

	// L2
	copy(h2, b2)
	accelGEMV(len(h2), len(h1), w2, h1, h2)
	for i, v := range h2 {
		if v < 0 { h2[i] = 0 }
	}

	if t.W.Hidden3 > 0 {
		// L3
		copy(h3, t.W.B3H)
		accelGEMV(len(h3), len(h2), w3, h2, h3)
		for i, v := range h3 {
			if v < 0 { h3[i] = 0 }
		}
		// Output
		out := t.W.B4
		for i, v := range h3 {
			out += t.F4[i] * v
		}
		return out
	}

	// Output (2-layer)
	out := t.W.B3
	for i, v := range h2 {
		out += t.F3[i] * v
	}
	return out
}

// Backward computes gradients for a single example and accumulates them.
func (t *MLPTrainer) Backward(x []float32, gradOut float32) {
	h1, h2, h3 := t.H1, t.H2, t.H3
	d1, d2, d3 := t.D1, t.D2, t.D3

	if t.W.Hidden3 > 0 {
		// Grad wrt L3 pre-activation
		for i := range d3 {
			if h3[i] > 0 {
				d3[i] = gradOut * t.F4[i]
			} else {
				d3[i] = 0
			}
		}
		// Accumulate L4 (output) weights grad
		for i, v := range h3 {
			t.G_F4[i] += gradOut * v
		}
		t.G_B4 += gradOut

		// Grad wrt L2 output: d2_out = F3^T @ d3
		for i := range d2 { d2[i] = 0 }
		accelGEMVTrans(len(h3), len(h2), t.F3, d3, d2)
		// Grad wrt L2 pre-activation: d2 = d2_out * relu'(h2)
		for i := range d2 {
			if h2[i] <= 0 { d2[i] = 0 }
		}
		// Accumulate L3 weight/bias grads: G_F3 += d3 @ h2^T
		accelGER(len(h3), len(h2), 1.0, d3, h2, t.G_F3)
		for i, v := range d3 { t.G_B3[i] += v }
	} else {
		// Grad wrt L2 pre-activation
		for i := range d2 {
			if h2[i] > 0 {
				d2[i] = gradOut * t.F3[i]
			} else {
				d2[i] = 0
			}
		}
		// Accumulate output weights grad
		for i, v := range h2 {
			t.G_F3[i] += gradOut * v
		}
		t.G_B3[0] += gradOut // W.B3 is a single float
	}

	// Grad wrt L1 output: d1_out = F2^T @ d2
	for i := range d1 { d1[i] = 0 }
	accelGEMVTrans(len(h2), len(h1), t.F2, d2, d1)
	// Grad wrt L1 pre-activation: d1 = d1_out * relu'(h1)
	for i := range d1 {
		if h1[i] <= 0 { d1[i] = 0 }
	}

	// Accumulate L2 weight/bias grads
	accelGER(len(h2), len(h1), 1.0, d2, h1, t.G_F2)
	for i, v := range d2 { t.G_B2[i] += v }

	// Accumulate L1 weight/bias grads
	accelGER(len(h1), len(x), 1.0, d1, x, t.G_F1)
	for i, v := range d1 { t.G_B1[i] += v }
}

func (t *MLPTrainer) ZeroGrad() {
	zero := func(s []float32) {
		for i := range s { s[i] = 0 }
	}
	zero(t.G_F1)
	zero(t.G_B1)
	zero(t.G_F2)
	zero(t.G_B2)
	zero(t.G_F3)
	zero(t.G_B3)
	zero(t.G_F4)
	t.G_B4 = 0
}

func (t *MLPTrainer) Step(lr, wd float32, batchSize int) {
	t.T++
	beta1 := float32(0.9)
	beta2 := float32(0.999)
	eps := float32(1e-8)
	invN := 1.0 / float32(batchSize)

	update := func(w, g, m1, v1 []float32) {
		for i := range w {
			gi := g[i]*invN + wd*w[i]
			m1[i] = beta1*m1[i] + (1-beta1)*gi
			v1[i] = beta2*v1[i] + (1-beta2)*gi*gi

			mHat := m1[i] / (1 - float32(math.Pow(float64(beta1), float64(t.T))))
			vHat := v1[i] / (1 - float32(math.Pow(float64(beta2), float64(t.T))))
			w[i] -= lr * mHat / (float32(math.Sqrt(float64(vHat))) + eps)
		}
	}

	update(t.F1, t.G_F1, t.M1_F1, t.V1_F1)
	update(t.W.B1, t.G_B1, t.M1_B1, t.V1_B1)
	update(t.F2, t.G_F2, t.M1_F2, t.V1_F2)
	update(t.W.B2, t.G_B2, t.M1_B2, t.V1_B2)

	if t.W.Hidden3 > 0 {
		update(t.F3, t.G_F3, t.M1_F3, t.V1_F3)
		update(t.W.B3H, t.G_B3, t.M1_B3, t.V1_B3)
		update(t.F4, t.G_F4, t.M1_F4, t.V1_F4)
		// Update scalar B4
		gi := t.G_B4*invN + wd*t.W.B4
		t.M1_B4 = beta1*t.M1_B4 + (1-beta1)*gi
		t.V1_B4 = beta2*t.V1_B4 + (1-beta2)*gi*gi
		mHat := t.M1_B4 / (1 - float32(math.Pow(float64(beta1), float64(t.T))))
		vHat := t.V1_B4 / (1 - float32(math.Pow(float64(beta2), float64(t.T))))
		t.W.B4 -= lr * mHat / (float32(math.Sqrt(float64(vHat))) + eps)
	} else {
		update(t.F3, t.G_F3, t.M1_F3, t.V1_F3)
		// Update scalar B3
		gi := t.G_B3[0]*invN + wd*t.W.B3
		t.M1_B3[0] = beta1*t.M1_B3[0] + (1-beta1)*gi
		t.V1_B3[0] = beta2*t.V1_B3[0] + (1-beta2)*gi*gi
		mHat := t.M1_B3[0] / (1 - float32(math.Pow(float64(beta1), float64(t.T))))
		vHat := t.V1_B3[0] / (1 - float32(math.Pow(float64(beta2), float64(t.T))))
		t.W.B3 -= lr * mHat / (float32(math.Sqrt(float64(vHat))) + eps)
	}
}

func (t *MLPTrainer) Finalize() MLPWeights {
	unflatten := func(flat []float32, rows, cols int) [][]float32 {
		res := make([][]float32, rows)
		for i := 0; i < rows; i++ {
			res[i] = make([]float32, cols)
			copy(res[i], flat[i*cols:(i+1)*cols])
		}
		return res
	}

	res := t.W
	res.W1 = unflatten(t.F1, t.W.Hidden1, t.W.NFeatures)
	res.W2 = unflatten(t.F2, t.W.Hidden2, t.W.Hidden1)
	if t.W.Hidden3 > 0 {
		res.W3H = unflatten(t.F3, t.W.Hidden3, t.W.Hidden2)
		res.W4 = make([]float32, len(t.F4))
		copy(res.W4, t.F4)
	} else {
		res.W3 = make([]float32, len(t.F3))
		copy(res.W3, t.F3)
	}
	return res
}
