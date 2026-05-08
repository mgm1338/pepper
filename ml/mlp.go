package ml

import (
	"encoding/json"
	"fmt"
	"os"
)

// MLPWeights holds the serialized weights of a 37→H1→H2→1 MLP (2-layer)
// or a 37→H1→H2→H3→1 MLP (3-layer). Exported by train.py as JSON.
// 3-layer models have non-empty W3H/B3H/W4/B4; 2-layer models use W3/B3 as output.
type MLPWeights struct {
	W1       [][]float32 `json:"w1"`  // [H1][feat]
	B1       []float32   `json:"b1"`  // [H1]
	W2       [][]float32 `json:"w2"`  // [H2][H1]
	B2       []float32   `json:"b2"`  // [H2]
	W3       []float32   `json:"w3"`  // [H2] output weights (2-layer only)
	B3       float32     `json:"b3"`  // output bias (2-layer only)
	// Optional 3rd hidden layer (present when hidden3 > 0)
	W3H      [][]float32 `json:"w3h"` // [H3][H2]
	B3H      []float32   `json:"b3h"` // [H3]
	W4       []float32   `json:"w4"`  // [H3] output weights (3-layer only)
	B4       float32     `json:"b4"`  // output bias (3-layer only)
	YMean      float32   `json:"y_mean"`             // target mean used during normalization
	YStd       float32   `json:"y_std"`              // target std used during normalization
	FeatureMean []float32 `json:"feat_mean,omitempty"` // per-feature input mean (nil = no normalization)
	FeatureStd  []float32 `json:"feat_std,omitempty"`  // per-feature input std
	NFeatures int        `json:"n_features"`
	Hidden1  int         `json:"hidden1"`
	Hidden2  int         `json:"hidden2"`
	Hidden3  int         `json:"hidden3"`
}

// MaxCardBatch is the maximum number of valid plays at any card-play decision point.
const MaxCardBatch = 8

// MLP is a trained multi-layer perceptron for card-play score prediction.
// Predicts score_delta (bidding team's perspective) for a given feature vector.
// Not safe for concurrent use — callers sharing across goroutines must Clone() first.
type MLP struct {
	w     MLPWeights
	h1    []float32 // reusable scratch buffer
	h2    []float32 // reusable scratch buffer
	h3    []float32 // reusable scratch buffer (nil for 2-layer models)
	// Flattened weight matrices for contiguous memory access (SIMD-friendly).
	flat1  []float32 // W1 row-major: [H1 * nFeat]
	flat2  []float32 // W2 row-major: [H2 * H1]
	flat3h []float32 // W3H row-major: [H3 * H2], nil for 2-layer models
	nFeat  int
	featBuf []float32 // heap-allocated copy of input features — avoids CGo escape

	// Batch scratch buffers for ScoreBatch (sized for MaxCardBatch candidates).
	batchH1Buf   []float32 // [MaxCardBatch * H1]
	batchH2Buf   []float32 // [MaxCardBatch * H2]
	batchH3Buf   []float32 // [MaxCardBatch * H3], nil for 2-layer models
	BatchFeatBuf []float32 // [MaxCardBatch * TotalFeatureLen] — reuse to avoid cgo heap escape
	BatchScoreBuf []float32 // [MaxCardBatch] — reuse to avoid per-call alloc
}

// flattenJagged converts a jagged [][]float32 matrix to a row-major []float32 slice.
func flattenJagged(mat [][]float32) []float32 {
	if len(mat) == 0 {
		return nil
	}
	cols := len(mat[0])
	flat := make([]float32, len(mat)*cols)
	for i, row := range mat {
		copy(flat[i*cols:], row)
	}
	return flat
}

// LoadMLP reads model weights from a JSON file written by train.py.
func LoadMLP(path string) (*MLP, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("LoadMLP: %w", err)
	}
	var w MLPWeights
	if err := json.Unmarshal(data, &w); err != nil {
		return nil, fmt.Errorf("LoadMLP parse: %w", err)
	}
	isThreeLayer := len(w.W3H) > 0
	if len(w.W1) == 0 || len(w.W2) == 0 || (!isThreeLayer && len(w.W3) == 0) || (isThreeLayer && len(w.W4) == 0) {
		return nil, fmt.Errorf("LoadMLP: empty weight arrays in %s", path)
	}
	flat1 := flattenJagged(w.W1)
	flat2 := flattenJagged(w.W2)
	H1 := len(w.B1)
	H2 := len(w.B2)
	nFeat := len(w.W1[0])
	m := &MLP{
		w:          w,
		h1:         make([]float32, H1),
		h2:         make([]float32, H2),
		flat1:      flat1,
		flat2:      flat2,
		nFeat:      nFeat,
		featBuf:    make([]float32, nFeat),
		batchH1Buf:    make([]float32, MaxCardBatch*H1),
		batchH2Buf:    make([]float32, MaxCardBatch*H2),
		BatchFeatBuf:  make([]float32, MaxCardBatch*TotalFeatureLen),
		BatchScoreBuf: make([]float32, MaxCardBatch),
	}
	if len(w.B3H) > 0 {
		m.h3 = make([]float32, len(w.B3H))
		m.flat3h = flattenJagged(w.W3H)
		m.batchH3Buf = make([]float32, MaxCardBatch*len(w.B3H))
	}
	return m, nil
}

// Weights returns the MLPWeights for use with MLPTrainer.LoadWeights.
func (m *MLP) Weights() MLPWeights { return m.w }

// Clone returns a new MLP with the same weights but independent scratch buffers,
// safe to use concurrently with the original.
func (m *MLP) Clone() *MLP {
	H1 := len(m.w.B1)
	H2 := len(m.w.B2)
	c := &MLP{
		w:          m.w,
		h1:         make([]float32, H1),
		h2:         make([]float32, H2),
		flat1:      m.flat1,  // shared read-only
		flat2:      m.flat2,  // shared read-only
		flat3h:     m.flat3h, // shared read-only
		nFeat:      m.nFeat,
		featBuf:    make([]float32, m.nFeat),
		batchH1Buf:    make([]float32, MaxCardBatch*H1),
		batchH2Buf:    make([]float32, MaxCardBatch*H2),
		BatchFeatBuf:  make([]float32, MaxCardBatch*TotalFeatureLen),
		BatchScoreBuf: make([]float32, MaxCardBatch),
	}
	if len(m.w.B3H) > 0 {
		H3 := len(m.w.B3H)
		c.h3 = make([]float32, H3)
		c.batchH3Buf = make([]float32, MaxCardBatch*H3)
	}
	return c
}

// ScoreBatch scores n card feature vectors in a single cgo call (all layers fused).
// featFlat must be n*TotalFeatureLen elements (row-major). Normalized in-place when
// FeatureMean is set — callers must use MLP-owned buffers (BatchFeatBuf / featBuf).
// out must have capacity >= n.
func (m *MLP) ScoreBatch(n int, featFlat []float32, out []float32) {
	if len(m.w.FeatureMean) > 0 {
		nf := m.nFeat
		for i := 0; i < n; i++ {
			row := featFlat[i*nf : (i+1)*nf]
			for j, mean := range m.w.FeatureMean {
				row[j] = (row[j] - mean) / (m.w.FeatureStd[j] + 1e-8)
			}
		}
	}
	H1 := len(m.h1)
	H2 := len(m.h2)
	H3 := 0
	var w3h, b3h, h3Buf []float32
	if m.h3 != nil {
		H3 = len(m.h3)
		w3h = m.flat3h
		b3h = m.w.B3H
		h3Buf = m.batchH3Buf[:n*H3]
	}
	wOut, bOut := m.w.W3, m.w.B3
	if H3 > 0 {
		wOut, bOut = m.w.W4, m.w.B4
	}
	AccelMLPForwardBatch(
		n, m.nFeat, H1, H2, H3,
		featFlat, m.flat1, m.w.B1, m.flat2, m.w.B2,
		w3h, b3h, wOut, bOut,
		m.w.YStd, m.w.YMean,
		m.batchH1Buf[:n*H1], m.batchH2Buf[:n*H2], h3Buf,
		out[:n],
	)
}

// Score runs a forward pass and returns the predicted score_delta
// (in original units, not normalized) for the given feature vector.
// Uses the fused batch path (single cgo call) for lower overhead.
func (m *MLP) Score(features [TotalFeatureLen]float32) float32 {
	copy(m.featBuf, features[:])
	m.ScoreBatch(1, m.featBuf, m.BatchScoreBuf[:1])
	return m.BatchScoreBuf[0]
}

func (m *MLP) scoreGoFallback(features [TotalFeatureLen]float32) float32 {
	nFeat := m.nFeat
	flat1 := m.flat1
	flat2 := m.flat2
	H1 := len(m.h1)
	{
		i := 0
		for ; i+7 < H1; i += 8 {
			off0 := i * nFeat
			v0 := m.w.B1[i]
			v1 := m.w.B1[i+1]
			v2 := m.w.B1[i+2]
			v3 := m.w.B1[i+3]
			v4 := m.w.B1[i+4]
			v5 := m.w.B1[i+5]
			v6 := m.w.B1[i+6]
			v7 := m.w.B1[i+7]
			r0 := flat1[off0 : off0+nFeat]
			r1 := flat1[off0+nFeat : off0+2*nFeat]
			r2 := flat1[off0+2*nFeat : off0+3*nFeat]
			r3 := flat1[off0+3*nFeat : off0+4*nFeat]
			r4 := flat1[off0+4*nFeat : off0+5*nFeat]
			r5 := flat1[off0+5*nFeat : off0+6*nFeat]
			r6 := flat1[off0+6*nFeat : off0+7*nFeat]
			r7 := flat1[off0+7*nFeat : off0+8*nFeat]
			for j, fj := range features {
				v0 += r0[j] * fj
				v1 += r1[j] * fj
				v2 += r2[j] * fj
				v3 += r3[j] * fj
				v4 += r4[j] * fj
				v5 += r5[j] * fj
				v6 += r6[j] * fj
				v7 += r7[j] * fj
			}
			if v0 < 0 { v0 = 0 }
			if v1 < 0 { v1 = 0 }
			if v2 < 0 { v2 = 0 }
			if v3 < 0 { v3 = 0 }
			if v4 < 0 { v4 = 0 }
			if v5 < 0 { v5 = 0 }
			if v6 < 0 { v6 = 0 }
			if v7 < 0 { v7 = 0 }
			m.h1[i] = v0
			m.h1[i+1] = v1
			m.h1[i+2] = v2
			m.h1[i+3] = v3
			m.h1[i+4] = v4
			m.h1[i+5] = v5
			m.h1[i+6] = v6
			m.h1[i+7] = v7
		}
		for ; i < H1; i++ {
			off := i * nFeat
			v := m.w.B1[i]
			for j, fj := range features {
				v += flat1[off+j] * fj
			}
			if v < 0 { v = 0 }
			m.h1[i] = v
		}
	}

	// h2 = relu(W2 @ h1 + b2)  — flat2 is row-major [H2 * H1]
	H2 := len(m.h2)
	{
		i := 0
		for ; i+7 < H2; i += 8 {
			off0 := i * H1
			v0 := m.w.B2[i]
			v1 := m.w.B2[i+1]
			v2 := m.w.B2[i+2]
			v3 := m.w.B2[i+3]
			v4 := m.w.B2[i+4]
			v5 := m.w.B2[i+5]
			v6 := m.w.B2[i+6]
			v7 := m.w.B2[i+7]
			r0 := flat2[off0 : off0+H1]
			r1 := flat2[off0+H1 : off0+2*H1]
			r2 := flat2[off0+2*H1 : off0+3*H1]
			r3 := flat2[off0+3*H1 : off0+4*H1]
			r4 := flat2[off0+4*H1 : off0+5*H1]
			r5 := flat2[off0+5*H1 : off0+6*H1]
			r6 := flat2[off0+6*H1 : off0+7*H1]
			r7 := flat2[off0+7*H1 : off0+8*H1]
			for j, hj := range m.h1 {
				v0 += r0[j] * hj
				v1 += r1[j] * hj
				v2 += r2[j] * hj
				v3 += r3[j] * hj
				v4 += r4[j] * hj
				v5 += r5[j] * hj
				v6 += r6[j] * hj
				v7 += r7[j] * hj
			}
			if v0 < 0 { v0 = 0 }
			if v1 < 0 { v1 = 0 }
			if v2 < 0 { v2 = 0 }
			if v3 < 0 { v3 = 0 }
			if v4 < 0 { v4 = 0 }
			if v5 < 0 { v5 = 0 }
			if v6 < 0 { v6 = 0 }
			if v7 < 0 { v7 = 0 }
			m.h2[i] = v0
			m.h2[i+1] = v1
			m.h2[i+2] = v2
			m.h2[i+3] = v3
			m.h2[i+4] = v4
			m.h2[i+5] = v5
			m.h2[i+6] = v6
			m.h2[i+7] = v7
		}
		for ; i < H2; i++ {
			off := i * H1
			v := m.w.B2[i]
			for j, hj := range m.h1 {
				v += flat2[off+j] * hj
			}
			if v < 0 { v = 0 }
			m.h2[i] = v
		}
	}

	if m.h3 != nil {
		// h3 = relu(W3H @ h2 + b3h)  — flat3h is row-major [H3 * H2]
		flat3h := m.flat3h
		off := 0
		for i, bi := range m.w.B3H {
			v := bi
			row := flat3h[off : off+H2]
			for j, hj := range m.h2 {
				v += row[j] * hj
			}
			if v < 0 {
				v = 0
			}
			m.h3[i] = v
			off += H2
		}
		out := m.w.B4
		for j, hj := range m.h3 {
			out += m.w.W4[j] * hj
		}
		return out*m.w.YStd + m.w.YMean
	}

	// out = W3 @ h2 + b3  (normalized)
	out := m.w.B3
	for j, hj := range m.h2 {
		out += m.w.W3[j] * hj
	}
	return out*m.w.YStd + m.w.YMean
}
