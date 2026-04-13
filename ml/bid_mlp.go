package ml

import (
	"encoding/json"
	"fmt"
	"os"
)

// BidMLP is a trained multi-layer perceptron for bid decision scoring.
type BidMLP struct {
	w       MLPWeights
	h1      []float32
	h2      []float32
	h3      []float32 // nil for 2-layer models
	flat1   []float32 // W1 row-major: [H1 * nFeat]
	flat2   []float32 // W2 row-major: [H2 * H1]
	flat3h  []float32 // W3H row-major: [H3 * H2], nil for 2-layer models
	nFeat   int
	featBuf []float32 // heap-allocated copy of input features — avoids CGo escape
}

func LoadBidMLP(path string) (*BidMLP, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("LoadBidMLP: %w", err)
	}
	var w MLPWeights
	if err := json.Unmarshal(data, &w); err != nil {
		return nil, fmt.Errorf("LoadBidMLP parse: %w", err)
	}
	isThreeLayer := len(w.W3H) > 0
	if len(w.W1) == 0 || len(w.W2) == 0 || (!isThreeLayer && len(w.W3) == 0) || (isThreeLayer && len(w.W4) == 0) {
		return nil, fmt.Errorf("LoadBidMLP: empty weight arrays in %s", path)
	}
	if w.NFeatures != BidTotalLen {
		return nil, fmt.Errorf("LoadBidMLP: model expects %d features, bid vector is %d", w.NFeatures, BidTotalLen)
	}
	m := &BidMLP{
		w:       w,
		h1:      make([]float32, len(w.B1)),
		h2:      make([]float32, len(w.B2)),
		flat1:   flattenJagged(w.W1),
		flat2:   flattenJagged(w.W2),
		nFeat:   len(w.W1[0]),
		featBuf: make([]float32, len(w.W1[0])),
	}
	if len(w.B3H) > 0 {
		m.h3 = make([]float32, len(w.B3H))
		m.flat3h = flattenJagged(w.W3H)
	}
	return m, nil
}

func (m *BidMLP) Score(features [BidTotalLen]float32) float32 {
	H1 := len(m.h1)
	H2 := len(m.h2)

	copy(m.featBuf, features[:])

	// h1 = relu(W1 @ features + b1)
	copy(m.h1, m.w.B1)
	accelGEMV(H1, m.nFeat, m.flat1, m.featBuf, m.h1)
	for i, v := range m.h1 {
		if v < 0 {
			m.h1[i] = 0
		}
	}

	// h2 = relu(W2 @ h1 + b2)
	copy(m.h2, m.w.B2)
	accelGEMV(H2, H1, m.flat2, m.h1, m.h2)
	for i, v := range m.h2 {
		if v < 0 {
			m.h2[i] = 0
		}
	}

	if m.h3 != nil {
		H3 := len(m.h3)
		copy(m.h3, m.w.B3H)
		accelGEMV(H3, H2, m.flat3h, m.h2, m.h3)
		for i, v := range m.h3 {
			if v < 0 {
				m.h3[i] = 0
			}
		}
		out := m.w.B4
		for j, hj := range m.h3 {
			out += m.w.W4[j] * hj
		}
		return out*m.w.YStd + m.w.YMean
	}

	out := m.w.B3
	for j, hj := range m.h2 {
		out += m.w.W3[j] * hj
	}
	return out*m.w.YStd + m.w.YMean
}

func (m *BidMLP) scoreGoFallback(features [BidTotalLen]float32) float32 {
	nFeat := m.nFeat
	flat1 := m.flat1
	flat2 := m.flat2
	H1 := len(m.h1)
	H2 := len(m.h2)

	// h1 = relu(W1 @ features + b1) — 8-wide unroll
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

	// h2 = relu(W2 @ h1 + b2) — 8-wide unroll
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
		flat3h := m.flat3h
		off := 0
		for i, bi := range m.w.B3H {
			v := bi
			row := flat3h[off : off+H2]
			for j, hj := range m.h2 {
				v += row[j] * hj
			}
			if v < 0 { v = 0 }
			m.h3[i] = v
			off += H2
		}
		out := m.w.B4
		for j, hj := range m.h3 {
			out += m.w.W4[j] * hj
		}
		return out*m.w.YStd + m.w.YMean
	}

	out := m.w.B3
	for j, hj := range m.h2 {
		out += m.w.W3[j] * hj
	}
	return out*m.w.YStd + m.w.YMean
}
