package ml

import (
	"math"
	"math/rand"
	"testing"
)

// TestMLPScoreFlatMatchesJagged verifies that the flat-weight forward pass
// produces bit-identical results to the jagged reference implementation.
func TestMLPScoreFlatMatchesJagged(t *testing.T) {
	m, err := LoadMLP("../model_weights.json")
	if err != nil {
		t.Skip("model_weights.json not found:", err)
	}

	rng := rand.New(rand.NewSource(99))
	for trial := 0; trial < 1000; trial++ {
		var feat [TotalFeatureLen]float32
		for i := range feat {
			feat[i] = float32(rng.NormFloat64())
		}
		got := m.Score(feat)
		want := scoreJagged(m, feat)
		if math.Abs(float64(got-want)) > 1e-4 {
			t.Errorf("trial %d: Score()=%v jaggedScore()=%v diff=%v", trial, got, want, got-want)
		}
	}
}

// TestBidMLPScoreFlatMatchesJagged verifies the BidMLP flat forward pass is correct.
func TestBidMLPScoreFlatMatchesJagged(t *testing.T) {
	m, err := LoadBidMLP("../bid_model_weights.json")
	if err != nil {
		t.Skip("bid_model_weights.json not found:", err)
	}

	rng := rand.New(rand.NewSource(77))
	for trial := 0; trial < 1000; trial++ {
		var feat [BidTotalLen]float32
		for i := range feat {
			feat[i] = float32(rng.NormFloat64())
		}
		got := m.Score(feat)
		want := bidScoreJagged(m, feat)
		if math.Abs(float64(got-want)) > 1e-4 {
			t.Errorf("trial %d: Score()=%v jaggedScore()=%v diff=%v", trial, got, want, got-want)
		}
	}
}

// bidScoreJagged is the original jagged BidMLP forward pass kept as reference.
func bidScoreJagged(m *BidMLP, features [BidTotalLen]float32) float32 {
	h1 := make([]float32, len(m.w.B1))
	for i, bi := range m.w.B1 {
		v := bi
		for j := range features {
			v += m.w.W1[i][j] * features[j]
		}
		if v < 0 { v = 0 }
		h1[i] = v
	}
	h2 := make([]float32, len(m.w.B2))
	for i, bi := range m.w.B2 {
		v := bi
		for j, hj := range h1 {
			v += m.w.W2[i][j] * hj
		}
		if v < 0 { v = 0 }
		h2[i] = v
	}
	if len(m.w.B3H) > 0 {
		h3 := make([]float32, len(m.w.B3H))
		for i, bi := range m.w.B3H {
			v := bi
			for j, hj := range h2 {
				v += m.w.W3H[i][j] * hj
			}
			if v < 0 { v = 0 }
			h3[i] = v
		}
		out := m.w.B4
		for j, hj := range h3 {
			out += m.w.W4[j] * hj
		}
		return out*m.w.YStd + m.w.YMean
	}
	out := m.w.B3
	for j, hj := range h2 {
		out += m.w.W3[j] * hj
	}
	return out*m.w.YStd + m.w.YMean
}

// scoreJagged is the original jagged-slice forward pass, kept as a reference.
func scoreJagged(m *MLP, features [TotalFeatureLen]float32) float32 {
	h1 := make([]float32, len(m.w.B1))
	for i, bi := range m.w.B1 {
		v := bi
		for j := range features {
			v += m.w.W1[i][j] * features[j]
		}
		if v < 0 {
			v = 0
		}
		h1[i] = v
	}

	h2 := make([]float32, len(m.w.B2))
	for i, bi := range m.w.B2 {
		v := bi
		for j, hj := range h1 {
			v += m.w.W2[i][j] * hj
		}
		if v < 0 {
			v = 0
		}
		h2[i] = v
	}

	if len(m.w.B3H) > 0 {
		h3 := make([]float32, len(m.w.B3H))
		for i, bi := range m.w.B3H {
			v := bi
			for j, hj := range h2 {
				v += m.w.W3H[i][j] * hj
			}
			if v < 0 {
				v = 0
			}
			h3[i] = v
		}
		out := m.w.B4
		for j, hj := range h3 {
			out += m.w.W4[j] * hj
		}
		return out*m.w.YStd + m.w.YMean
	}

	out := m.w.B3
	for j, hj := range h2 {
		out += m.w.W3[j] * hj
	}
	return out*m.w.YStd + m.w.YMean
}
