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

// TestBidMLPScoreBatchMatchesScore verifies that ScoreBatch produces the same results as Score.
func TestBidMLPScoreBatchMatchesScore(t *testing.T) {
	m, err := LoadBidMLP("../bid_model_weights.json")
	if err != nil {
		t.Skip("bid_model_weights.json not found:", err)
	}
	mc := m.Clone()

	rng := rand.New(rand.NewSource(42))
	for trial := 0; trial < 200; trial++ {
		n := rng.Intn(MaxBidBatch) + 1
		var featFlat [MaxBidBatch * BidTotalLen]float32
		for i := range featFlat[:n*BidTotalLen] {
			featFlat[i] = float32(rng.NormFloat64())
		}

		var batchOut [MaxBidBatch]float32
		mc.ScoreBatch(n, featFlat[:n*BidTotalLen], batchOut[:n])

		for i := 0; i < n; i++ {
			var feat [BidTotalLen]float32
			copy(feat[:], featFlat[i*BidTotalLen:])
			want := m.Score(feat)
			if math.Abs(float64(batchOut[i]-want)) > 1e-3 {
				t.Errorf("trial %d bid %d: ScoreBatch=%v Score=%v diff=%v", trial, i, batchOut[i], want, batchOut[i]-want)
			}
		}
	}
}

// BenchmarkBidMLPScoreSequential benchmarks scoring MaxBidBatch candidates one at a time.
func BenchmarkBidMLPScoreSequential(b *testing.B) {
	m, err := LoadBidMLP("../bid_model_weights.json")
	if err != nil {
		b.Skip("bid_model_weights.json not found:", err)
	}
	rng := rand.New(rand.NewSource(1))
	var feats [MaxBidBatch][BidTotalLen]float32
	for i := range feats {
		for j := range feats[i] {
			feats[i][j] = float32(rng.NormFloat64())
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, f := range feats {
			_ = m.Score(f)
		}
	}
}

// BenchmarkBidMLPScoreBatch benchmarks scoring MaxBidBatch candidates via ScoreBatch.
func BenchmarkBidMLPScoreBatch(b *testing.B) {
	m, err := LoadBidMLP("../bid_model_weights.json")
	if err != nil {
		b.Skip("bid_model_weights.json not found:", err)
	}
	rng := rand.New(rand.NewSource(1))
	var featFlat [MaxBidBatch * BidTotalLen]float32
	for i := range featFlat {
		featFlat[i] = float32(rng.NormFloat64())
	}
	var out [MaxBidBatch]float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.ScoreBatch(MaxBidBatch, featFlat[:], out[:])
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

// BenchmarkMLPScoreBatch benchmarks scoring MaxCardBatch cards via ScoreBatch.
func BenchmarkMLPScoreBatch(b *testing.B) {
	m, err := LoadMLP("../model_weights.json")
	if err != nil {
		b.Skip("model_weights.json not found:", err)
	}
	rng := rand.New(rand.NewSource(1))
	var featFlat [MaxCardBatch * TotalFeatureLen]float32
	for i := range featFlat {
		featFlat[i] = float32(rng.NormFloat64())
	}
	var out [MaxCardBatch]float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.ScoreBatch(MaxCardBatch, featFlat[:], out[:])
	}
}

// BenchmarkMLPScoreSequentialN benchmarks scoring MaxCardBatch cards one at a time.
func BenchmarkMLPScoreSequentialN(b *testing.B) {
	m, err := LoadMLP("../model_weights.json")
	if err != nil {
		b.Skip("model_weights.json not found:", err)
	}
	rng := rand.New(rand.NewSource(1))
	var feats [MaxCardBatch][TotalFeatureLen]float32
	for i := range feats {
		for j := range feats[i] {
			feats[i][j] = float32(rng.NormFloat64())
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, f := range feats {
			_ = m.Score(f)
		}
	}
}
