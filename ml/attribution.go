package ml

// Attribution computes gradient×input feature attribution for MLP forward passes.
// For ReLU networks this is exact: each value is the contribution of that feature
// to the final score in the same units as score_delta.
// Positive = feature pushed the score up. Negative = pushed it down.

// attributeWeights performs a forward pass saving ReLU gate masks, then
// backpropagates to compute gradient×input for each input feature.
// flat1: W1 row-major [H1*nFeat], flat2: W2 row-major [H2*H1],
// flat3h: W3H row-major [H3*H2] (nil for 2-layer), features: input slice.
// Returns gradient×input×YStd (attribution in score_delta units).
func attributeWeights(w MLPWeights, flat1, flat2, flat3h []float32, nFeat int, features []float32) []float32 {
	H1 := len(w.B1)
	H2 := len(w.B2)

	// Forward pass layer 1.
	pre1 := make([]float32, H1)
	copy(pre1, w.B1)
	for i := 0; i < H1; i++ {
		off := i * nFeat
		row := flat1[off : off+nFeat]
		for j, fj := range features[:nFeat] {
			pre1[i] += row[j] * fj
		}
	}
	h1 := make([]float32, H1)
	gate1 := make([]float32, H1)
	for i, v := range pre1 {
		if v > 0 {
			h1[i] = v
			gate1[i] = 1
		}
	}

	// Forward pass layer 2.
	pre2 := make([]float32, H2)
	copy(pre2, w.B2)
	for i := 0; i < H2; i++ {
		off := i * H1
		row := flat2[off : off+H1]
		for j, hj := range h1 {
			pre2[i] += row[j] * hj
		}
	}
	gate2 := make([]float32, H2)
	h2 := make([]float32, H2)
	for i, v := range pre2 {
		if v > 0 {
			h2[i] = v
			gate2[i] = 1
		}
	}

	// Backprop: compute gradient at input.
	grad := make([]float32, nFeat)

	if flat3h != nil {
		// 3-layer: forward pass layer 3.
		H3 := len(w.B3H)
		pre3 := make([]float32, H3)
		copy(pre3, w.B3H)
		for i := 0; i < H3; i++ {
			off := i * H2
			row := flat3h[off : off+H2]
			for j, hj := range h2 {
				pre3[i] += row[j] * hj
			}
		}
		gate3 := make([]float32, H3)
		for i, v := range pre3 {
			if v > 0 {
				gate3[i] = 1
			}
		}

		// delta3 = W4 * gate3
		delta3 := make([]float32, H3)
		for i := range delta3 {
			delta3[i] = w.W4[i] * gate3[i]
		}

		// delta2 = W3H^T @ delta3 * gate2
		delta2 := make([]float32, H2)
		for i := 0; i < H2; i++ {
			if gate2[i] == 0 {
				continue
			}
			var sum float32
			for k := 0; k < H3; k++ {
				sum += flat3h[k*H2+i] * delta3[k]
			}
			delta2[i] = sum
		}

		// delta1 = W2^T @ delta2 * gate1
		delta1 := make([]float32, H1)
		for i := 0; i < H1; i++ {
			if gate1[i] == 0 {
				continue
			}
			var sum float32
			for k := 0; k < H2; k++ {
				sum += flat2[k*H1+i] * delta2[k]
			}
			delta1[i] = sum
		}

		// grad = W1^T @ delta1
		for j := 0; j < nFeat; j++ {
			var sum float32
			for i := 0; i < H1; i++ {
				sum += flat1[i*nFeat+j] * delta1[i]
			}
			grad[j] = sum * features[j] * w.YStd
		}
	} else {
		// 2-layer: delta2 = W3 * gate2
		delta2 := make([]float32, H2)
		for i := range delta2 {
			delta2[i] = w.W3[i] * gate2[i]
		}

		// delta1 = W2^T @ delta2 * gate1
		delta1 := make([]float32, H1)
		for i := 0; i < H1; i++ {
			if gate1[i] == 0 {
				continue
			}
			var sum float32
			for k := 0; k < H2; k++ {
				sum += flat2[k*H1+i] * delta2[k]
			}
			delta1[i] = sum
		}

		// grad = W1^T @ delta1
		for j := 0; j < nFeat; j++ {
			var sum float32
			for i := 0; i < H1; i++ {
				sum += flat1[i*nFeat+j] * delta1[i]
			}
			grad[j] = sum * features[j] * w.YStd
		}
	}

	return grad
}

// FeatureAttribution holds one feature's name, raw model input value, and impact.
type FeatureAttribution struct {
	Feature string  // human-readable feature name
	Value   float32 // raw value seen by the model (normalized)
	Impact  float32 // gradient×input in score_delta units
}

// AttributePlay computes gradient×input attribution for a card play feature vector.
// isBidder should be true when the player is on the bidding team — flips sign so
// attribution is always from that player's perspective (positive = helped them).
func (m *MLP) AttributePlay(features [TotalFeatureLen]float32, isBidder bool) []FeatureAttribution {
	grad := attributeWeights(m.w, m.flat1, m.flat2, m.flat3h, m.nFeat, features[:])
	sign := float32(1)
	if !isBidder {
		sign = -1
	}
	out := make([]FeatureAttribution, TotalFeatureLen)
	for i, g := range grad {
		out[i] = FeatureAttribution{
			Feature: FeatureNames[i],
			Value:   features[i],
			Impact:  g * sign,
		}
	}
	return out
}

// AttributeBid computes gradient×input attribution for a bid feature vector.
func (m *BidMLP) AttributeBid(features [BidTotalLen]float32) []FeatureAttribution {
	grad := attributeWeights(m.w, m.flat1, m.flat2, m.flat3h, m.nFeat, features[:])
	out := make([]FeatureAttribution, BidTotalLen)
	for i, g := range grad {
		out[i] = FeatureAttribution{
			Feature: BidFeatureNames[i],
			Value:   features[i],
			Impact:  g,
		}
	}
	return out
}
