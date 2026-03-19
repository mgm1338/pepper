package ml

import (
	"encoding/json"
	"fmt"
	"os"
)

// BidMLP is a trained multi-layer perceptron for bid decision scoring.
// Predicts expected score delta for a given (hand context, bid level) pair,
// from the perspective of the seat making the bid decision.
//
// Uses the same MLPWeights JSON format as the card-play MLP (exported by train.py),
// but operates on BidTotalLen features instead of TotalFeatureLen.
type BidMLP struct {
	w  MLPWeights
	h1 []float32
	h2 []float32
}

// LoadBidMLP reads bid model weights from a JSON file written by train.py.
func LoadBidMLP(path string) (*BidMLP, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("LoadBidMLP: %w", err)
	}
	var w MLPWeights
	if err := json.Unmarshal(data, &w); err != nil {
		return nil, fmt.Errorf("LoadBidMLP parse: %w", err)
	}
	if len(w.W1) == 0 || len(w.W2) == 0 || len(w.W3) == 0 {
		return nil, fmt.Errorf("LoadBidMLP: empty weight arrays in %s", path)
	}
	if w.NFeatures != BidTotalLen {
		return nil, fmt.Errorf("LoadBidMLP: model expects %d features, bid vector is %d", w.NFeatures, BidTotalLen)
	}
	return &BidMLP{
		w:  w,
		h1: make([]float32, len(w.B1)),
		h2: make([]float32, len(w.B2)),
	}, nil
}

// Score runs a forward pass and returns the predicted expected score delta
// for the bidding seat's team for the given bid feature vector.
func (m *BidMLP) Score(features [BidTotalLen]float32) float32 {
	// h1 = relu(W1 @ features + b1)
	for i, bi := range m.w.B1 {
		v := bi
		for j := range features {
			v += m.w.W1[i][j] * features[j]
		}
		if v < 0 {
			v = 0
		}
		m.h1[i] = v
	}

	// h2 = relu(W2 @ h1 + b2)
	for i, bi := range m.w.B2 {
		v := bi
		for j, hj := range m.h1 {
			v += m.w.W2[i][j] * hj
		}
		if v < 0 {
			v = 0
		}
		m.h2[i] = v
	}

	// out = W3 @ h2 + b3  (normalized)
	out := m.w.B3
	for j, hj := range m.h2 {
		out += m.w.W3[j] * hj
	}

	// Denormalize back to score_delta units.
	return out*m.w.YStd + m.w.YMean
}
