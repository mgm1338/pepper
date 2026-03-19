package ml

import (
	"encoding/json"
	"fmt"
	"os"
)

// MLPWeights holds the serialized weights of a 37→H1→H2→1 MLP.
// Exported by train.py as JSON.
type MLPWeights struct {
	W1       [][]float32 `json:"w1"` // [H1][37]
	B1       []float32   `json:"b1"` // [H1]
	W2       [][]float32 `json:"w2"` // [H2][H1]
	B2       []float32   `json:"b2"` // [H2]
	W3       []float32   `json:"w3"` // [H2]  (output layer, single neuron)
	B3       float32     `json:"b3"`
	YMean    float32     `json:"y_mean"` // target mean used during normalization
	YStd     float32     `json:"y_std"`  // target std used during normalization
	NFeatures int        `json:"n_features"`
	Hidden1  int         `json:"hidden1"`
	Hidden2  int         `json:"hidden2"`
}

// MLP is a trained multi-layer perceptron for card-play score prediction.
// Predicts score_delta (bidding team's perspective) for a given feature vector.
type MLP struct {
	w  MLPWeights
	h1 []float32 // reusable scratch buffer
	h2 []float32 // reusable scratch buffer
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
	if len(w.W1) == 0 || len(w.W2) == 0 || len(w.W3) == 0 {
		return nil, fmt.Errorf("LoadMLP: empty weight arrays in %s", path)
	}
	return &MLP{
		w:  w,
		h1: make([]float32, len(w.B1)),
		h2: make([]float32, len(w.B2)),
	}, nil
}

// Clone returns a new MLP with the same weights but independent scratch buffers,
// safe to use concurrently with the original.
func (m *MLP) Clone() *MLP {
	return &MLP{
		w:  m.w,
		h1: make([]float32, len(m.w.B1)),
		h2: make([]float32, len(m.w.B2)),
	}
}

// Score runs a forward pass and returns the predicted score_delta
// (in original units, not normalized) for the given feature vector.
// Higher = better for the bidding team.
func (m *MLP) Score(features [TotalFeatureLen]float32) float32 {
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

	// Denormalize back to score_delta units
	return out*m.w.YStd + m.w.YMean
}
