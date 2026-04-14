package ml

import (
	"math/rand"
	"testing"
)

func TestTrainerFunctional(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	nFeat, h1, h2 := 10, 8, 4
	trainer := NewMLPTrainer(nFeat, h1, h2, 0, rng)

	// Test 1: Forward pass produces non-zero output
	x := make([]float32, nFeat)
	for i := range x { x[i] = rng.Float32() }
	out := trainer.Forward(x)
	if out == 0 {
		t.Errorf("Forward produced 0, expected non-zero")
	}

	// Test 2: Backward pass doesn't crash and populates gradients
	trainer.ZeroGrad()
	trainer.Backward(x, 1.0)
	
	hasGrad := false
	for _, g := range trainer.G_F1 {
		if g != 0 { hasGrad = true; break }
	}
	if !hasGrad {
		t.Errorf("Backward did not produce gradients for F1")
	}

	// Test 3: Optimization step reduces loss on a single example
	initialPred := trainer.Forward(x)
	target := initialPred + 0.1
	
	for i := 0; i < 50; i++ {
		trainer.ZeroGrad()
		pred := trainer.Forward(x)
		diff := pred - target
		trainer.Backward(x, diff)
		trainer.Step(0.01, 0, 1) // Smaller LR, more steps
	}
	
	finalPred := trainer.Forward(x)
	initialDiff := mathAbs(initialPred - target)
	finalDiff := mathAbs(finalPred - target)
	if finalDiff >= initialDiff {
		t.Errorf("Trainer failed to reduce loss: initial_diff=%f, final_diff=%f, final_pred=%f, target=%f", 
			initialDiff, finalDiff, finalPred, target)
	}
}

func mathAbs(v float32) float32 {
	if v < 0 { return -v }
	return v
}

func BenchmarkTrainerStep(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	nFeat := 150 // Typical feature length
	h1, h2 := 128, 64
	trainer := NewMLPTrainer(nFeat, h1, h2, 0, rng)
	
	x := make([]float32, nFeat)
	for i := range x { x[i] = rng.Float32() }
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainer.Forward(x)
		trainer.Backward(x, 0.5)
		if i % 128 == 0 {
			trainer.Step(0.001, 1e-4, 128)
			trainer.ZeroGrad()
		}
	}
}
