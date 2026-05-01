package ml

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/strategy"
)

// TestPipeline_BidCollectTrainScore is an end-to-end test covering the full bid pipeline:
//   collect bid data → write binary file → train a few epochs → finalize weights →
//   write to JSON → load as BidMLP → score a feature vector
func TestPipeline_BidCollectTrainScore(t *testing.T) {
	const (
		nHands    = 80
		rollouts  = 3
		batchSize = 16
		h1, h2    = 32, 16
	)

	// 1. Collect bid training data.
	rng := rand.New(rand.NewSource(42))
	gs := game.NewGame(0)
	var strats [6]game.Strategy
	for i := range strats {
		strats[i] = strategy.NewStandard(strategy.Balanced)
	}

	var allRows []BidCollectRow
	for i := 0; i < nHands; i++ {
		rows := CollectBidHand(i, gs, strats, strats, rng, rollouts)
		allRows = append(allRows, rows...)
		ReleaseBidRows(rows)
		gs.NextDealer()
	}
	if len(allRows) == 0 {
		t.Fatal("no rows collected")
	}

	// 2. Write rows to a binary temp file.
	dir := t.TempDir()
	dataPath := filepath.Join(dir, "bid_train.bin")
	f, err := os.Create(dataPath)
	if err != nil {
		t.Fatalf("create data file: %v", err)
	}
	for _, r := range allRows {
		if err := r.WriteBinary(f); err != nil {
			t.Fatalf("WriteBinary: %v", err)
		}
	}
	f.Close()

	// 3. Read rows back and verify round-trip.
	f2, err := os.Open(dataPath)
	if err != nil {
		t.Fatalf("open data file: %v", err)
	}
	var readBack BidCollectRow
	if err := readBack.ReadBinary(f2); err != nil {
		t.Fatalf("ReadBinary: %v", err)
	}
	f2.Close()
	if readBack.Seat != allRows[0].Seat {
		t.Errorf("round-trip Seat: got %d want %d", readBack.Seat, allRows[0].Seat)
	}
	if readBack.BidLevel != allRows[0].BidLevel {
		t.Errorf("round-trip BidLevel: got %d want %d", readBack.BidLevel, allRows[0].BidLevel)
	}

	// 4. Train for a few iterations.
	trainRng := rand.New(rand.NewSource(7))
	trainer := NewMLPTrainer(BidTotalLen, h1, h2, 0, trainRng)
	trainer.ResizeBatch(batchSize)

	preds := make([]float32, batchSize)
	for epoch := 0; epoch < 3; epoch++ {
		for start := 0; start+batchSize <= len(allRows); start += batchSize {
			batch := allRows[start : start+batchSize]
			for j, r := range batch {
				copy(trainer.BatchX[j*BidTotalLen:(j+1)*BidTotalLen], r.Features[:])
				trainer.BatchY[j] = r.ScoreDelta
			}
			trainer.ForwardBatch(batchSize, preds)
			trainer.BackwardBatch(batchSize, preds, trainer.BatchY[:batchSize])
			trainer.Step(0.001, 0.0, batchSize)
			trainer.ZeroGrad()
		}
	}

	// 5. Finalize weights and serialize to JSON.
	weights := trainer.Finalize()
	weights.NFeatures = BidTotalLen
	jsonData, err := json.Marshal(weights)
	if err != nil {
		t.Fatalf("marshal weights: %v", err)
	}
	weightsPath := filepath.Join(dir, "bid_model.json")
	if err := os.WriteFile(weightsPath, jsonData, 0o644); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	// 6. Load the weights as a BidMLP.
	model, err := LoadBidMLP(weightsPath)
	if err != nil {
		t.Fatalf("LoadBidMLP: %v", err)
	}

	// 7. Score a feature vector — should not panic, NaN, or Inf.
	var feat [BidTotalLen]float32
	for i := range feat {
		feat[i] = float32(i) * 0.01
	}
	score := model.Score(feat)
	if isNaNOrInf(score) {
		t.Errorf("model.Score returned %f", score)
	}
}

// TestPipeline_BidBinaryRoundTrip verifies WriteBinary/ReadBinary are exact inverses.
func TestPipeline_BidBinaryRoundTrip(t *testing.T) {
	orig := BidCollectRow{
		HandID:   42,
		Seat:     3,
		BidLevel: 5,
	}
	for i := range orig.Features {
		orig.Features[i] = float32(i) * 0.001
	}
	orig.ScoreDelta = -3.14

	var buf bytes.Buffer
	if err := orig.WriteBinary(&buf); err != nil {
		t.Fatalf("WriteBinary: %v", err)
	}

	var got BidCollectRow
	if err := got.ReadBinary(&buf); err != nil {
		t.Fatalf("ReadBinary: %v", err)
	}

	if got.HandID != orig.HandID {
		t.Errorf("HandID: got %d want %d", got.HandID, orig.HandID)
	}
	if got.Seat != orig.Seat {
		t.Errorf("Seat: got %d want %d", got.Seat, orig.Seat)
	}
	if got.BidLevel != orig.BidLevel {
		t.Errorf("BidLevel: got %d want %d", got.BidLevel, orig.BidLevel)
	}
	if got.ScoreDelta != orig.ScoreDelta {
		t.Errorf("ScoreDelta: got %f want %f", got.ScoreDelta, orig.ScoreDelta)
	}
	for i, f := range orig.Features {
		if got.Features[i] != f {
			t.Errorf("Features[%d]: got %f want %f", i, got.Features[i], f)
		}
	}
}

func isNaNOrInf(v float32) bool {
	return v != v || v > 1e38 || v < -1e38
}
