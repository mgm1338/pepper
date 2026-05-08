package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/max/pepper/ml"
)

func main() {
	dataPath   := flag.String("data", "collect.csv", "Input file from cmd/collect")
	format     := flag.String("format", "csv", "Input format: csv or bin")
	outPath    := flag.String("out", "model_weights.json", "Output weights JSON")
	warmStart      := flag.String("warm-start", "", "path to JSON weights to warm-start from (skips random init)")
	warmHiddenOnly := flag.Bool("warm-hidden-only", false, "warm-start hidden layers only, reset output layer")
	epochs     := flag.Int("epochs", 100, "Max training epochs")
	batch      := flag.Int("batch", 4096, "Mini-batch size")
	lr         := flag.Float64("lr", 1e-3, "Adam learning rate")
	valFrac    := flag.Float64("val", 0.1, "Validation fraction")
	h1         := flag.Int("h1", 128, "Hidden layer 1 size")
	h2         := flag.Int("h2", 64, "Hidden layer 2 size")
	h3         := flag.Int("h3", 0, "Hidden layer 3 size (0=2-layer)")
	wd         := flag.Float64("wd", 1e-4, "L2 weight decay")
	patience   := flag.Int("patience", 5, "ReduceLROnPlateau patience")
	minLR      := flag.Float64("min-lr", 1e-6, "Stop when LR drops below this")
	minDelta   := flag.Float64("min-delta", 0.0, "Minimum relative MSE improvement to reset patience (e.g. 0.001)")
	target     := flag.String("target", "score_delta", "Target column")
	seed       := flag.Int64("seed", 42, "Random seed")
	mode        := flag.String("mode", "card", "Training mode: card or bid")
	lrSchedule  := flag.String("lr-schedule", "plateau", "LR schedule: plateau or cosine")
	cosineT0    := flag.Int("cosine-t0", 50, "Cosine annealing initial cycle length (epochs)")
	cosineTMult := flag.Float64("cosine-tmult", 2.0, "Cosine annealing cycle length multiplier per restart")
	replayData   := flag.String("replay-data", "", "comma-separated extra data files for experience replay")
	shuffleBuf   := flag.Int("shuffle-buf", 0, "Shuffle buffer size for streaming training (0=disabled, recommend 65536)")
	huberDelta   := flag.Float64("huber-delta", 0, "Huber loss delta (0=MSE)")
	replayWeight := flag.Float64("replay-weight", 0.5, "Loss weight for replay-data rows (0=ignore, 1=full weight)")
	normFeatures := flag.Bool("norm-features", false, "Compute and apply per-feature input normalization")
	flag.Parse()

	rng := rand.New(rand.NewSource(*seed))

	dataPaths := []string{*dataPath}
	if *replayData != "" {
		for _, p := range strings.Split(*replayData, ",") {
			if p = strings.TrimSpace(p); p != "" {
				dataPaths = append(dataPaths, p)
			}
		}
	}

	if *format == "bin" && *mode == "bid" {
		runBidBinaryStreaming(rng, dataPaths, *outPath, *warmStart, *warmHiddenOnly, *lrSchedule, *cosineT0, *cosineTMult, *h1, *h2, *h3,
			*batch, *epochs, *patience, *valFrac,
			float32(*lr), float32(*wd), float32(*minLR), float32(*minDelta),
			*shuffleBuf, float32(*huberDelta), float32(*replayWeight), *normFeatures)
	} else if *format == "bin" {
		runBinaryStreaming(rng, dataPaths, *outPath, *warmStart, *target, *lrSchedule, *cosineT0, *cosineTMult, *h1, *h2, *h3,
			*batch, *epochs, *patience, *valFrac,
			float32(*lr), float32(*wd), float32(*minLR), float32(*minDelta),
			*shuffleBuf, float32(*huberDelta), float32(*replayWeight), *normFeatures)
	} else {
		runCSV(rng, *dataPath, *outPath, *target, *h1, *h2, *h3,
			*batch, *epochs, *patience, *valFrac,
			float32(*lr), float32(*wd), float32(*minLR), float32(*minDelta))
	}
}

// runBinaryStreaming handles binary format with streaming (no full data in memory).
// Pass 1: compute stats. Each epoch: re-read each file, stream batches.
func runBinaryStreaming(rng *rand.Rand, dataPaths []string, outPath, warmStartPath, target, lrSchedule string, cosineT0 int, cosineTMult float64,
	h1, h2, h3, batch, epochs, patience int, valFrac float64,
	lr, wd, minLR, minDelta float32,
	shuffleBufSize int, huberDelta, replayWeight float32, normFeatures bool) {

	nFeat := ml.TotalFeatureLen
	fmt.Printf("Loading %v (format=bin, streaming) ...\n", dataPaths)
	t0 := time.Now()

	// --- Pass 1: count rows, compute target stats, compute per-feature stats ---
	var nRows int64
	var tSum, tSum2 float64
	featSum  := make([]float64, nFeat)
	featSum2 := make([]float64, nFeat)
	for _, path := range dataPaths {
		f, err := os.Open(path)
		if err != nil { log.Fatal(err) }
		rd := bufio.NewReaderSize(f, 1<<20)
		var row ml.CollectRow
		for {
			err := row.ReadBinary(rd)
			if err == io.EOF { break }
			if err != nil { log.Fatal(err) }
			var tv float32
			if target == "made_bid_rate" {
				tv = row.MadeBidRate
			} else {
				tv = row.ScoreDelta
			}
			tSum += float64(tv)
			tSum2 += float64(tv) * float64(tv)
			for j, fv := range row.Features {
				featSum[j]  += float64(fv)
				featSum2[j] += float64(fv) * float64(fv)
			}
			nRows++
		}
		f.Close()
	}

	featMean := make([]float32, nFeat)
	featStd  := make([]float32, nFeat)
	for j := 0; j < nFeat; j++ {
		m := featSum[j] / float64(nRows)
		v := featSum2[j]/float64(nRows) - m*m
		if v < 0 { v = 0 }
		featMean[j] = float32(m)
		featStd[j]  = float32(math.Sqrt(v))
	}

	yMeanF64 := tSum / float64(nRows)
	yStdF64 := math.Sqrt(tSum2/float64(nRows) - yMeanF64*yMeanF64)
	yMean := float32(yMeanF64)
	yStd := float32(yStdF64)

	nVal := int64(float64(nRows) * valFrac)
	nTrain := nRows - nVal

	fmt.Printf("  %12d rows total (%v)\n", nRows, time.Since(t0).Round(time.Millisecond))
	fmt.Printf("  %d features  target=%s\n", nFeat, target)
	fmt.Printf("  mean=%.4f  std=%.4f\n", yMean, yStd)
	fmt.Printf("  train=%10d  val=%10d\n\n", nTrain, nVal)

	// --- Setup trainer ---
	trainer := ml.NewMLPTrainer(nFeat, h1, h2, h3, rng)
	trainer.ResizeBatch(batch)
	trainer.W.YMean = yMean
	trainer.W.YStd = yStd
	trainer.HuberDelta = huberDelta
	if normFeatures {
		trainer.W.FeatureMean = featMean
		trainer.W.FeatureStd  = featStd
	}

	if warmStartPath != "" {
		w, err := ml.LoadMLP(warmStartPath)
		if err != nil {
			log.Printf("warm start load failed (%v) — training from scratch", err)
		} else if wsErr := trainer.LoadWeights(w.Weights()); wsErr != nil {
			log.Printf("warm start skipped: %v", wsErr)
		} else {
			fmt.Printf("  Warm start: %s\n\n", warmStartPath)
		}
	}

	bestValLoss := float32(math.Inf(1))
	var bestWeights ml.MLPWeights

	currentLR := lr
	plateauCount := 0
	cosineMode := lrSchedule == "cosine"
	cosineCurEpoch := 0
	cosineCycleLen := cosineT0
	preds := make([]float32, batch)

	fmt.Printf("Model:  %d → %d → %d", nFeat, h1, h2)
	if h3 > 0 { fmt.Printf(" → %d", h3) }
	nParams := nFeat*h1 + h1 + h1*h2 + h2
	if h3 > 0 {
		nParams += h2*h3 + h3 + h3 + 1
	} else {
		nParams += h2 + 1
	}
	fmt.Printf(" → 1  (%d params)\n", nParams)
	extra := ""
	if shuffleBufSize > 0 { extra += fmt.Sprintf("  shuffle-buf=%d", shuffleBufSize) }
	if huberDelta > 0     { extra += fmt.Sprintf("  huber=%.2f", huberDelta) }
	if normFeatures       { extra += "  norm-features" }
	if cosineMode {
		fmt.Printf("Config: lr=%.3f  wd=%.4f  batch=%d  max_epochs=%d  schedule=cosine(T0=%d,Tmult=%.0f)%s\n\n",
			currentLR, wd, batch, epochs, cosineT0, cosineTMult, extra)
	} else {
		fmt.Printf("Config: lr=%.3f  wd=%.4f  batch=%d  max_epochs=%d  patience=%d%s\n\n",
			currentLR, wd, batch, epochs, patience, extra)
	}

	fmt.Printf("%6s  %10s  %10s  %10s  %8s  %6s\n", "Epoch", "Train MSE", "Val MSE", "Val RMSE", "LR", "Time")
	fmt.Println("--------------------------------------------------------------")

	// Shuffle buffer for training rows (reused across epochs, avoids per-epoch alloc).
	type cardEntry struct {
		features [ml.TotalFeatureLen]float32
		tv       float32 // raw target before y-normalization
		weight   float32 // 1.0 for current iter, replayWeight for replay
	}
	var shufBuf []cardEntry
	if shuffleBufSize > 0 {
		shufBuf = make([]cardEntry, shuffleBufSize)
	}

	// --- Training loop: stream each epoch ---
	for epoch := 1; epoch <= epochs; epoch++ {
		epStart := time.Now()

		// Compute LR for this epoch.
		if cosineMode {
			ratio := float64(cosineCurEpoch) / float64(cosineCycleLen)
			currentLR = float32(float64(minLR) + 0.5*(float64(lr)-float64(minLR))*(1+math.Cos(math.Pi*ratio)))
		}

		var trainLoss float32
		var trainN int64
		var valLoss float32
		var valN int64
		batchCount := 0
		readIdx := int64(0)
		inTrain := true

		processBatch := func(n int, isTrain bool) {
			trainer.ForwardBatch(n, preds[:n])
			for j := 0; j < n; j++ {
				diff := preds[j] - trainer.BatchY[j]
				if isTrain {
					trainLoss += diff * diff
				} else {
					valLoss += diff * diff
				}
			}
			if isTrain {
				trainN += int64(n)
				trainer.BackwardBatch(n, preds[:n], trainer.BatchY[:n])
				trainer.Step(currentLR, wd, n)
				trainer.ZeroGrad()
			} else {
				valN += int64(n)
			}
		}

		emit := func(features *[ml.TotalFeatureLen]float32, normTV, weight float32) {
			dstOff := batchCount * nFeat
			if normFeatures {
				for j, f := range features {
					trainer.BatchX[dstOff+j] = (f - featMean[j]) / (featStd[j] + 1e-8)
				}
			} else {
				copy(trainer.BatchX[dstOff:dstOff+nFeat], (*features)[:])
			}
			trainer.BatchY[batchCount] = normTV
			trainer.BatchW[batchCount] = weight
			batchCount++
			if batchCount >= batch {
				processBatch(batchCount, inTrain)
				batchCount = 0
			}
		}

		shufN := 0
		flushShuf := func() {
			for i := shufN - 1; i > 0; i-- {
				j := rng.Intn(i + 1)
				shufBuf[i], shufBuf[j] = shufBuf[j], shufBuf[i]
			}
			for i := 0; i < shufN; i++ {
				normTV := (shufBuf[i].tv - yMean) / (yStd + 1e-8)
				emit(&shufBuf[i].features, normTV, shufBuf[i].weight)
			}
			shufN = 0
		}

		for fileIdx, path := range dataPaths {
			rowWeight := float32(1.0)
			if fileIdx > 0 { rowWeight = replayWeight }

			f, err := os.Open(path)
			if err != nil { log.Fatal(err) }
			rd := bufio.NewReaderSize(f, 1<<20)
			var row ml.CollectRow
			for {
				err := row.ReadBinary(rd)
				if err == io.EOF { break }
				if err != nil { log.Fatal(err) }

				// Transition: flush any buffered training rows before switching to val.
				if inTrain && readIdx >= nTrain {
					if shufN > 0 { flushShuf() }
					if batchCount > 0 { processBatch(batchCount, true); batchCount = 0 }
					inTrain = false
				}

				var tv float32
				if target == "made_bid_rate" { tv = row.MadeBidRate } else { tv = row.ScoreDelta }

				if inTrain && shuffleBufSize > 0 {
					shufBuf[shufN].features = row.Features
					shufBuf[shufN].tv       = tv
					shufBuf[shufN].weight   = rowWeight
					shufN++
					if shufN == shuffleBufSize { flushShuf() }
				} else {
					normTV := (tv - yMean) / (yStd + 1e-8)
					emit(&row.Features, normTV, rowWeight)
				}
				readIdx++
			}
			f.Close()
		}
		// Flush remaining training rows then any partial val batch.
		if shufN > 0 { flushShuf() }
		if batchCount > 0 { processBatch(batchCount, inTrain) }

		trainMSE := trainLoss / float32(trainN)
		valMSE := valLoss / float32(valN)
		valRMSE := float32(math.Sqrt(float64(valMSE))) * yStd

		improved := valMSE < bestValLoss*(1-minDelta)
		marker := ""
		if improved {
			bestValLoss = valMSE
			bestWeights = trainer.Finalize()
			marker = " ◀"
		}

		if cosineMode {
			cosineCurEpoch++
			if cosineCurEpoch >= cosineCycleLen {
				cosineCurEpoch = 0
				cosineCycleLen = int(float64(cosineCycleLen) * cosineTMult)
				marker += " ↺"
			}
		} else {
			if improved {
				plateauCount = 0
			} else {
				plateauCount++
				if plateauCount >= patience {
					currentLR *= 0.5
					plateauCount = 0
				}
			}
			if currentLR < minLR {
				fmt.Printf("%6d  %10.6f  %10.6f  %10.4f  %8.2e  %5.1fs%s\n",
					epoch, trainMSE, valMSE, valRMSE, currentLR, time.Since(epStart).Seconds(), marker)
				fmt.Printf("  Early stop: LR=%.2e reached minimum.\n", currentLR)
				break
			}
		}
		fmt.Printf("%6d  %10.6f  %10.6f  %10.4f  %8.2e  %5.1fs%s\n",
			epoch, trainMSE, valMSE, valRMSE, currentLR, time.Since(epStart).Seconds(), marker)
	}

	fmt.Printf("\nBest val MSE:  %.6f  (RMSE ≈ %.4f score points)\n",
		bestValLoss, float32(math.Sqrt(float64(bestValLoss)))*yStd)

	outF, _ := os.Create(outPath)
	enc := json.NewEncoder(outF)
	enc.SetIndent("", "  ")
	enc.Encode(bestWeights)
	outF.Close()
	fmt.Printf("Weights saved → %s\n", outPath)
}

// runBidBinaryStreaming handles binary format for bid training data (BidCollectRow).
func runBidBinaryStreaming(rng *rand.Rand, dataPaths []string, outPath, warmStartPath string, warmHiddenOnly bool, lrSchedule string, cosineT0 int, cosineTMult float64,
	h1, h2, h3, batch, epochs, patience int, valFrac float64,
	lr, wd, minLR, minDelta float32,
	shuffleBufSize int, huberDelta, replayWeight float32, normFeatures bool) {

	nFeat := ml.BidTotalLen
	fmt.Printf("Loading %v (format=bin, mode=bid, streaming) ...\n", dataPaths)
	t0 := time.Now()

	// --- Pass 1: count rows, compute target stats, compute per-feature stats ---
	var nRows int64
	var tSum, tSum2 float64
	featSum  := make([]float64, nFeat)
	featSum2 := make([]float64, nFeat)
	for _, path := range dataPaths {
		f, err := os.Open(path)
		if err != nil { log.Fatal(err) }
		rd := bufio.NewReaderSize(f, 1<<20)
		var row ml.BidCollectRow
		for {
			err := row.ReadBinary(rd)
			if err == io.EOF { break }
			if err != nil { log.Fatal(err) }
			tSum += float64(row.ScoreDelta)
			tSum2 += float64(row.ScoreDelta) * float64(row.ScoreDelta)
			for j, fv := range row.Features {
				featSum[j]  += float64(fv)
				featSum2[j] += float64(fv) * float64(fv)
			}
			nRows++
		}
		f.Close()
	}

	featMean := make([]float32, nFeat)
	featStd  := make([]float32, nFeat)
	for j := 0; j < nFeat; j++ {
		m := featSum[j] / float64(nRows)
		v := featSum2[j]/float64(nRows) - m*m
		if v < 0 { v = 0 }
		featMean[j] = float32(m)
		featStd[j]  = float32(math.Sqrt(v))
	}

	yMeanF64 := tSum / float64(nRows)
	yStdF64 := math.Sqrt(tSum2/float64(nRows) - yMeanF64*yMeanF64)
	yMean := float32(yMeanF64)
	yStd := float32(yStdF64)

	fmt.Printf("  %12d rows total (%v)\n", nRows, time.Since(t0).Round(time.Millisecond))
	fmt.Printf("  %d features  target=score_delta\n", nFeat)
	fmt.Printf("  mean=%.4f  std=%.4f\n", yMean, yStd)
	fmt.Printf("  rows=%10d\n\n", nRows)

	// --- Setup trainer ---
	trainer := ml.NewMLPTrainer(nFeat, h1, h2, h3, rng)
	trainer.ResizeBatch(batch)
	trainer.W.YMean = yMean
	trainer.W.YStd = yStd
	trainer.HuberDelta = huberDelta
	if normFeatures {
		trainer.W.FeatureMean = featMean
		trainer.W.FeatureStd  = featStd
	}

	if warmStartPath != "" {
		w, err := ml.LoadBidMLP(warmStartPath)
		if err != nil {
			log.Printf("warm start load failed (%v) — training from scratch", err)
		} else if wsErr := trainer.LoadWeights(w.Weights()); wsErr != nil {
			log.Printf("warm start skipped: %v", wsErr)
		} else {
			if warmHiddenOnly {
				trainer.ResetOutputLayer(rng)
				fmt.Printf("  Warm start (hidden only): %s\n\n", warmStartPath)
			} else {
				fmt.Printf("  Warm start: %s\n\n", warmStartPath)
			}
		}
	}

	bestValLoss := float32(math.Inf(1))
	var bestWeights ml.MLPWeights

	currentLR := lr
	plateauCount := 0
	cosineMode := lrSchedule == "cosine"
	cosineCurEpoch := 0
	cosineCycleLen := cosineT0
	preds := make([]float32, batch)

	fmt.Printf("Model:  %d → %d → %d", nFeat, h1, h2)
	if h3 > 0 { fmt.Printf(" → %d", h3) }
	nParams := nFeat*h1 + h1 + h1*h2 + h2
	if h3 > 0 {
		nParams += h2*h3 + h3 + h3 + 1
	} else {
		nParams += h2 + 1
	}
	fmt.Printf(" → 1  (%d params)\n", nParams)
	extra := ""
	if shuffleBufSize > 0 { extra += fmt.Sprintf("  shuffle-buf=%d", shuffleBufSize) }
	if huberDelta > 0     { extra += fmt.Sprintf("  huber=%.2f", huberDelta) }
	if normFeatures       { extra += "  norm-features" }
	if cosineMode {
		fmt.Printf("Config: lr=%.3f  wd=%.4f  batch=%d  max_epochs=%d  schedule=cosine(T0=%d,Tmult=%.0f)%s\n\n",
			currentLR, wd, batch, epochs, cosineT0, cosineTMult, extra)
	} else {
		fmt.Printf("Config: lr=%.3f  wd=%.4f  batch=%d  max_epochs=%d  patience=%d%s\n\n",
			currentLR, wd, batch, epochs, patience, extra)
	}

	fmt.Printf("%6s  %10s  %10s  %10s  %8s  %6s\n", "Epoch", "Train MSE", "Val MSE", "Val RMSE", "LR", "Time")
	fmt.Println("--------------------------------------------------------------")

	// Separate train/val staging buffers (reused across epochs).
	trainBatchX := make([]float32, batch*nFeat)
	valBatchX   := make([]float32, batch*nFeat)
	trainBatchY := make([]float32, batch)
	valBatchY   := make([]float32, batch)
	trainBatchW := make([]float32, batch)
	valBatchW   := make([]float32, batch)
	for i := range trainBatchW { trainBatchW[i] = 1.0 }
	for i := range valBatchW   { valBatchW[i]   = 1.0 }

	// Shuffle buffer (all rows — bid uses modulo val split, not positional).
	type bidEntry struct {
		features [ml.BidTotalLen]float32
		tv       float32 // raw score delta before y-normalization
		weight   float32
	}
	var shufBuf []bidEntry
	if shuffleBufSize > 0 {
		shufBuf = make([]bidEntry, shuffleBufSize)
	}

	// --- Training loop: stream each epoch ---
	for epoch := 1; epoch <= epochs; epoch++ {
		epStart := time.Now()

		// Compute LR for this epoch.
		if cosineMode {
			ratio := float64(cosineCurEpoch) / float64(cosineCycleLen)
			currentLR = float32(float64(minLR) + 0.5*(float64(lr)-float64(minLR))*(1+math.Cos(math.Pi*ratio)))
		}

		var trainLoss float32
		var trainN int64
		var valLoss float32
		var valN int64
		rowIdx := int64(0) // emitted row counter (determines val split)

		processBatch := func(n int, isTrain bool) {
			trainer.ForwardBatch(n, preds[:n])
			for j := 0; j < n; j++ {
				diff := preds[j] - trainer.BatchY[j]
				if isTrain {
					trainLoss += diff * diff
				} else {
					valLoss += diff * diff
				}
			}
			if isTrain {
				trainN += int64(n)
				trainer.BackwardBatch(n, preds[:n], trainer.BatchY[:n])
				trainer.Step(currentLR, wd, n)
				trainer.ZeroGrad()
			} else {
				valN += int64(n)
			}
		}

		var trainBatch, valBatch int
		flushTrain := func() {
			if trainBatch > 0 {
				copy(trainer.BatchX, trainBatchX[:trainBatch*nFeat])
				copy(trainer.BatchY, trainBatchY[:trainBatch])
				copy(trainer.BatchW, trainBatchW[:trainBatch])
				processBatch(trainBatch, true)
				trainBatch = 0
			}
		}
		flushVal := func() {
			if valBatch > 0 {
				copy(trainer.BatchX, valBatchX[:valBatch*nFeat])
				copy(trainer.BatchY, valBatchY[:valBatch])
				processBatch(valBatch, false)
				valBatch = 0
			}
		}

		// emitBidRow assigns a row to train or val based on emitted rowIdx.
		emitBidRow := func(features *[ml.BidTotalLen]float32, normTV, weight float32) {
			isVal := rowIdx%10 == 0
			rowIdx++
			if isVal {
				dstOff := valBatch * nFeat
				if normFeatures {
					for j, f := range features {
						valBatchX[dstOff+j] = (f - featMean[j]) / (featStd[j] + 1e-8)
					}
				} else {
					copy(valBatchX[dstOff:dstOff+nFeat], (*features)[:])
				}
				valBatchY[valBatch] = normTV
				valBatch++
				if valBatch >= batch { flushVal() }
			} else {
				dstOff := trainBatch * nFeat
				if normFeatures {
					for j, f := range features {
						trainBatchX[dstOff+j] = (f - featMean[j]) / (featStd[j] + 1e-8)
					}
				} else {
					copy(trainBatchX[dstOff:dstOff+nFeat], (*features)[:])
				}
				trainBatchY[trainBatch] = normTV
				trainBatchW[trainBatch] = weight
				trainBatch++
				if trainBatch >= batch { flushTrain() }
			}
		}

		shufN := 0
		flushShuf := func() {
			for i := shufN - 1; i > 0; i-- {
				j := rng.Intn(i + 1)
				shufBuf[i], shufBuf[j] = shufBuf[j], shufBuf[i]
			}
			for i := 0; i < shufN; i++ {
				normTV := (shufBuf[i].tv - yMean) / (yStd + 1e-8)
				emitBidRow(&shufBuf[i].features, normTV, shufBuf[i].weight)
			}
			shufN = 0
		}

		for fileIdx, path := range dataPaths {
			rowWeight := float32(1.0)
			if fileIdx > 0 { rowWeight = replayWeight }

			f, err := os.Open(path)
			if err != nil { log.Fatal(err) }
			rd := bufio.NewReaderSize(f, 1<<20)
			var row ml.BidCollectRow
			for {
				err := row.ReadBinary(rd)
				if err == io.EOF { break }
				if err != nil { log.Fatal(err) }

				if shuffleBufSize > 0 {
					shufBuf[shufN].features = row.Features
					shufBuf[shufN].tv       = row.ScoreDelta
					shufBuf[shufN].weight   = rowWeight
					shufN++
					if shufN == shuffleBufSize { flushShuf() }
				} else {
					normTV := (row.ScoreDelta - yMean) / (yStd + 1e-8)
					emitBidRow(&row.Features, normTV, rowWeight)
				}
			}
			f.Close()
		}
		if shufN > 0 { flushShuf() }
		flushTrain()
		flushVal()

		trainMSE := trainLoss / float32(trainN)
		valMSE := valLoss / float32(valN)
		valRMSE := float32(math.Sqrt(float64(valMSE))) * yStd

		improved := valMSE < bestValLoss*(1-minDelta)
		marker := ""
		if improved {
			bestValLoss = valMSE
			bestWeights = trainer.Finalize()
			marker = " ◀"
		}

		if cosineMode {
			cosineCurEpoch++
			if cosineCurEpoch >= cosineCycleLen {
				cosineCurEpoch = 0
				cosineCycleLen = int(float64(cosineCycleLen) * cosineTMult)
				marker += " ↺"
			}
		} else {
			if improved {
				plateauCount = 0
			} else {
				plateauCount++
				if plateauCount >= patience {
					currentLR *= 0.5
					plateauCount = 0
				}
			}
			if currentLR < minLR {
				fmt.Printf("%6d  %10.6f  %10.6f  %10.4f  %8.2e  %5.1fs%s\n",
					epoch, trainMSE, valMSE, valRMSE, currentLR, time.Since(epStart).Seconds(), marker)
				fmt.Printf("  Early stop: LR=%.2e reached minimum.\n", currentLR)
				break
			}
		}
		fmt.Printf("%6d  %10.6f  %10.6f  %10.4f  %8.2e  %5.1fs%s\n",
			epoch, trainMSE, valMSE, valRMSE, currentLR, time.Since(epStart).Seconds(), marker)
	}

	fmt.Printf("\nBest val MSE:  %.6f  (RMSE ≈ %.4f score points)\n",
		bestValLoss, float32(math.Sqrt(float64(bestValLoss)))*yStd)

	outF, _ := os.Create(outPath)
	enc := json.NewEncoder(outF)
	enc.SetIndent("", "  ")
	enc.Encode(bestWeights)
	outF.Close()
	fmt.Printf("Weights saved → %s\n", outPath)
}

// runCSV handles CSV format with in-memory loading (two-pass).
func runCSV(rng *rand.Rand, dataPath, outPath, target string,
	h1, h2, h3, batch, epochs, patience int, valFrac float64,
	lr, wd, minLR, minDelta float32) {

	fmt.Printf("Loading %s (format=csv) ...\n", dataPath)
	t0 := time.Now()

	var nRows int64
	var tSum, tSum2 float64

	// Pass 1: stats
	f, err := os.Open(dataPath)
	if err != nil { log.Fatal(err) }
	r := csv.NewReader(f)
	r.ReuseRecord = true
	header, err := r.Read()
	if err != nil { log.Fatal(err) }

	targetIdx := -1
	meta := map[string]bool{
		"hand_id": true, "trick_num": true, "seat": true,
		"is_bidding_team": true, "score_delta": true,
		"made_bid_rate": true, "bid_level": true,
	}
	var featureIndices []int
	for i, col := range header {
		if col == target { targetIdx = i }
		if !meta[col] { featureIndices = append(featureIndices, i) }
	}
	if targetIdx == -1 { log.Fatalf("target column %s not found", target) }
	nFeat := len(featureIndices)

	for {
		record, err := r.Read()
		if err == io.EOF { break }
		if err != nil { log.Fatal(err) }
		val, _ := strconv.ParseFloat(record[targetIdx], 32)
		nRows++
		tSum += val
		tSum2 += val * val
	}
	f.Close()

	yMean := float32(tSum / float64(nRows))
	yStd := float32(math.Sqrt(tSum2/float64(nRows) - float64(yMean*yMean)))

	nVal := int64(float64(nRows) * valFrac)
	nTrain := nRows - nVal

	fmt.Printf("  %12d rows total (%v)\n", nRows, time.Since(t0).Round(time.Millisecond))
	fmt.Printf("  %d features  target=%s\n", nFeat, target)
	fmt.Printf("  mean=%.4f  std=%.4f\n", yMean, yStd)
	fmt.Printf("  train=%10d  val=%10d\n", nTrain, nVal)

	// Pass 2: load into memory
	fmt.Printf("Pass 2: loading into memory ...\n")
	t1 := time.Now()

	allX := make([]float32, nRows*int64(nFeat))
	allY := make([]float32, nRows)

	f, err = os.Open(dataPath)
	if err != nil { log.Fatal(err) }
	r = csv.NewReader(f)
	r.ReuseRecord = true
	r.Read() // skip header

	for i := int64(0); i < nRows; i++ {
		record, err := r.Read()
		if err != nil { break }
		tv, _ := strconv.ParseFloat(record[targetIdx], 32)
		allY[i] = (float32(tv) - yMean) / (yStd + 1e-8)
		off := i * int64(nFeat)
		for j, idx := range featureIndices {
			fv, _ := strconv.ParseFloat(record[idx], 32)
			allX[off+int64(j)] = float32(fv)
		}
	}
	f.Close()
	fmt.Printf("  loaded (%v)\n\n", time.Since(t1).Round(time.Millisecond))

	runInMemoryTraining(rng, allX, allY, nFeat, int(nRows), int(nTrain), int(nVal),
		yMean, yStd, h1, h2, h3, batch, epochs, patience,
		lr, wd, minLR, minDelta, outPath)
}

func runInMemoryTraining(rng *rand.Rand, allX, allY []float32, nFeat, nRows, nTrain, nVal int,
	yMean, yStd float32, h1, h2, h3, batch, epochs, patience int,
	lr, wd, minLR, minDelta float32, outPath string) {

	trainX := allX[:nTrain*nFeat]
	trainY := allY[:nTrain]
	valX := allX[nTrain*nFeat:]
	valY := allY[nTrain:]

	indices := make([]int, nTrain)
	for i := range indices { indices[i] = i }

	trainer := ml.NewMLPTrainer(nFeat, h1, h2, h3, rng)
	trainer.ResizeBatch(batch)
	trainer.W.YMean = yMean
	trainer.W.YStd = yStd

	bestValLoss := float32(math.Inf(1))
	var bestWeights ml.MLPWeights

	currentLR := lr
	plateauCount := 0
	preds := make([]float32, batch)

	fmt.Printf("Model:  %d → %d → %d", nFeat, h1, h2)
	if h3 > 0 { fmt.Printf(" → %d", h3) }
	nParams := nFeat*h1 + h1 + h1*h2 + h2
	if h3 > 0 {
		nParams += h2*h3 + h3 + h3 + 1
	} else {
		nParams += h2 + 1
	}
	fmt.Printf(" → 1  (%d params)\n", nParams)
	fmt.Printf("Config: lr=%.3f  wd=%.4f  batch=%d  max_epochs=%d  patience=%d\n\n",
		currentLR, wd, batch, epochs, patience)

	fmt.Printf("%6s  %10s  %10s  %10s  %8s  %6s\n", "Epoch", "Train MSE", "Val MSE", "Val RMSE", "LR", "Time")
	fmt.Println("--------------------------------------------------------------")

	for epoch := 1; epoch <= epochs; epoch++ {
		epStart := time.Now()

		// Shuffle training indices
		for i := len(indices) - 1; i > 0; i-- {
			j := rng.Intn(i + 1)
			indices[i], indices[j] = indices[j], indices[i]
		}

		// --- Train ---
		var trainLoss float32
		var trainN int64
		batchCount := 0

		for _, idx := range indices {
			srcOff := idx * nFeat
			dstOff := batchCount * nFeat
			copy(trainer.BatchX[dstOff:dstOff+nFeat], trainX[srcOff:srcOff+nFeat])
			trainer.BatchY[batchCount] = trainY[idx]
			batchCount++

			if batchCount >= batch {
				trainer.ForwardBatch(batchCount, preds)
				for j := 0; j < batchCount; j++ {
					diff := preds[j] - trainer.BatchY[j]
					trainLoss += diff * diff
				}
				trainN += int64(batchCount)
				trainer.BackwardBatch(batchCount, preds, trainer.BatchY)
				trainer.Step(currentLR, wd, batchCount)
				trainer.ZeroGrad()
				batchCount = 0
			}
		}
		if batchCount > 0 {
			trainer.ForwardBatch(batchCount, preds[:batchCount])
			for j := 0; j < batchCount; j++ {
				diff := preds[j] - trainer.BatchY[j]
				trainLoss += diff * diff
			}
			trainN += int64(batchCount)
			trainer.BackwardBatch(batchCount, preds[:batchCount], trainer.BatchY[:batchCount])
			trainer.Step(currentLR, wd, batchCount)
			trainer.ZeroGrad()
		}

		// --- Validate ---
		var valLoss float32
		var valN int64
		for start := 0; start < nVal; start += batch {
			end := start + batch
			if end > nVal { end = nVal }
			n := end - start
			copy(trainer.BatchX[:n*nFeat], valX[start*nFeat:end*nFeat])
			copy(trainer.BatchY[:n], valY[start:end])
			trainer.ForwardBatch(n, preds[:n])
			for j := 0; j < n; j++ {
				diff := preds[j] - trainer.BatchY[j]
				valLoss += diff * diff
			}
			valN += int64(n)
		}

		trainMSE := trainLoss / float32(trainN)
		valMSE := valLoss / float32(valN)
		valRMSE := float32(math.Sqrt(float64(valMSE))) * yStd

		marker := ""
		if valMSE < bestValLoss*(1-minDelta) {
			bestValLoss = valMSE
			bestWeights = trainer.Finalize()
			marker = " ◀"
			plateauCount = 0
		} else {
			plateauCount++
			if plateauCount >= patience {
				currentLR *= 0.5
				plateauCount = 0
			}
		}
		fmt.Printf("%6d  %10.6f  %10.6f  %10.4f  %8.2e  %5.1fs%s\n",
			epoch, trainMSE, valMSE, valRMSE, currentLR, time.Since(epStart).Seconds(), marker)
		if currentLR < minLR {
			fmt.Printf("  Early stop: LR=%.2e reached minimum.\n", currentLR)
			break
		}
	}

	fmt.Printf("\nBest val MSE:  %.6f  (RMSE ≈ %.4f score points)\n",
		bestValLoss, float32(math.Sqrt(float64(bestValLoss)))*yStd)

	outF, _ := os.Create(outPath)
	enc := json.NewEncoder(outF)
	enc.SetIndent("", "  ")
	enc.Encode(bestWeights)
	outF.Close()
	fmt.Printf("Weights saved → %s\n", outPath)
}
