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
	"time"

	"github.com/max/pepper/ml"
)

func main() {
	dataPath := flag.String("data", "collect.csv", "Input file from cmd/collect")
	format   := flag.String("format", "csv", "Input format: csv or bin")
	outPath  := flag.String("out", "model_weights.json", "Output weights JSON")
	epochs   := flag.Int("epochs", 100, "Max training epochs")
	batch    := flag.Int("batch", 4096, "Mini-batch size")
	lr       := flag.Float64("lr", 1e-3, "Adam learning rate")
	valFrac  := flag.Float64("val", 0.1, "Validation fraction")
	h1       := flag.Int("h1", 128, "Hidden layer 1 size")
	h2       := flag.Int("h2", 64, "Hidden layer 2 size")
	h3       := flag.Int("h3", 0, "Hidden layer 3 size (0=2-layer)")
	wd       := flag.Float64("wd", 1e-4, "L2 weight decay")
	patience := flag.Int("patience", 5, "ReduceLROnPlateau patience")
	minLR    := flag.Float64("min-lr", 1e-6, "Stop when LR drops below this")
	target   := flag.String("target", "score_delta", "Target column")
	seed     := flag.Int64("seed", 42, "Random seed")
	flag.Parse()

	rng := rand.New(rand.NewSource(*seed))

	if *format == "bin" {
		runBinaryStreaming(rng, *dataPath, *outPath, *target, *h1, *h2, *h3,
			*batch, *epochs, *patience, *valFrac,
			float32(*lr), float32(*wd), float32(*minLR))
	} else {
		runCSV(rng, *dataPath, *outPath, *target, *h1, *h2, *h3,
			*batch, *epochs, *patience, *valFrac,
			float32(*lr), float32(*wd), float32(*minLR))
	}
}

// runBinaryStreaming handles binary format with streaming (no full data in memory).
// Pass 1: compute stats. Each epoch: re-read file, stream batches.
func runBinaryStreaming(rng *rand.Rand, dataPath, outPath, target string,
	h1, h2, h3, batch, epochs, patience int, valFrac float64,
	lr, wd, minLR float32) {

	nFeat := ml.TotalFeatureLen
	fmt.Printf("Loading %s (format=bin, streaming) ...\n", dataPath)
	t0 := time.Now()

	// --- Pass 1: count rows and compute target stats ---
	var nRows int64
	var tSum, tSum2 float64
	{
		f, err := os.Open(dataPath)
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
			nRows++
		}
		f.Close()
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

	// --- Training loop: stream each epoch ---
	for epoch := 1; epoch <= epochs; epoch++ {
		epStart := time.Now()

		f, err := os.Open(dataPath)
		if err != nil { log.Fatal(err) }
		rd := bufio.NewReaderSize(f, 1<<20)

		var trainLoss float32
		var trainN int64
		var valLoss float32
		var valN int64
		batchCount := 0
		rowIdx := int64(0)

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

		var row ml.CollectRow
		inTrain := true
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
			normTV := (tv - yMean) / (yStd + 1e-8)

			// Flush training batch at train/val boundary
			if inTrain && rowIdx >= nTrain {
				if batchCount > 0 {
					processBatch(batchCount, true)
					batchCount = 0
				}
				inTrain = false
			}

			dstOff := batchCount * nFeat
			copy(trainer.BatchX[dstOff:dstOff+nFeat], row.Features[:])
			trainer.BatchY[batchCount] = normTV
			batchCount++
			rowIdx++

			if batchCount >= batch {
				processBatch(batchCount, inTrain)
				batchCount = 0
			}
		}
		// Flush remaining
		if batchCount > 0 {
			processBatch(batchCount, inTrain)
		}
		f.Close()

		trainMSE := trainLoss / float32(trainN)
		valMSE := valLoss / float32(valN)
		valRMSE := float32(math.Sqrt(float64(valMSE))) * yStd

		marker := ""
		if valMSE < bestValLoss {
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

// runCSV handles CSV format with in-memory loading (two-pass).
func runCSV(rng *rand.Rand, dataPath, outPath, target string,
	h1, h2, h3, batch, epochs, patience int, valFrac float64,
	lr, wd, minLR float32) {

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
		lr, wd, minLR, outPath)
}

func runInMemoryTraining(rng *rand.Rand, allX, allY []float32, nFeat, nRows, nTrain, nVal int,
	yMean, yStd float32, h1, h2, h3, batch, epochs, patience int,
	lr, wd, minLR float32, outPath string) {

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
		if valMSE < bestValLoss {
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
