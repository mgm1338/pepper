package ml

import (
	"encoding/csv"
	"io"
	"os"
	"strconv"
)

// DataBatch holds a mini-batch of features and targets.
type DataBatch struct {
	X []float32 // [N * nFeat]
	Y []float32 // [N]
	N int
}

// DataLoader streams CSV data in batches.
type DataLoader struct {
	path      string
	format    string
	target    string
	file      *os.File
	csvR      *csv.Reader
	featIdx   []int
	targetIdx int
	nFeat     int
}

// NewDataLoader creates a streaming data loader.
// Currently supports CSV format only.
func NewDataLoader(path, format, target string) *DataLoader {
	f, err := os.Open(path)
	if err != nil {
		panic("DataLoader: " + err.Error())
	}

	csvR := csv.NewReader(f)
	csvR.ReuseRecord = true

	// Read header
	header, err := csvR.Read()
	if err != nil {
		panic("DataLoader: no header")
	}

	var featIdx []int
	targetIdx := -1
	for i, col := range header {
		if col == target {
			targetIdx = i
		} else if col != "hand_id" && col != "seat" && col != "bid_level" {
			featIdx = append(featIdx, i)
		}
	}
	if targetIdx < 0 {
		panic("DataLoader: target column not found: " + target)
	}

	return &DataLoader{
		path:      path,
		format:    format,
		target:    target,
		file:      f,
		csvR:      csvR,
		featIdx:   featIdx,
		targetIdx: targetIdx,
		nFeat:     len(featIdx),
	}
}

// Stream returns a channel of DataBatch.
func (dl *DataLoader) Stream(batchSize, nFeat int) <-chan DataBatch {
	ch := make(chan DataBatch, 4)
	go func() {
		defer close(ch)
		x := make([]float32, batchSize*nFeat)
		y := make([]float32, batchSize)
		n := 0

		for {
			record, err := dl.csvR.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				break
			}

			off := n * nFeat
			for j, idx := range dl.featIdx {
				fv, _ := strconv.ParseFloat(record[idx], 32)
				x[off+j] = float32(fv)
			}
			tv, _ := strconv.ParseFloat(record[dl.targetIdx], 32)
			y[n] = float32(tv)
			n++

			if n >= batchSize {
				// Send a copy
				bx := make([]float32, n*nFeat)
				by := make([]float32, n)
				copy(bx, x[:n*nFeat])
				copy(by, y[:n])
				ch <- DataBatch{X: bx, Y: by, N: n}
				n = 0
			}
		}

		// Send remaining
		if n > 0 {
			bx := make([]float32, n*nFeat)
			by := make([]float32, n)
			copy(bx, x[:n*nFeat])
			copy(by, y[:n])
			ch <- DataBatch{X: bx, Y: by, N: n}
		}
	}()
	return ch
}

// Close releases the file handle.
func (dl *DataLoader) Close() {
	if dl.file != nil {
		dl.file.Close()
	}
}
