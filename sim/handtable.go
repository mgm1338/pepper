package sim

import (
	"encoding/csv"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// HandProfile describes a hand in a suit-agnostic way for aggregation.
// This lets us answer: "given this hand shape, what bid wins most often?"
type HandProfile struct {
	RightBowers    int // 0, 1, or 2
	LeftBowers     int // 0, 1, or 2
	HighTrump      int // A or K of trump suit (non-bower winners after bowers pulled)
	OtherTrump     int // remaining non-bower trump (Q, 10, 9)
	OffSuitAces    int // non-trump aces (near-certain extra tricks)
	VoidSuits      int // number of void off-suits (0–3)
	SingletonSuits int // number of singleton off-suits (0–3)
}

// BidOutcome tracks how often each bid amount succeeded for a given hand profile.
type BidOutcome struct {
	Profile    HandProfile
	BidResults map[int]*BidStat // key = bid amount
}

// BidStat tracks attempts and successes for one bid amount.
type BidStat struct {
	Attempts int
	Made     int
}

func (b *BidStat) WinRate() float64 {
	if b.Attempts == 0 {
		return 0
	}
	return float64(b.Made) / float64(b.Attempts)
}

// HandTable accumulates bid outcomes grouped by hand profile.
type HandTable struct {
	data map[HandProfile]*BidOutcome
}

func NewHandTable() *HandTable {
	return &HandTable{data: map[HandProfile]*BidOutcome{}}
}

// Record adds one hand's outcome to the table.
func (t *HandTable) Record(profile HandProfile, bidAmount int, made bool) {
	if _, ok := t.data[profile]; !ok {
		t.data[profile] = &BidOutcome{
			Profile:    profile,
			BidResults: map[int]*BidStat{},
		}
	}
	bo := t.data[profile]
	if _, ok := bo.BidResults[bidAmount]; !ok {
		bo.BidResults[bidAmount] = &BidStat{}
	}
	bo.BidResults[bidAmount].Attempts++
	if made {
		bo.BidResults[bidAmount].Made++
	}
}

// Merge adds all data from other into this table.
func (t *HandTable) Merge(other *HandTable) {
	for p, bo := range other.data {
		if _, ok := t.data[p]; !ok {
			t.data[p] = &BidOutcome{Profile: p, BidResults: map[int]*BidStat{}}
		}
		for bid, stat := range bo.BidResults {
			if _, ok := t.data[p].BidResults[bid]; !ok {
				t.data[p].BidResults[bid] = &BidStat{}
			}
			t.data[p].BidResults[bid].Attempts += stat.Attempts
			t.data[p].BidResults[bid].Made += stat.Made
		}
	}
}

// OptimalBid returns the bid amount with the highest win rate for a profile,
// among bids with at least minAttempts samples.
func (t *HandTable) OptimalBid(profile HandProfile, minAttempts int) (int, float64) {
	bo, ok := t.data[profile]
	if !ok {
		return 0, 0
	}
	bestBid := 0
	bestRate := -1.0
	for bid, stat := range bo.BidResults {
		if stat.Attempts >= minAttempts && stat.WinRate() > bestRate {
			bestRate = stat.WinRate()
			bestBid = bid
		}
	}
	return bestBid, bestRate
}

// WriteCSV writes the full bid outcome table sorted by hand profile then bid.
func (t *HandTable) WriteCSV(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	w.Write([]string{
		"right_bowers", "left_bowers", "high_trump", "other_trump", "offsuit_aces",
		"void_suits", "singleton_suits", "total_trump",
		"bid_amount", "attempts", "made", "win_rate", "optimal_bid",
	})

	// Sort profiles for stable output.
	type row struct {
		p   HandProfile
		bid int
		s   *BidStat
	}
	var rows []row
	for p, bo := range t.data {
		for bid, stat := range bo.BidResults {
			rows = append(rows, row{p, bid, stat})
		}
	}
	sort.Slice(rows, func(i, j int) bool {
		pi, pj := rows[i].p, rows[j].p
		ti := pi.RightBowers*2 + pi.LeftBowers*2 + pi.OtherTrump
		tj := pj.RightBowers*2 + pj.LeftBowers*2 + pj.OtherTrump
		if ti != tj {
			return ti > tj
		}
		if pi.RightBowers != pj.RightBowers {
			return pi.RightBowers > pj.RightBowers
		}
		if pi.LeftBowers != pj.LeftBowers {
			return pi.LeftBowers > pj.LeftBowers
		}
		if pi.VoidSuits != pj.VoidSuits {
			return pi.VoidSuits > pj.VoidSuits
		}
		return rows[i].bid < rows[j].bid
	})

	for _, r := range rows {
		optBid, _ := t.OptimalBid(r.p, 100)
		totalTrump := r.p.RightBowers + r.p.LeftBowers + r.p.HighTrump + r.p.OtherTrump
		w.Write([]string{
			strconv.Itoa(r.p.RightBowers),
			strconv.Itoa(r.p.LeftBowers),
			strconv.Itoa(r.p.HighTrump),
			strconv.Itoa(r.p.OtherTrump),
			strconv.Itoa(r.p.OffSuitAces),
			strconv.Itoa(r.p.VoidSuits),
			strconv.Itoa(r.p.SingletonSuits),
			strconv.Itoa(totalTrump),
			strconv.Itoa(r.bid),
			strconv.Itoa(r.s.Attempts),
			strconv.Itoa(r.s.Made),
			fmt.Sprintf("%.3f", r.s.WinRate()),
			strconv.Itoa(optBid),
		})
	}

	w.Flush()
	return w.Error()
}

// PrintSummary prints a detailed bid distribution table by hand profile.
// For each profile it shows the make rate at each bid level (4-8),
// the optimal bid, and total sample count.
//
// Column guide:
//   Trump = total trump in hand   RR = right bowers   LL = left bowers
//   V     = void non-trump suits
//   Bid4..Bid8 = make% at that bid level (blank if < minAttempts/10 samples)
//   Opt   = bid level with highest make rate (min samples required)
//   N     = total hands as bidding team with this profile
func (t *HandTable) PrintSummary(minAttempts int) {
	type profEntry struct {
		p       HandProfile
		bo      *BidOutcome
		total   int
		optBid  int
		optRate float64
	}

	var entries []profEntry
	for p, bo := range t.data {
		total := 0
		for _, s := range bo.BidResults {
			total += s.Attempts
		}
		if total < minAttempts {
			continue
		}
		optBid, rate := t.OptimalBid(p, minAttempts/10)
		entries = append(entries, profEntry{p, bo, total, optBid, rate})
	}

	sort.Slice(entries, func(i, j int) bool {
		pi, pj := entries[i].p, entries[j].p
		ti := pi.RightBowers + pi.LeftBowers + pi.OtherTrump
		tj := pj.RightBowers + pj.LeftBowers + pj.OtherTrump
		if ti != tj {
			return ti > tj
		}
		if pi.RightBowers != pj.RightBowers {
			return pi.RightBowers > pj.RightBowers
		}
		if pi.LeftBowers != pj.LeftBowers {
			return pi.LeftBowers > pj.LeftBowers
		}
		if pi.OtherTrump != pj.OtherTrump {
			return pi.OtherTrump > pj.OtherTrump
		}
		return pi.VoidSuits > pj.VoidSuits
	})

	fmt.Println()
	fmt.Println("=== Bid Distribution by Hand Profile ===")
	fmt.Println("  RR=right bowers  LL=left bowers  Hi=A/K trump  Oth=Q/10/9 trump")
	fmt.Println("  Ac=offsuit aces  V=voids")
	fmt.Println("  Bid columns show make% (n) — blank if fewer than min samples")
	fmt.Println()
	fmt.Printf("  %-5s %-3s %-3s %-3s %-4s %-3s %-3s  %-14s %-14s %-14s %-14s %-14s  %-4s  %s\n",
		"Trump", "RR", "LL", "Hi", "Oth", "Ac", "V",
		"Bid4", "Bid5", "Bid6", "Bid7", "Bid8",
		"Opt", "N")
	fmt.Println("  " + strings.Repeat("-", 110))

	minBidSamples := minAttempts / 10
	fmtBid := func(bo *BidOutcome, bid int) string {
		s, ok := bo.BidResults[bid]
		if !ok || s.Attempts < minBidSamples {
			return fmt.Sprintf("%-14s", "--")
		}
		return fmt.Sprintf("%-14s", fmt.Sprintf("%.1f%%(%d)", s.WinRate()*100, s.Attempts))
	}

	for _, e := range entries {
		trump := e.p.RightBowers + e.p.LeftBowers + e.p.HighTrump + e.p.OtherTrump
		fmt.Printf("  %-5d %-3d %-3d %-3d %-4d %-3d %-3d  %s%s%s%s%s  %-4d  %d\n",
			trump, e.p.RightBowers, e.p.LeftBowers, e.p.HighTrump, e.p.OtherTrump,
			e.p.OffSuitAces, e.p.VoidSuits,
			fmtBid(e.bo, 4), fmtBid(e.bo, 5), fmtBid(e.bo, 6), fmtBid(e.bo, 7), fmtBid(e.bo, 8),
			e.optBid, e.total)
	}
}
