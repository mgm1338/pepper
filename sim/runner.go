package sim

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

// StrategyFactory creates 6 strategies for a game given team assignments.
// team0Seats = [0,2,4], team1Seats = [1,3,5]
type StrategyFactory func(rng *rand.Rand) [6]game.Strategy

// RunMatchup runs N games between two strategy factories and returns stats + records.
func RunMatchup(n int, team0Name, team1Name string, factory StrategyFactory, seed int64) (*Stats, []GameRecord, []HandRecord, *HandTable) {
	rng := rand.New(rand.NewSource(seed))
	stats := &Stats{StratTeam0: team0Name, StratTeam1: team1Name}
	var gameRecords []GameRecord
	var handRecords []HandRecord
	handTable := NewHandTable()

	for gameID := 0; gameID < n; gameID++ {
		gs := game.NewGame(rng.Intn(6))
		strategies := factory(rng)

		for {
			result := game.PlayHand(gs, strategies, rng, game.NoopLogger{})

			hr := HandRecord{
				GameID:        gameID,
				HandNum:       gs.Round,
				BidderSeat:    result.BidderSeat,
				BidderTeam:    int(game.TeamOf(result.BidderSeat)),
				BidAmount:     result.BidAmount,
				IsPepper:      result.IsPepper,
				MadeBid:       result.MadeBid,
				TricksTeam0:   result.TricksTaken[0],
				TricksTeam1:   result.TricksTaken[1],
				ScoreTeam0:    result.ScoreDelta[0],
				ScoreTeam1:    result.ScoreDelta[1],
				StratTeam0:    team0Name,
				StratTeam1:    team1Name,
				BidderTrump:   result.TrumpStats.BidderTrump,
				OpponentTrump: result.TrumpStats.OpponentTrump,
				TrumpByTrick:  result.TrumpStats.TrumpPlayedByTrick,
			}
			handRecords = append(handRecords, hr)
			stats.AddHand(hr)

			// Record hand profile for the bid lookup table (skip pepper and stuck hands).
			if !result.IsPepper && !result.IsStuck && len(result.BidderHand) > 0 {
				profile := profileFromHand(result.BidderHand, result.Trump)
				handTable.Record(profile, result.BidAmount, result.MadeBid)
			}

			gs.ApplyScore(game.Team0, result.ScoreDelta[0])
			gs.ApplyScore(game.Team1, result.ScoreDelta[1])
			gs.NextDealer()

			if over, winner := gs.IsOver(); over {
				gr := GameRecord{
					GameID:      gameID,
					Winner:      int(winner),
					FinalScore0: gs.Scores[0],
					FinalScore1: gs.Scores[1],
					Rounds:      gs.Round,
					StratTeam0:  team0Name,
					StratTeam1:  team1Name,
				}
				gameRecords = append(gameRecords, gr)
				stats.AddGame(gr)
				break
			}
		}
	}

	return stats, gameRecords, handRecords, handTable
}

// WriteGameCSV writes game records to a CSV file.
func WriteGameCSV(path string, records []GameRecord) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	w.Write([]string{"game_id", "winner", "score0", "score1", "rounds", "strat0", "strat1"})
	for _, r := range records {
		w.Write([]string{
			strconv.Itoa(r.GameID),
			strconv.Itoa(r.Winner),
			strconv.Itoa(r.FinalScore0),
			strconv.Itoa(r.FinalScore1),
			strconv.Itoa(r.Rounds),
			r.StratTeam0,
			r.StratTeam1,
		})
	}
	w.Flush()
	return w.Error()
}

// WriteHandCSV writes hand records to a CSV file.
func WriteHandCSV(path string, records []HandRecord) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	w.Write([]string{
		"game_id", "hand_num", "bidder_seat", "bidder_team",
		"bid_amount", "is_pepper", "made_bid",
		"tricks_team0", "tricks_team1",
		"score_team0", "score_team1",
		"strat0", "strat1",
	})
	for _, r := range records {
		w.Write([]string{
			strconv.Itoa(r.GameID),
			strconv.Itoa(r.HandNum),
			strconv.Itoa(r.BidderSeat),
			strconv.Itoa(r.BidderTeam),
			strconv.Itoa(r.BidAmount),
			boolStr(r.IsPepper),
			boolStr(r.MadeBid),
			strconv.Itoa(r.TricksTeam0),
			strconv.Itoa(r.TricksTeam1),
			strconv.Itoa(r.ScoreTeam0),
			strconv.Itoa(r.ScoreTeam1),
			r.StratTeam0,
			r.StratTeam1,
		})
	}
	w.Flush()
	return w.Error()
}

func boolStr(b bool) string {
	if b {
		return "1"
	}
	return "0"
}

// RunSweep runs all combinations of provided strategies against each other.
func RunSweep(n int, configs []NamedFactory, seed int64) {
	for i, a := range configs {
		for j, b := range configs {
			if j <= i {
				continue // skip duplicates and self-play
			}
			fmt.Printf("Running %s vs %s (%d games)...\n", a.Name, b.Name, n)
			factory := func(rng *rand.Rand) [6]game.Strategy {
				var strats [6]game.Strategy
				for seat := 0; seat < 6; seat++ {
					if seat%2 == 0 {
						strats[seat] = a.Factory(rng)
					} else {
						strats[seat] = b.Factory(rng)
					}
				}
				return strats
			}
			stats, gameRecs, handRecs, ht := RunMatchup(n, a.Name, b.Name, factory, seed+int64(i*100+j))
			stats.Print()
			ht.PrintSummary(200)

			gameFile := fmt.Sprintf("results_%s_vs_%s_games.csv", a.Name, b.Name)
			handFile := fmt.Sprintf("results_%s_vs_%s_hands.csv", a.Name, b.Name)
			tableFile := fmt.Sprintf("results_%s_vs_%s_bidtable.csv", a.Name, b.Name)
			WriteGameCSV(gameFile, gameRecs)
			WriteHandCSV(handFile, handRecs)
			ht.WriteCSV(tableFile)
			fmt.Printf("  Wrote %s, %s, %s\n\n", gameFile, handFile, tableFile)
		}
	}
}

// RunHandTable runs n games across multiple workers, accumulating only the
// HandTable (no raw records kept in memory). Writes a checkpoint CSV every
// checkpointEvery games. Progress is printed to stdout.
func RunHandTable(n int, factory StrategyFactory, seed int64, workers int, checkpointEvery int, outPath string) *HandTable {
	perWorker := n / workers
	results := make([]*HandTable, workers)
	done := make(chan int, workers)

	for w := 0; w < workers; w++ {
		go func(wid int) {
			rng := rand.New(rand.NewSource(seed + int64(wid*1000000)))
			ht := NewHandTable()
			checkpoint := checkpointEvery / workers
			gs := game.NewGame(rng.Intn(6))
			strategies := factory(rng)

			for i := 0; i < perWorker; i++ {
				result := game.PlayHand(gs, strategies, rng, game.NoopLogger{})

				if !result.IsPepper && !result.IsStuck && len(result.BidderHand) > 0 {
					profile := profileFromHand(result.BidderHand, result.Trump)
					ht.Record(profile, result.BidAmount, result.MadeBid)
				}

				gs.ApplyScore(game.Team0, result.ScoreDelta[0])
				gs.ApplyScore(game.Team1, result.ScoreDelta[1])
				gs.NextDealer()
				if over, _ := gs.IsOver(); over {
					gs = game.NewGame(rng.Intn(6))
					strategies = factory(rng)
				}

				// Checkpoint from worker 0 only to avoid concurrent writes.
				if wid == 0 && checkpoint > 0 && (i+1)%checkpoint == 0 {
					gamesTotal := (i + 1) * workers
					fmt.Printf("  checkpoint: %d / %d games\n", gamesTotal, n)
					// Merge current worker's table into a temp table for checkpoint.
					tmp := NewHandTable()
					tmp.Merge(ht)
					tmp.WriteCSV(outPath)
				}
			}

			results[wid] = ht
			done <- wid
		}(w)
	}

	for range make([]struct{}, workers) {
		<-done
	}

	// Merge all worker tables.
	merged := NewHandTable()
	for _, ht := range results {
		merged.Merge(ht)
	}
	return merged
}

// PairedResult holds the outcome of a paired same-deal evaluation.
type PairedResult struct {
	Hands          int
	TotalAdvantage int     // cumulative score(A team0) - score(B team0) across all hands
	AvgAdvantage   float64 // TotalAdvantage / Hands
}

// RunPairedHands deals n hands once each and plays them with two strategy sets.
// factoryA and factoryB each produce 6 strategies; team0 = seats 0,2,4.
// Returns the per-hand average score advantage of A over B on team0.
// Using a neutral game state (scores reset each hand) isolates decision quality
// from cumulative score context, giving a cleaner signal per hand.
func RunPairedHands(n int, factoryA, factoryB StrategyFactory, seed int64) PairedResult {
	rng := rand.New(rand.NewSource(seed))
	total := 0
	dealer := 0

	for i := 0; i < n; i++ {
		hands := card.Deal(rng)

		gsA := &game.GameState{Dealer: dealer, Scores: [2]int{0, 0}}
		gsB := &game.GameState{Dealer: dealer, Scores: [2]int{0, 0}}

		stratsA := factoryA(rng)
		stratsB := factoryB(rng)

		rA := game.PlayHandFrom(gsA, stratsA, rng, game.NoopLogger{}, hands, game.BidResult{})
		rB := game.PlayHandFrom(gsB, stratsB, rng, game.NoopLogger{}, hands, game.BidResult{})

		total += rA.ScoreDelta[0] - rB.ScoreDelta[0]
		dealer = (dealer + 1) % 6
	}

	avg := 0.0
	if n > 0 {
		avg = float64(total) / float64(n)
	}
	return PairedResult{Hands: n, TotalAdvantage: total, AvgAdvantage: avg}
}

// NamedFactory pairs a strategy name with a factory function.
type NamedFactory struct {
	Name    string
	Factory func(rng *rand.Rand) game.Strategy
}

// profileFromHand builds a suit-agnostic HandProfile from a hand and trump suit.
func profileFromHand(hand []card.Card, trump card.Suit) HandProfile {
	var rights, lefts, highTrump, otherTrump, offSuitAces int
	suitCounts := [4]int{}

	for _, c := range hand {
		effSuit := card.EffectiveSuit(c, trump)
		if effSuit == trump {
			switch {
			case card.IsRightBower(c, trump):
				rights++
			case card.IsLeftBower(c, trump):
				lefts++
			default:
				// A and K of trump are high trump (winners after bowers pulled).
				if c.Rank == card.Ace || c.Rank == card.King {
					highTrump++
				} else {
					otherTrump++
				}
			}
		} else {
			suitCounts[c.Suit]++
			if c.Rank == card.Ace {
				offSuitAces++
			}
		}
	}

	var voids, singletons int
	suits := []card.Suit{card.Spades, card.Clubs, card.Hearts, card.Diamonds}
	for _, s := range suits {
		if s == trump || s == card.PartnerSuit(trump) {
			continue
		}
		switch suitCounts[s] {
		case 0:
			voids++
		case 1:
			singletons++
		}
	}
	// Also check partner suit (left bower removed from it).
	partner := card.PartnerSuit(trump)
	switch suitCounts[partner] {
	case 0:
		voids++
	case 1:
		singletons++
	}

	return HandProfile{
		RightBowers:    rights,
		LeftBowers:     lefts,
		HighTrump:      highTrump,
		OtherTrump:     otherTrump,
		OffSuitAces:    offSuitAces,
		VoidSuits:      voids,
		SingletonSuits: singletons,
	}
}
