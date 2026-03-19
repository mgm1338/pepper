package sim

import "fmt"

// HandRecord stores the outcome of a single hand for analysis.
type HandRecord struct {
	GameID      int
	HandNum     int
	BidderSeat  int
	BidderTeam  int
	BidAmount   int
	IsPepper    bool
	IsStuck     bool
	MadeBid     bool
	TricksTeam0 int
	TricksTeam1 int
	ScoreTeam0  int // delta this hand
	ScoreTeam1  int // delta this hand
	Trump       int // suit index
	StratTeam0  string
	StratTeam1  string

	// Trump distribution at start of play.
	BidderTrump   int // trump held by bidding team
	OpponentTrump int // trump held by opponents
	// Trump exhausted after each trick (cumulative).
	TrumpByTrick [8]int
}

// GameRecord stores the outcome of a complete game.
type GameRecord struct {
	GameID      int
	Winner      int // team index
	FinalScore0 int
	FinalScore1 int
	Rounds      int
	StratTeam0  string
	StratTeam1  string
}

// Stats aggregates results across many games for one matchup.
type Stats struct {
	StratTeam0 string
	StratTeam1 string
	Games      int

	Wins [2]int

	TotalScore    [2]int
	TotalRounds   int

	HandsPlayed     int
	BidsMade        int
	BidsMissed      int
	PepperCalls     int
	PepperMade      int
	StuckDealer     int

	TricksAsBidder  [2]int // tricks taken when your team bid
	TricksAsDefense [2]int // tricks taken when opponent bid

	BidDistribution [9]int // index = bid amount (0–8), count of hands with that bid
}

func (s *Stats) AddGame(g GameRecord) {
	s.Games++
	s.Wins[g.Winner]++
	s.TotalScore[0] += g.FinalScore0
	s.TotalScore[1] += g.FinalScore1
	s.TotalRounds += g.Rounds
}

func (s *Stats) AddHand(h HandRecord) {
	s.HandsPlayed++
	if h.IsPepper {
		s.PepperCalls++
		if h.MadeBid {
			s.PepperMade++
		}
	} else {
		if h.IsStuck {
			s.StuckDealer++
		}
		if h.MadeBid {
			s.BidsMade++
		} else {
			s.BidsMissed++
		}
		if h.BidAmount >= 0 && h.BidAmount <= 8 {
			s.BidDistribution[h.BidAmount]++
		}
	}
	s.TricksAsBidder[h.BidderTeam] += h.TricksTeam0*zeroIf(h.BidderTeam != 0) + h.TricksTeam1*zeroIf(h.BidderTeam != 1)
}

func zeroIf(cond bool) int {
	if cond {
		return 0
	}
	return 1
}

func (s *Stats) WinRate(team int) float64 {
	if s.Games == 0 {
		return 0
	}
	return float64(s.Wins[team]) / float64(s.Games)
}

func (s *Stats) BidAccuracy() float64 {
	total := s.BidsMade + s.BidsMissed
	if total == 0 {
		return 0
	}
	return float64(s.BidsMade) / float64(total)
}

func (s *Stats) PepperAccuracy() float64 {
	if s.PepperCalls == 0 {
		return 0
	}
	return float64(s.PepperMade) / float64(s.PepperCalls)
}

func (s *Stats) AvgRoundsPerGame() float64 {
	if s.Games == 0 {
		return 0
	}
	return float64(s.TotalRounds) / float64(s.Games)
}

func (s *Stats) Print() {
	fmt.Printf("=== %s vs %s (%d games) ===\n", s.StratTeam0, s.StratTeam1, s.Games)
	fmt.Printf("  Win rate:       %s=%.1f%%  %s=%.1f%%\n",
		s.StratTeam0, s.WinRate(0)*100,
		s.StratTeam1, s.WinRate(1)*100)
	fmt.Printf("  Avg score/game: Team0=%.1f  Team1=%.1f\n",
		float64(s.TotalScore[0])/float64(s.Games),
		float64(s.TotalScore[1])/float64(s.Games))
	fmt.Printf("  Avg rounds:     %.1f\n", s.AvgRoundsPerGame())
	fmt.Printf("  Bid accuracy:   %.1f%%  (%d made / %d missed)\n",
		s.BidAccuracy()*100, s.BidsMade, s.BidsMissed)
	fmt.Printf("  Pepper:         %.1f%% success  (%d/%d calls)\n",
		s.PepperAccuracy()*100, s.PepperMade, s.PepperCalls)
	fmt.Printf("  Stuck dealer:   %d hands\n", s.StuckDealer)
	fmt.Printf("  Bid distribution: ")
	for i := 3; i <= 8; i++ {
		fmt.Printf("%d:%d ", i, s.BidDistribution[i])
	}
	fmt.Println()
}
