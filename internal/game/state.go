package game

// TeamIndex identifies one of the two teams.
// Team 0 = seats 0, 2, 4
// Team 1 = seats 1, 3, 5
type TeamIndex int

const (
	Team0 TeamIndex = 0
	Team1 TeamIndex = 1
)

// TeamOf returns the team for a given seat (0–5).
func TeamOf(seat int) TeamIndex {
	return TeamIndex(seat % 2)
}

// Partners returns the two partner seats for a given seat.
func Partners(seat int) [2]int {
	team := seat % 2
	var result [2]int
	idx := 0
	for s := 0; s < 6; s++ {
		if s != seat && s%2 == team {
			result[idx] = s
			idx++
		}
	}
	return result
}

// GameState holds the persistent state across hands.
type GameState struct {
	Scores [2]int // scores[0] = Team0, scores[1] = Team1
	Dealer int    // seat index of the current dealer (0–5)
	Round  int    // number of hands played
}

// NewGame creates a fresh game with the given starting dealer.
func NewGame(dealer int) *GameState {
	return &GameState{Dealer: dealer}
}

// NextDealer advances the dealer seat clockwise.
func (g *GameState) NextDealer() {
	g.Dealer = (g.Dealer + 1) % 6
	g.Round++
}

// IsOver returns true and the winning team if the game has ended.
func (g *GameState) IsOver() (bool, TeamIndex) {
	s0, s1 := g.Scores[0], g.Scores[1]

	// Normal win: reach 64
	if s0 >= 64 && s1 >= 0 {
		return true, Team0
	}
	if s1 >= 64 && s0 >= 0 {
		return true, Team1
	}

	// Blowout win: lead opponent by 64 when opponent is negative
	if s1 < 0 && s0-s1 >= 64 {
		return true, Team0
	}
	if s0 < 0 && s1-s0 >= 64 {
		return true, Team1
	}

	return false, Team0
}

// ApplyScore adds delta to the given team's score.
func (g *GameState) ApplyScore(team TeamIndex, delta int) {
	g.Scores[team] += delta
}
