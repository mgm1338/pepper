package game

import "testing"

func TestTeamOf(t *testing.T) {
	for seat := 0; seat < 6; seat++ {
		want := TeamIndex(seat % 2)
		if got := TeamOf(seat); got != want {
			t.Errorf("TeamOf(%d) = %d, want %d", seat, got, want)
		}
	}
}

func TestPartners(t *testing.T) {
	cases := map[int][2]int{
		0: {2, 4},
		2: {0, 4},
		4: {0, 2},
		1: {3, 5},
		3: {1, 5},
		5: {1, 3},
	}
	for seat, want := range cases {
		got := Partners(seat)
		if got != want {
			t.Errorf("Partners(%d) = %v, want %v", seat, got, want)
		}
	}
}

func TestNewGame(t *testing.T) {
	g := NewGame(3)
	if g.Dealer != 3 {
		t.Errorf("Dealer = %d, want 3", g.Dealer)
	}
	if g.Scores[0] != 0 || g.Scores[1] != 0 {
		t.Error("new game scores should be 0")
	}
	if g.Round != 0 {
		t.Error("new game round should be 0")
	}
}

func TestNextDealer(t *testing.T) {
	g := NewGame(5)
	g.NextDealer()
	if g.Dealer != 0 {
		t.Errorf("Dealer = %d, want 0 (wrap)", g.Dealer)
	}
	if g.Round != 1 {
		t.Errorf("Round = %d, want 1", g.Round)
	}
	g.NextDealer()
	if g.Dealer != 1 {
		t.Errorf("Dealer = %d, want 1", g.Dealer)
	}
}

func TestIsOver(t *testing.T) {
	cases := []struct {
		s0, s1 int
		over   bool
		winner TeamIndex
	}{
		{0, 0, false, Team0},
		{63, 30, false, Team0},
		{64, 30, true, Team0},
		{30, 64, true, Team1},
		{50, -5, false, Team0},  // not enough for blowout, negative blocks normal
		{70, -7, true, Team0},   // 70 - (-7) = 77 >= 64 blowout
		{-10, 55, true, Team1},  // 55 - (-10) = 65 blowout
	}
	for _, tc := range cases {
		g := &GameState{Scores: [2]int{tc.s0, tc.s1}}
		over, w := g.IsOver()
		if over != tc.over {
			t.Errorf("IsOver(%d,%d) over = %v, want %v", tc.s0, tc.s1, over, tc.over)
		}
		if over && w != tc.winner {
			t.Errorf("IsOver(%d,%d) winner = %d, want %d", tc.s0, tc.s1, w, tc.winner)
		}
	}
}

func TestApplyScore(t *testing.T) {
	g := NewGame(0)
	g.ApplyScore(Team0, 5)
	g.ApplyScore(Team1, 3)
	g.ApplyScore(Team0, -2)
	if g.Scores[0] != 3 {
		t.Errorf("team0 = %d, want 3", g.Scores[0])
	}
	if g.Scores[1] != 3 {
		t.Errorf("team1 = %d, want 3", g.Scores[1])
	}
}
