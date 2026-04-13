package game

import "testing"

func TestScoreHand_madeBid(t *testing.T) {
	// bidder seat 0 (team 0), bid 4, takes 5 tricks
	r := ScoreHand(0, 4, false, [2]int{5, 3})
	if !r.MadeBid {
		t.Error("should have made bid")
	}
	if r.ScoreDelta[0] != 5 {
		t.Errorf("team0 delta = %d, want 5", r.ScoreDelta[0])
	}
	if r.ScoreDelta[1] != 3 {
		t.Errorf("team1 delta = %d, want 3", r.ScoreDelta[1])
	}
}

func TestScoreHand_setBid(t *testing.T) {
	// bid 5, took 3
	r := ScoreHand(1, 5, false, [2]int{3, 3})
	if r.MadeBid {
		t.Error("should be set")
	}
	// bidder is team 1 (seat 1 % 2 = 1)
	if r.ScoreDelta[1] != -5 {
		t.Errorf("team1 delta = %d, want -5", r.ScoreDelta[1])
	}
	if r.ScoreDelta[0] != 3 {
		t.Errorf("team0 delta = %d, want 3", r.ScoreDelta[0])
	}
}

func TestScoreHand_pepperMade(t *testing.T) {
	r := ScoreHand(0, PepperBid, true, [2]int{TotalTricks, 0})
	if !r.MadeBid {
		t.Error("pepper taking all tricks should be made")
	}
	if r.ScoreDelta[0] != PepperPoints {
		t.Errorf("team0 delta = %d, want %d", r.ScoreDelta[0], PepperPoints)
	}
	if r.ScoreDelta[1] != 0 {
		t.Errorf("team1 delta = %d, want 0", r.ScoreDelta[1])
	}
}

func TestScoreHand_pepperSet(t *testing.T) {
	// caller team 0, takes 6, opponents take 2
	r := ScoreHand(0, PepperBid, true, [2]int{6, 2})
	if r.MadeBid {
		t.Error("pepper missing a trick should not be made")
	}
	if r.ScoreDelta[0] != -PepperPoints {
		t.Errorf("team0 delta = %d, want %d", r.ScoreDelta[0], -PepperPoints)
	}
	if r.ScoreDelta[1] != 4 {
		t.Errorf("team1 delta = %d, want 4 (2 tricks * 2)", r.ScoreDelta[1])
	}
}

func TestScoreHand_moonMade(t *testing.T) {
	// bid 7, took all 8
	r := ScoreHand(2, 7, false, [2]int{8, 0})
	if !r.MadeBid {
		t.Error("moon taken should be made")
	}
	if r.ScoreDelta[0] != 8 {
		t.Errorf("team0 delta = %d, want 8", r.ScoreDelta[0])
	}
}

func TestScoreHand_moonSet(t *testing.T) {
	r := ScoreHand(0, 7, false, [2]int{6, 2})
	if r.MadeBid {
		t.Error("moon not reached should be set")
	}
	if r.ScoreDelta[0] != -7 {
		t.Errorf("team0 delta = %d, want -7", r.ScoreDelta[0])
	}
	if r.ScoreDelta[1] != 2 {
		t.Errorf("team1 delta = %d, want 2", r.ScoreDelta[1])
	}
}
