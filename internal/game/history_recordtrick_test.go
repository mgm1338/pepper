package game

import (
	"testing"

	"github.com/max/pepper/internal/card"
)

// --- RecordTrick ---

func TestHistoryRecordTrick_addsCards(t *testing.T) {
	var h HandHistory
	trick := buildTrick(seat0, sp,
		play{trumpAce, seat0},
		play{heartAce, seat1},
		play{diamondAce, seat2},
	)
	h.RecordTrick(trick.Cards[:trick.NCards])

	played := h.PlayedSlice()
	if len(played) != 3 {
		t.Fatalf("after RecordTrick 3-card trick, Played = %d, want 3", len(played))
	}
}

func TestHistoryRecordTrick_preservesOrder(t *testing.T) {
	var h HandHistory
	trick := buildTrick(seat0, sp,
		play{trumpAce, seat0},
		play{heartKing, seat1},
		play{heartNine, seat2},
	)
	h.RecordTrick(trick.Cards[:trick.NCards])

	played := h.PlayedSlice()
	if !played[0].Equal(trumpAce) {
		t.Errorf("played[0] = %v, want trumpAce", played[0])
	}
	if !played[1].Equal(heartKing) {
		t.Errorf("played[1] = %v, want heartKing", played[1])
	}
	if !played[2].Equal(heartNine) {
		t.Errorf("played[2] = %v, want heartNine", played[2])
	}
}

func TestHistoryRecordTrick_multipleAccumulate(t *testing.T) {
	var h HandHistory

	trick1 := buildTrick(seat0, sp,
		play{trumpAce, seat0},
		play{trumpKing, seat1},
	)
	trick2 := buildTrick(seat2, sp,
		play{heartAce, seat2},
		play{heartNine, seat3},
	)
	h.RecordTrick(trick1.Cards[:trick1.NCards])
	h.RecordTrick(trick2.Cards[:trick2.NCards])

	played := h.PlayedSlice()
	if len(played) != 4 {
		t.Fatalf("after 2 tricks, Played = %d, want 4", len(played))
	}
	if !played[2].Equal(heartAce) {
		t.Errorf("played[2] = %v, want heartAce", played[2])
	}
}

func TestHistoryRecordTrick_isSeen(t *testing.T) {
	var h HandHistory
	trick := buildTrick(seat0, sp, play{trumpAce, seat0})
	h.RecordTrick(trick.Cards[:trick.NCards])

	if !h.IsSeen(trumpAce) {
		t.Error("trumpAce should be seen after RecordTrick")
	}
	if h.IsSeen(trumpKing) {
		t.Error("trumpKing should not be seen")
	}
}

func TestHistoryRecordTrick_resetClears(t *testing.T) {
	var h HandHistory
	trick := buildTrick(seat0, sp, play{trumpAce, seat0}, play{heartAce, seat1})
	h.RecordTrick(trick.Cards[:trick.NCards])
	h.Reset()

	if len(h.PlayedSlice()) != 0 {
		t.Errorf("after Reset, Played = %d, want 0", len(h.PlayedSlice()))
	}
}

func TestHistoryRecordTrick_sixCardTrick(t *testing.T) {
	var h HandHistory
	trick := NewTrick(0, sp)
	cards := []card.Card{trumpAce, trumpKing, trumpNine, heartAce, heartKing, diamondAce}
	for i, c := range cards {
		trick.Add(c, i)
	}
	h.RecordTrick(trick.Cards[:trick.NCards])

	played := h.PlayedSlice()
	if len(played) != 6 {
		t.Fatalf("6-card trick: Played = %d, want 6", len(played))
	}
	for i, c := range cards {
		if !played[i].Equal(c) {
			t.Errorf("played[%d] = %v, want %v", i, played[i], c)
		}
	}
}

func TestHistoryRecordTrick_trumpPlayedCount(t *testing.T) {
	var h HandHistory
	trump := card.Spades
	trick := buildTrick(seat0, trump,
		play{trumpAce, seat0},
		play{heartAce, seat1},
		play{trumpKing, seat2},
	)
	h.RecordTrick(trick.Cards[:trick.NCards])

	n := h.TrumpPlayed(trump)
	if n != 2 {
		t.Errorf("TrumpPlayed = %d, want 2", n)
	}
}
