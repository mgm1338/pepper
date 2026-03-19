package ml

import (
	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

// Bid feature vector layout:
//   0–16  context features shared across all bid options at a decision point
//   17–19 per-bid features appended once per candidate bid level
const (
	BidContextLen = 17
	BidActionLen  = 3
	BidTotalLen   = BidContextLen + BidActionLen // 20
)

// BidFeatureNames provides human-readable column names for the full bid feature vector.
var BidFeatureNames = [BidTotalLen]string{
	// Hand composition relative to best possible trump suit (0–8)
	"best_trump_count",     // trump cards in best suit / 14
	"best_has_right",       // 1 if has right bower in best suit
	"best_has_left",        // 1 if has left bower in best suit
	"best_trump_rank",      // highest trump rank in best suit / 13
	"best_off_aces",        // off-suit aces (relative to best suit) / 6
	"best_void_suits",      // void non-trump suits / 3
	"best_singleton_suits", // singleton non-trump suits / 3
	"suit_dominance",       // (best trump count - 2nd best trump count) / 14
	"second_trump_count",   // trump count in 2nd best suit / 14

	// Position and bidding state (9–12)
	"seat_pos_norm",    // position in bidding order / 5  (0=left of dealer, 1.0=dealer)
	"is_dealer",        // 1 if this seat is the dealer
	"current_high_norm", // current high bid / 7  (0 if no bids yet)
	"no_bids_yet",      // 1 if current_high == 0

	// Score context (13–16)
	"score_us",         // my team's score / 64
	"score_them",       // opponent score / 64
	"score_gap",        // (us - them) / 64  (signed)
	"closeout_window",  // 1 if either team is at 48+ (within 16 of 64)

	// Per-bid features (17–19)
	"bid_level_norm", // bid / 8  (pass=0, 4=0.5, 5=0.625, 6=0.75, 7=0.875, pepper=1.0)
	"bid_is_pass",    // 1 if this is a pass
	"bid_is_pepper",  // 1 if this is a pepper call
}

// BidContext extracts the 17 shared context features for a bid decision.
// hand is the bidding player's full 8-card hand.
// dealer is the dealer seat index, seat is the current bidder's seat index.
// currentHigh is the current highest bid (0 if no bids yet).
// scores is the current game score.
func BidContext(seat int, hand []card.Card, dealer int, currentHigh int, scores [2]int) [BidContextLen]float32 {
	var f [BidContextLen]float32
	i := 0

	// Find the best and second-best trump suit by counting trump cards.
	type suitInfo struct {
		suit     card.Suit
		count    int
		hasRight bool
		hasLeft  bool
		bestRank int
	}

	suits := make([]suitInfo, 4)
	for s := 0; s < 4; s++ {
		suits[s].suit = card.Suit(s)
		suits[s].bestRank = -1
	}

	for _, c := range hand {
		for s := 0; s < 4; s++ {
			tr := card.TrumpRank(c, card.Suit(s))
			if tr >= 0 {
				suits[s].count++
				if tr > suits[s].bestRank {
					suits[s].bestRank = tr
				}
				if card.IsRightBower(c, card.Suit(s)) {
					suits[s].hasRight = true
				} else if card.IsLeftBower(c, card.Suit(s)) {
					suits[s].hasLeft = true
				}
			}
		}
	}

	// Sort to find best and second-best suit (by trump count, tie-break by bestRank).
	best := suits[0]
	second := suitInfo{bestRank: -1}
	for _, si := range suits[1:] {
		if si.count > best.count || (si.count == best.count && si.bestRank > best.bestRank) {
			second = best
			best = si
		} else if second.bestRank == -1 || si.count > second.count || (si.count == second.count && si.bestRank > second.bestRank) {
			second = si
		}
	}

	// Off-suit aces and suit distribution relative to best trump suit.
	offAces, voids, singletons := 0, 0, 0
	suitCounts := [4]int{}
	for _, c := range hand {
		if card.TrumpRank(c, best.suit) < 0 {
			suitCounts[c.Suit]++
			if c.Rank == card.Ace {
				offAces++
			}
		}
	}
	for s := card.Suit(0); s < 4; s++ {
		if s == best.suit {
			continue
		}
		switch suitCounts[s] {
		case 0:
			voids++
		case 1:
			singletons++
		}
	}

	// Hand composition features.
	f[i] = float32(best.count) / card.TotalTrumpCards
	i++ // best_trump_count
	if best.hasRight {
		f[i] = 1.0
	}
	i++ // best_has_right
	if best.hasLeft {
		f[i] = 1.0
	}
	i++ // best_has_left
	if best.bestRank >= 0 {
		f[i] = float32(best.bestRank) / card.TrumpRankRight
	}
	i++ // best_trump_rank
	f[i] = float32(offAces) / card.NonTrumpRankAce
	i++ // best_off_aces
	f[i] = float32(voids) / 3.0
	i++ // best_void_suits
	f[i] = float32(singletons) / 3.0
	i++ // best_singleton_suits

	dominance := best.count - second.count
	if dominance < 0 {
		dominance = 0
	}
	f[i] = float32(dominance) / card.TotalTrumpCards
	i++ // suit_dominance
	f[i] = float32(second.count) / card.TotalTrumpCards
	i++ // second_trump_count

	// Position and bidding state.
	// Bidding order starts left of dealer. Dealer bids last (position 5).
	// Position in round: 0 = first to act (seat left of dealer), 5 = dealer.
	pos := (seat - dealer + 6) % 6
	if pos == 0 {
		pos = 6 // dealer bids last, re-encode
	}
	// Now pos is 1-6 where 1=left of dealer, 6=dealer. Normalize to 0-1.
	f[i] = float32(pos-1) / 5.0
	i++ // seat_pos_norm
	if seat == dealer {
		f[i] = 1.0
	}
	i++ // is_dealer

	if currentHigh > 0 {
		f[i] = float32(currentHigh) / 7.0
	}
	i++ // current_high_norm
	if currentHigh == 0 {
		f[i] = 1.0
	}
	i++ // no_bids_yet

	// Score context.
	myTeam := game.TeamOf(seat)
	myScore := scores[myTeam]
	themScore := scores[1-myTeam]
	f[i] = float32(myScore) / game.WinScore
	i++ // score_us
	f[i] = float32(themScore) / game.WinScore
	i++ // score_them
	f[i] = float32(myScore-themScore) / game.WinScore
	i++ // score_gap
	if myScore >= game.CloseoutScore || themScore >= game.CloseoutScore {
		f[i] = 1.0
	}
	i++ // closeout_window

	_ = i // i == BidContextLen here
	return f
}

// AppendBidAction appends the 3 per-bid features to a context vector.
// bidLevel: 0=pass, 4-7=normal bid, 8=pepper.
func AppendBidAction(ctx [BidContextLen]float32, bidLevel int) [BidTotalLen]float32 {
	var f [BidTotalLen]float32
	copy(f[:BidContextLen], ctx[:])

	i := BidContextLen
	f[i] = float32(bidLevel) / game.TotalTricks
	i++ // bid_level_norm
	if bidLevel == game.PassBid {
		f[i] = 1.0
	}
	i++ // bid_is_pass
	if bidLevel == game.PepperBid {
		f[i] = 1.0
	}
	i++ // bid_is_pepper

	_ = i
	return f
}

// ValidBidLevels returns the candidate bid levels for this decision point,
// including pass (0), all valid overcall levels, and pepper (8).
func ValidBidLevels(currentHigh int) []int {
	levels := []int{game.PassBid} // always can pass
	min := game.MinBid
	if currentHigh >= game.MinBid {
		min = currentHigh + 1
	}
	for level := min; level <= 7; level++ {
		levels = append(levels, level)
	}
	levels = append(levels, game.PepperBid)
	return levels
}
