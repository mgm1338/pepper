package ml

import (
	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

// Bid feature vector layout:
//   0–16  context features shared across all bid options at a decision point
//   17–18 opportunity cost features (who holds the high bid)
//   19–20 per-bid features appended once per candidate bid level
const (
	BidContextLen = 32
	BidActionLen  = 3
	BidTotalLen   = BidContextLen + BidActionLen // 35
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

	// Extended hand quality (9–13)
	"high_trump_count",    // trumps ranked king+ in best suit / 14
	"second_has_right",    // 1 if 2nd best suit has right bower
	"second_best_rank",    // best trump rank in 2nd suit / 13
	"off_kings",           // off-suit kings / 3
	"sister_suit_void",    // 1 if void in same-color non-trump suit (left bower depleted)
	"sister_suit_singleton", // 1 if singleton in same-color non-trump suit

	// Position and bidding state (15–18)
	"seat_pos_norm",     // position in bidding order / 5  (0=left of dealer, 1.0=dealer)
	"is_dealer",         // 1 if this seat is the dealer
	"current_high_norm", // current high bid / 7  (0 if no bids yet)
	"no_bids_yet",       // 1 if current_high == 0

	// Score context (19–22)
	"score_us",        // my team's score / 64
	"score_them",      // opponent score / 64
	"score_gap",       // (us - them) / 64  (signed)
	"closeout_window", // 1 if either team is at 48+ (within 16 of 64)

	// Bidding history context (23–28)
	"high_bid_is_teammate",   // 1 if the current high bid is held by a teammate
	"opponent_holding_high",  // 1 if opponent (not teammate) holds the high bid
	"seats_left_norm",        // seats remaining to bid after this one / 5
	"passes_so_far",          // players who have passed so far / 5
	"partner_has_bid",        // 1 if any teammate has placed a non-pass bid before this seat
	"last_to_bid_flag",       // 1 if this seat is last to act (dealer, all others passed)

	// Derived hand strength (29–31)
	"both_bowers",         // 1 if has both right and left bower in best suit (strong pepper signal)
	"partner_bid_level",   // partner's bid level / 8  (0 if no partner bid)
	"guaranteed_tricks",   // pre-computed trick estimate (bowers + high trumps + aces + voids) / 8

	// Per-bid features (32–34)
	"bid_level_norm", // bid / 8  (pass=0, 4=0.5, 5=0.625, 6=0.75, 7=0.875, pepper=1.0)
	"bid_is_pass",    // 1 if this is a pass
	"bid_is_pepper",  // 1 if this is a pepper call
}

// BidContext extracts the 19 shared context features for a bid decision.
// hand is the bidding player's full 8-card hand.
// dealer is the dealer seat index, seat is the current bidder's seat index.
// currentHigh is the current highest bid (0 if no bids yet).
// highSeat is the seat holding the current high bid (-1 if no bids yet).
// seatsLeft is the number of seats still to bid after this one.
// scores is the current game score.
func BidContext(seat int, hand []card.Card, dealer int, currentHigh int, highSeat int, seatsLeft int, scores [2]int, passesSoFar int, partnerHasBid bool, partnerBidLevel int) [BidContextLen]float32 {
	var f [BidContextLen]float32
	i := 0

	// Find the best and second-best trump suit by counting trump cards.
	// Use a fixed-size array (stack allocated) to avoid heap pressure on hot path.
	type suitInfo struct {
		suit      card.Suit
		count     int
		hasRight  bool
		hasLeft   bool
		bestRank  int
		highCount int // trumps ranked king+
	}

	var suits [4]suitInfo
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
				if tr >= card.TrumpRankKing {
					suits[s].highCount++
				}
			}
		}
	}

	// Find best and second-best suit (by trump count, tie-break by bestRank).
	best := suits[0]
	second := suitInfo{bestRank: -1}
	for s := 1; s < 4; s++ {
		si := suits[s]
		if si.count > best.count || (si.count == best.count && si.bestRank > best.bestRank) {
			second = best
			best = si
		} else if second.bestRank == -1 || si.count > second.count || (si.count == second.count && si.bestRank > second.bestRank) {
			second = si
		}
	}

	// Off-suit card counts relative to best trump suit.
	offAces, offKings, voids, singletons := 0, 0, 0, 0
	suitCounts := [4]int{}
	for _, c := range hand {
		if card.TrumpRank(c, best.suit) < 0 {
			suitCounts[c.Suit]++
			if c.Rank == card.Ace {
				offAces++
			} else if c.Rank == card.King {
				offKings++
			}
		}
	}
	sisterVoid, sisterSingleton := false, false
	for s := card.Suit(0); s < 4; s++ {
		if s == best.suit {
			continue
		}
		isSister := card.SameColor(s, best.suit)
		switch suitCounts[s] {
		case 0:
			voids++
			if isSister {
				sisterVoid = true
			}
		case 1:
			singletons++
			if isSister {
				sisterSingleton = true
			}
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

	// Extended hand quality features.
	f[i] = float32(best.highCount) / card.TotalTrumpCards
	i++ // high_trump_count
	if second.hasRight {
		f[i] = 1.0
	}
	i++ // second_has_right
	if second.bestRank >= 0 {
		f[i] = float32(second.bestRank) / card.TrumpRankRight
	}
	i++ // second_best_rank
	f[i] = float32(offKings) / 3.0
	i++ // off_kings
	if sisterVoid {
		f[i] = 1.0
	}
	i++ // sister_suit_void
	if sisterSingleton {
		f[i] = 1.0
	}
	i++ // sister_suit_singleton

	// Position and bidding state.
	pos := (seat - dealer + 6) % 6
	if pos == 0 {
		pos = 6 // dealer bids last, re-encode
	}
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

	// Bidding history context.
	if highSeat >= 0 && game.TeamOf(highSeat) == game.TeamOf(seat) {
		f[i] = 1.0
	}
	i++ // high_bid_is_teammate
	if highSeat >= 0 && game.TeamOf(highSeat) != game.TeamOf(seat) {
		f[i] = 1.0
	}
	i++ // opponent_holding_high
	f[i] = float32(seatsLeft) / 5.0
	i++ // seats_left_norm
	f[i] = float32(passesSoFar) / 5.0
	i++ // passes_so_far
	if partnerHasBid {
		f[i] = 1.0
	}
	i++ // partner_has_bid
	if seatsLeft == 0 && seat == dealer {
		f[i] = 1.0
	}
	i++ // last_to_bid_flag

	// Derived hand strength features.
	if best.hasRight && best.hasLeft {
		f[i] = 1.0
	}
	i++ // both_bowers

	f[i] = float32(partnerBidLevel) / float32(game.TotalTricks)
	i++ // partner_bid_level

	// guaranteed_tricks: weighted sum of reliable trick sources, normalized by total tricks.
	// Right bower: ~1.0 trick, left bower: ~0.9, each non-bower high trump (ace/king): ~0.65,
	// off-suit aces: ~0.6, off-suit kings: ~0.4, voids (ruffing): ~0.5.
	{
		gt := 0.0
		if best.hasRight { gt += 1.0 }
		if best.hasLeft  { gt += 0.9 }
		nonBowerHigh := best.highCount
		if best.hasRight { nonBowerHigh-- }
		if best.hasLeft  { nonBowerHigh-- }
		if nonBowerHigh < 0 { nonBowerHigh = 0 }
		gt += float64(nonBowerHigh) * 0.65
		gt += float64(offAces) * 0.6
		gt += float64(offKings) * 0.4
		gt += float64(voids) * 0.5
		f[i] = float32(gt / float64(game.TotalTricks))
	}
	i++ // guaranteed_tricks

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

// validBidTable[currentHigh] returns the candidate bid levels for that high.
// currentHigh < MinBid (4) all map to the same full set [0,4,5,6,7,8].
// Slices are shared read-only — callers must not modify.
var validBidTable = [8][]int{
	0: {game.PassBid, 4, 5, 6, 7, game.PepperBid},
	1: {game.PassBid, 4, 5, 6, 7, game.PepperBid},
	2: {game.PassBid, 4, 5, 6, 7, game.PepperBid},
	3: {game.PassBid, 4, 5, 6, 7, game.PepperBid},
	4: {game.PassBid, 5, 6, 7, game.PepperBid},
	5: {game.PassBid, 6, 7, game.PepperBid},
	6: {game.PassBid, 7, game.PepperBid},
	7: {game.PassBid, game.PepperBid},
}

// ValidBidLevels returns the candidate bid levels for this decision point,
// including pass (0), all valid overcall levels, and pepper (8).
// Returns a shared slice — callers must not modify.
func ValidBidLevels(currentHigh int) []int {
	if currentHigh < 0 || currentHigh >= len(validBidTable) {
		currentHigh = 0
	}
	return validBidTable[currentHigh]
}
