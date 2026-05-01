package ml

import (
	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

// Bid feature vector layout:
//   0–31  original context features (unchanged)
//   32–53 extended context features (hand texture, suit breakdown, opponent signals)
//   54–59 per-bid action features
const (
	BidContextLen = 54
	BidActionLen  = 6
	BidTotalLen   = BidContextLen + BidActionLen // 60

	// bidCtxGTIdx is the index of guaranteed_tricks in the context vector.
	// AppendBidAction reads this to compute bid-relative stretch features.
	bidCtxGTIdx = 31
)

// BidFeatureNames provides human-readable column names for the full bid feature vector.
var BidFeatureNames = [BidTotalLen]string{
	// Hand composition relative to best possible trump suit (0–8)
	"best_trump_count",       // trump cards in best suit / 14
	"best_has_right",         // 1 if has right bower in best suit
	"best_has_left",          // 1 if has left bower in best suit
	"best_trump_rank",        // highest trump rank in best suit / 13
	"best_off_aces",          // off-suit aces (relative to best suit) / 6
	"best_void_suits",        // void non-trump suits / 3
	"best_singleton_suits",   // singleton non-trump suits / 3
	"suit_dominance",         // (best trump count - 2nd best trump count) / 14
	"second_trump_count",     // trump count in 2nd best suit / 14

	// Extended hand quality (9–14)
	"high_trump_count",       // trumps ranked king+ in best suit / 14
	"second_has_right",       // 1 if 2nd best suit has right bower
	"second_best_rank",       // best trump rank in 2nd suit / 13
	"off_kings",              // off-suit kings / 3
	"sister_suit_void",       // 1 if void in same-color non-trump suit
	"sister_suit_singleton",  // 1 if singleton in same-color non-trump suit

	// Position and bidding state (15–18)
	"seat_pos_norm",          // position in bidding order / 5
	"is_dealer",              // 1 if this seat is the dealer
	"current_high_norm",      // current high bid / 7
	"no_bids_yet",            // 1 if current_high == 0

	// Score context (19–22)
	"score_us",               // my team's score / 64
	"score_them",             // opponent score / 64
	"score_gap",              // (us - them) / 64
	"closeout_window",        // 1 if either team is at 48+

	// Bidding history context (23–28)
	"high_bid_is_teammate",   // 1 if the current high bid is held by a teammate
	"opponent_holding_high",  // 1 if opponent holds the high bid
	"seats_left_norm",        // seats remaining after this one / 5
	"passes_so_far",          // players who passed so far / 5
	"partner_has_bid",        // 1 if any teammate has placed a non-pass bid
	"last_to_bid_flag",       // 1 if this seat is last to act

	// Derived hand strength (29–31)
	"both_bowers",            // 1 if has both right and left bower in best suit
	"partner_bid_level",      // partner's bid level / 8
	"guaranteed_tricks",      // pre-computed trick estimate / 8

	// Third/fourth suit breakdown (32–35)
	"third_trump_count",      // 3rd best suit trump count / 14
	"third_has_right",        // 1 if 3rd suit has right bower
	"third_best_rank",        // 3rd suit best rank / 13
	"fourth_trump_count",     // 4th suit trump count / 14

	// Hand texture (36–40)
	"off_doubletons",         // doubleton off-suit count / 3
	"total_off_power",        // (off_aces + off_kings) / 9
	"strong_hand",            // 1 if right bower + 4+ non-bower trumps in best suit
	"has_4plus_trump",        // 1 if best_count >= 4
	"has_5plus_trump",        // 1 if best_count >= 5

	// Second suit extras (41–43)
	"second_has_left",        // 1 if 2nd suit has left bower
	"second_high_count",      // high trumps (king+) in 2nd suit / 14
	"second_sister_void",     // 1 if void in 2nd suit's partner suit

	// Opponent signals (44–48)
	"opp1_has_bid",           // 1 if highest-bidding opponent has bid
	"opp2_has_bid",           // 1 if 2nd opponent has bid
	"opp1_bid_level_norm",    // highest opp bid / 8
	"opp2_bid_level_norm",    // 2nd highest opp bid / 8
	"all_opps_passed",        // 1 if all opponents passed

	// Score near-win signals (49–50)
	"score_near_win_us",      // 1 if myScore >= 56
	"score_near_win_them",    // 1 if themScore >= 56

	// Competitive context (51–53)
	"bid_count_norm",         // non-pass bids placed so far / 5
	"suits_with_3plus",       // suits with 3+ trumps / 4
	"trump_concentration",    // best_count / (best_count + second_count + 1)

	// Per-bid features (54–59)
	"bid_level_norm",         // bid / 8
	"bid_is_pass",            // 1 if this is a pass
	"bid_is_pepper",          // 1 if this is a pepper call
	"bid_stretch",            // bid/8 - guaranteed_tricks/8 (how much we're overreaching)
	"is_minimum_raise",       // 1 if bid == currentHigh+1 (or 4 if no current high)
	"bid_gap_from_high",      // (bid - currentHigh) / 8 (0 if pass or no current high)
}

// BidContext extracts the 54 shared context features for a bid decision.
// allBids contains the bid level each seat has placed so far (0 = pass or not yet bid).
func BidContext(seat int, hand []card.Card, dealer int, currentHigh int, highSeat int, seatsLeft int, scores [2]int, passesSoFar int, partnerHasBid bool, partnerBidLevel int, allBids [6]int) [BidContextLen]float32 {
	var f [BidContextLen]float32
	i := 0

	type suitInfo struct {
		suit      card.Suit
		count     int
		hasRight  bool
		hasLeft   bool
		bestRank  int
		highCount int
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

	// Sort all 4 suits by (count desc, bestRank desc).
	ranked := suits
	for a := 0; a < 3; a++ {
		for b := a + 1; b < 4; b++ {
			if ranked[b].count > ranked[a].count ||
				(ranked[b].count == ranked[a].count && ranked[b].bestRank > ranked[a].bestRank) {
				ranked[a], ranked[b] = ranked[b], ranked[a]
			}
		}
	}
	best   := ranked[0]
	second := ranked[1]
	third  := ranked[2]
	fourth := ranked[3]

	// Off-suit card counts relative to best trump suit.
	offAces, offKings, voids, singletons, doubletons := 0, 0, 0, 0, 0
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
		case 2:
			doubletons++
		}
	}

	// Raw suit counts for second-suit sister void check.
	var rawSuitCount [4]int
	for _, c := range hand {
		rawSuitCount[c.Suit]++
	}
	secondSisterVoid := rawSuitCount[card.PartnerSuit(second.suit)] == 0

	// --- Original 32 context features (indices 0–31) ---

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

	pos := (seat - dealer + 6) % 6
	if pos == 0 {
		pos = 6
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

	if highSeat >= 0 && game.TeamOf(highSeat) == myTeam {
		f[i] = 1.0
	}
	i++ // high_bid_is_teammate
	if highSeat >= 0 && game.TeamOf(highSeat) != myTeam {
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

	if best.hasRight && best.hasLeft {
		f[i] = 1.0
	}
	i++ // both_bowers

	f[i] = float32(partnerBidLevel) / float32(game.TotalTricks)
	i++ // partner_bid_level

	// guaranteed_tricks at index bidCtxGTIdx (31)
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
	i++ // guaranteed_tricks (index 31 = bidCtxGTIdx)

	// --- New context features (indices 32–53) ---

	// Third/fourth suit breakdown.
	f[i] = float32(third.count) / card.TotalTrumpCards
	i++ // third_trump_count
	if third.hasRight {
		f[i] = 1.0
	}
	i++ // third_has_right
	if third.bestRank >= 0 {
		f[i] = float32(third.bestRank) / card.TrumpRankRight
	}
	i++ // third_best_rank
	f[i] = float32(fourth.count) / card.TotalTrumpCards
	i++ // fourth_trump_count

	// Hand texture.
	f[i] = float32(doubletons) / 3.0
	i++ // off_doubletons
	f[i] = float32(offAces+offKings) / 9.0
	i++ // total_off_power
	nonBowerTrumps := best.count
	if best.hasRight { nonBowerTrumps-- }
	if best.hasLeft  { nonBowerTrumps-- }
	if best.hasRight && nonBowerTrumps >= 4 {
		f[i] = 1.0
	}
	i++ // strong_hand
	if best.count >= 4 {
		f[i] = 1.0
	}
	i++ // has_4plus_trump
	if best.count >= 5 {
		f[i] = 1.0
	}
	i++ // has_5plus_trump

	// Second suit extras.
	if second.hasLeft {
		f[i] = 1.0
	}
	i++ // second_has_left
	f[i] = float32(second.highCount) / card.TotalTrumpCards
	i++ // second_high_count
	if secondSisterVoid {
		f[i] = 1.0
	}
	i++ // second_sister_void

	// Opponent signals: find top-2 opponent bids.
	opp1Bid, opp2Bid := 0, 0
	for s := 0; s < 6; s++ {
		if s == seat || game.TeamOf(s) == myTeam {
			continue
		}
		b := allBids[s]
		if b > opp1Bid {
			opp2Bid = opp1Bid
			opp1Bid = b
		} else if b > opp2Bid {
			opp2Bid = b
		}
	}
	if opp1Bid > 0 {
		f[i] = 1.0
	}
	i++ // opp1_has_bid
	if opp2Bid > 0 {
		f[i] = 1.0
	}
	i++ // opp2_has_bid
	f[i] = float32(opp1Bid) / float32(game.TotalTricks)
	i++ // opp1_bid_level_norm
	f[i] = float32(opp2Bid) / float32(game.TotalTricks)
	i++ // opp2_bid_level_norm
	if opp1Bid == 0 {
		f[i] = 1.0
	}
	i++ // all_opps_passed

	// Score near-win signals.
	if myScore >= game.WinScore-8 {
		f[i] = 1.0
	}
	i++ // score_near_win_us
	if themScore >= game.WinScore-8 {
		f[i] = 1.0
	}
	i++ // score_near_win_them

	// Competitive context.
	bidCount := 0
	for s := 0; s < 6; s++ {
		if s != seat && allBids[s] > 0 {
			bidCount++
		}
	}
	f[i] = float32(bidCount) / 5.0
	i++ // bid_count_norm

	suitsWith3plus := 0
	for _, si := range ranked {
		if si.count >= 3 {
			suitsWith3plus++
		}
	}
	f[i] = float32(suitsWith3plus) / 4.0
	i++ // suits_with_3plus

	denom := float32(best.count + second.count + 1)
	f[i] = float32(best.count) / denom
	i++ // trump_concentration

	_ = i // i == BidContextLen here
	return f
}

// AppendBidAction appends the 6 per-bid features to a context vector.
// currentHigh is the current highest bid (0 if none).
func AppendBidAction(ctx [BidContextLen]float32, bidLevel int, currentHigh int) [BidTotalLen]float32 {
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

	// Bid stretch: how much this bid exceeds the hand's guaranteed-tricks estimate.
	gtNorm := ctx[bidCtxGTIdx]
	if bidLevel != game.PassBid {
		f[i] = float32(bidLevel)/game.TotalTricks - gtNorm
	}
	i++ // bid_stretch

	// Minimum raise flag.
	minValid := game.MinBid
	if currentHigh >= game.MinBid {
		minValid = currentHigh + 1
	}
	if bidLevel == minValid {
		f[i] = 1.0
	}
	i++ // is_minimum_raise

	// Gap from current high (how much we're overcalling).
	if bidLevel != game.PassBid && currentHigh > 0 {
		f[i] = float32(bidLevel-currentHigh) / game.TotalTricks
	}
	i++ // bid_gap_from_high

	_ = i
	return f
}

// validBidTable[currentHigh] returns the candidate bid levels for that high.
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

// ValidBidLevels returns the candidate bid levels for this decision point.
func ValidBidLevels(currentHigh int) []int {
	if currentHigh < 0 || currentHigh >= len(validBidTable) {
		currentHigh = 0
	}
	return validBidTable[currentHigh]
}
