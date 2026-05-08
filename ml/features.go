package ml

import (
	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

// Feature vector layout: ContextFeatureLen context features shared across all legal card
// choices at a decision point, followed by CardFeatureLen card-specific features appended
// once per candidate card. One training row = one (decision point, candidate card) pair.
const (
	ContextFeatureLen = 74
	CardFeatureLen    = 8
	TotalFeatureLen   = ContextFeatureLen + CardFeatureLen
)

// FeatureNames provides human-readable column names for the full feature vector.
// Indices 0–73 are context, 74–81 are per-card.
var FeatureNames = [TotalFeatureLen]string{
	// Position / role (0–3)
	"seat_rel_bidder",   // (seat - bidderSeat + 6) % 6, normalized 0–1
	"is_bidding_team",   // 1 if on bidder's team
	"position_in_trick", // how far into trick order we are (0 = leading)
	"players_after_me",  // remaining players to act this trick, normalized

	// Trick progress (4–9)
	"trick_number",      // 0–7, normalized
	"tricks_taken_us",   // our team's tricks so far
	"tricks_taken_them", // opponent team's tricks so far
	"tricks_needed_us",  // tricks bidding team still needs to make bid
	"tricks_to_euchre",  // tricks defense still needs to set bidder
	"tricks_remaining",  // tricks left in hand including this one

	// Current trick context (10–15)
	"cards_in_trick",     // cards played so far this trick, normalized
	"am_winning_trick",   // 1 if I'm the current trick winner
	"partner_winning",    // 1 if my partner is winning
	"bidder_has_played",  // 1 if bidder has already played this trick
	"winning_trump_rank", // trump rank of current winner / 13 (0 if off-suit)
	"is_leader",          // 1 if I'm leading (no cards in trick yet)

	// My full hand composition (16–23)
	"trump_in_hand",      // trump count / 14
	"has_right",          // 1 if holding right bower
	"has_left",           // 1 if holding left bower
	"highest_trump_rank", // rank of my best trump / 13 (0 if none)
	"off_suit_aces",      // off-suit aces count / 6
	"void_suits",         // void non-trump suits / 3
	"singleton_suits",    // singleton non-trump suits / 3
	"cards_in_hand",      // total cards in hand / 8

	// Trump field knowledge (24–28)
	"trump_remaining",      // trump not yet played / 14
	"trump_played",         // trump played so far / 14
	"right_bowers_played",  // right bowers played / 2
	"left_bowers_played",   // left bowers played / 2
	"is_top_trump_in_hand", // 1 if my best trump is the current top unplayed trump

	// Score context (29–32)
	"score_us",         // my team's score / 64
	"score_them",       // opponent score / 64
	"score_gap",        // (us - them) / 64, signed
	"closeout_window",  // 1 if either team is within 16 of 64

	// Bid context (33)
	"bid_amount_norm", // bid amount / 8 (pepper treated as 8)

	// Trump dominance (34–36)
	"my_trump_fraction", // my trump / all trump remaining (0 if none left)
	"opp_trump_est",     // (trump_remaining - my_trump) / 14
	"is_last_to_play",   // 1 if I'm the last to act in this trick

	// Trick winner knowledge (37–39)
	"winner_is_unbeatable",      // 1 if current winner card is highest remaining of its kind
	"i_can_beat_winner",         // 1 if I hold any card that beats the current winner
	"partner_winning_unbeatable", // 1 if partner is winning with an unbeatable card

	// Off-suit field knowledge (40–44)
	"offsuit1_remaining",  // remaining cards in non-trump suit 1 (suit order 0-3, skipping trump) / 12
	"offsuit2_remaining",  // remaining cards in non-trump suit 2 / 12
	"offsuit3_remaining",  // remaining cards in non-trump suit 3 / 12
	"led_card_rank_norm",  // rank of led card normalized (0 if leading)
	"led_suit_remaining",  // remaining cards in led suit / 12 (0 if leading)

	// Void knowledge (45–50)
	"partner_void_1",    // fraction of partners void in off-suit 1 (0, 0.5, 1.0)
	"partner_void_2",    // fraction of partners void in off-suit 2
	"partner_void_3",    // fraction of partners void in off-suit 3
	"opp_void_count_1",  // fraction of opponents void in off-suit 1 (0–1)
	"opp_void_count_2",  // fraction of opponents void in off-suit 2
	"opp_void_count_3",  // fraction of opponents void in off-suit 3

	// Trick dynamics (51–55)
	"partner_has_played",      // 1 if any partner has already played in this trick
	"suit_led_is_trump",       // 1 if the led card is trump (0 if leading)
	"trump_in_current_trick",  // trump cards played so far this trick / 6
	"top_of_led_suit_in_hand", // 1 if I hold the highest remaining card of the led suit (0 if leading)
	"winner_seat_rel_norm",    // (winner_seat - seat + 6) % 6 / 5 (0 if leading)

	// Per-seat void flags (56–70): for each of 5 relative seats, void in off-suits 1/2/3
	// Relative seat order: (seat+1)%6, (seat+2)%6, (seat+3)%6, (seat+4)%6, (seat+5)%6
	"seat1_void_1", "seat1_void_2", "seat1_void_3",
	"seat2_void_1", "seat2_void_2", "seat2_void_3",
	"seat3_void_1", "seat3_void_2", "seat3_void_3",
	"seat4_void_1", "seat4_void_2", "seat4_void_3",
	"seat5_void_1", "seat5_void_2", "seat5_void_3",

	// Trick leader identity (71–72)
	"led_by_bidder",  // 1 if the trick was led by the bidder (0 when leading)
	"led_by_partner", // 1 if the trick was led by one of my partners (0 when leading)

	// Recent trump density (73)
	"trump_rate_recent", // fraction of trump in the last 2 completed tricks (0 if fewer than 1 trick done)

	// Per-card features (74–81)
	"card_is_trump",          // 1 if candidate card is trump
	"card_rank_norm",         // trump rank / 13 if trump, else off-suit rank / 6
	"card_is_top",            // 1 if candidate is the current top unplayed trump
	"can_beat_winner",        // 1 if this card beats the current trick leader
	"is_overtrump",           // 1 if card is trump and current winner is also trump
	"card_suit_count",        // cards of this suit in my hand / 8
	"card_is_top_in_suit",    // 1 if no higher-ranked card of this non-trump suit remains unplayed
	"card_suit_remaining",    // remaining cards of this card's effective suit globally / 12
}

// ExtractContext builds the 37 shared context features for a play decision.
// hand is the player's full hand (not filtered to legal plays).
// activePlayers is 4 for pepper hands, 6 for normal.
func ExtractContext(seat int, hand []card.Card, state game.TrickState, activePlayers int) [ContextFeatureLen]float32 {
	var f [ContextFeatureLen]float32
	i := 0

	trump := state.Trump
	trick := state.Trick
	bidderSeat := state.BidderSeat
	cardsInTrick := trick.NCards

	maxPos := float32(activePlayers - 1)
	if maxPos < 1 {
		maxPos = 1
	}

	// --- Position / role (0–3) ---
	f[i] = float32((seat-bidderSeat+6)%6) / 5.0
	i++ // seat_rel_bidder
	if game.TeamOf(seat) == game.TeamOf(bidderSeat) {
		f[i] = 1.0
	}
	i++ // is_bidding_team
	f[i] = float32(cardsInTrick) / maxPos
	i++ // position_in_trick
	remaining := activePlayers - 1 - cardsInTrick
	if remaining < 0 {
		remaining = 0
	}
	f[i] = float32(remaining) / maxPos
	i++ // players_after_me

	// --- Trick progress (4–9) ---
	trickNum := state.TrickNumber
	f[i] = float32(trickNum) / (game.TotalTricks - 1)
	i++ // trick_number

	myTeam := game.TeamOf(seat)
	bidderTeam := game.TeamOf(bidderSeat)
	var trickUs, tricksThem, bidTeamTricks, defTeamTricks int
	for s := 0; s < 6; s++ {
		t := state.TricksTaken[s]
		if game.TeamOf(s) == myTeam {
			trickUs += t
		} else {
			tricksThem += t
		}
		if game.TeamOf(s) == bidderTeam {
			bidTeamTricks += t
		} else {
			defTeamTricks += t
		}
	}
	f[i] = float32(trickUs) / game.TotalTricks
	i++ // tricks_taken_us
	f[i] = float32(tricksThem) / game.TotalTricks
	i++ // tricks_taken_them

	bidAmount := state.BidAmount
	if bidAmount > game.TotalTricks {
		bidAmount = game.TotalTricks // pepper → treat as 8
	}
	needed := bidAmount - bidTeamTricks
	if needed < 0 {
		needed = 0
	}
	f[i] = float32(needed) / game.TotalTricks
	i++ // tricks_needed_us

	toEuchre := (game.TotalTricks - bidAmount + 1) - defTeamTricks
	if toEuchre < 0 {
		toEuchre = 0
	}
	f[i] = float32(toEuchre) / game.TotalTricks
	i++ // tricks_to_euchre
	f[i] = float32(game.TotalTricks-trickNum) / game.TotalTricks
	i++ // tricks_remaining

	// --- Current trick context (10–15) ---
	f[i] = float32(cardsInTrick) / 5.0
	i++ // cards_in_trick

	if cardsInTrick > 0 {
		winnerSeat := trick.Winner()
		if winnerSeat == seat {
			f[i] = 1.0
		}
		i++ // am_winning_trick

		for _, p := range game.Partners(seat) {
			if winnerSeat == p {
				f[i] = 1.0
				break
			}
		}
		i++ // partner_winning

		for _, pc := range trick.Cards[:trick.NCards] {
			if pc.Seat == bidderSeat {
				f[i] = 1.0
				break
			}
		}
		i++ // bidder_has_played

		for _, pc := range trick.Cards[:trick.NCards] {
			if pc.Seat == winnerSeat {
				rank := card.TrumpRank(pc.Card, trump)
				if rank >= 0 {
					f[i] = float32(rank) / 13.0
				}
				break
			}
		}
		i++ // winning_trump_rank
	} else {
		i += 4 // am_winning_trick, partner_winning, bidder_has_played, winning_trump_rank all 0
		f[i] = 1.0
	}
	i++ // is_leader

	// --- My hand composition (16–23) ---
	var trumpCount, rightBowerCount, leftBowerCount int
	highestTrumpRank := card.TrumpRankNone
	var offAces, voids, singletons int
	suitCounts := [4]int{}

	for _, c := range hand {
		rank := card.TrumpRank(c, trump)
		if rank >= 0 {
			trumpCount++
			if card.IsRightBower(c, trump) {
				rightBowerCount++
			} else if card.IsLeftBower(c, trump) {
				leftBowerCount++
			}
			if rank > highestTrumpRank {
				highestTrumpRank = rank
			}
		} else {
			suitCounts[c.Suit]++
			if c.Rank == card.Ace {
				offAces++
			}
		}
	}

	partnerSuit := card.PartnerSuit(trump)
	for s := card.Suit(0); s < 4; s++ {
		if s == trump {
			continue
		}
		count := suitCounts[s]
		if s == partnerSuit {
			// Left bower already counted as trump, suitCounts[partnerSuit] is accurate.
		}
		switch count {
		case 0:
			voids++
		case 1:
			singletons++
		}
	}

	f[i] = float32(trumpCount) / card.TotalTrumpCards
	i++ // trump_in_hand
	if rightBowerCount > 0 {
		f[i] = 1.0
	}
	i++ // has_right
	if leftBowerCount > 0 {
		f[i] = 1.0
	}
	i++ // has_left
	if highestTrumpRank >= 0 {
		f[i] = float32(highestTrumpRank) / card.TrumpRankRight
	}
	i++ // highest_trump_rank
	f[i] = float32(offAces) / card.NonTrumpRankAce
	i++ // off_suit_aces
	f[i] = float32(voids) / 3.0
	i++ // void_suits
	f[i] = float32(singletons) / 3.0
	i++ // singleton_suits
	f[i] = float32(len(hand)) / game.TotalTricks
	i++ // cards_in_hand

	// --- Trump field knowledge (24–28) ---
	h := state.History
	trumpRemaining := h.TrumpRemaining(trump)
	f[i] = float32(trumpRemaining) / card.TotalTrumpCards
	i++ // trump_remaining
	f[i] = float32(h.TrumpPlayed(trump)) / card.TotalTrumpCards
	i++ // trump_played
	f[i] = float32(h.RightBowersPlayed(trump)) / 2.0
	i++ // right_bowers_played
	f[i] = float32(h.LeftBowersPlayed(trump)) / 2.0
	i++ // left_bowers_played

	if highestTrumpRank >= 0 {
		for _, c := range hand {
			if card.TrumpRank(c, trump) == highestTrumpRank {
				if h.IsTopTrump(c, trump) {
					f[i] = 1.0
				}
				break
			}
		}
	}
	i++ // is_top_trump_in_hand

	// --- Score context (29–32) ---
	scores := state.Scores
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

	// --- Bid context (33) ---
	f[i] = float32(bidAmount) / game.TotalTricks
	i++ // bid_amount_norm

	// --- Trump dominance (34–36) ---
	if trumpRemaining > 0 {
		f[i] = float32(trumpCount) / float32(trumpRemaining)
	}
	i++ // my_trump_fraction

	oppTrump := trumpRemaining - trumpCount
	if oppTrump < 0 {
		oppTrump = 0
	}
	f[i] = float32(oppTrump) / card.TotalTrumpCards
	i++ // opp_trump_est

	if cardsInTrick > 0 && remaining == 0 {
		f[i] = 1.0
	}
	i++ // is_last_to_play

	// --- Trick winner knowledge (37–39) ---
	if cardsInTrick > 0 {
		winnerCard := trick.WinnerCard()
		winnerTR := card.TrumpRank(winnerCard, trump)
		partnerWinning := false
		for _, p := range game.Partners(seat) {
			if trick.Winner() == p {
				partnerWinning = true
				break
			}
		}

		// winner_is_unbeatable: no unplayed card (outside the trick) beats the current winner.
		unbeatable := false
		if winnerTR >= 0 {
			unbeatable = h.IsTopTrump(winnerCard, trump)
		} else {
			// Non-trump winner: check if any higher non-trump card of the led suit is still out.
			ledSuit := trick.LedSuit()
			winnerNR := card.NonTrumpRank(winnerCard)
			unbeatable = true
		outerWin:
			for _, r := range []card.Rank{card.Nine, card.Ten, card.Jack, card.Queen, card.King, card.Ace} {
				cand := card.Card{Suit: ledSuit, Rank: r}
				if card.NonTrumpRank(cand) > winnerNR && card.TrumpRank(cand, trump) < 0 {
					for ci := 0; ci < 2; ci++ {
						if !h.IsSeen(card.Card{Suit: ledSuit, Rank: r, CopyIndex: ci}) {
							unbeatable = false
							break outerWin
						}
					}
				}
			}
			// Also check if any trump remains — trump always beats a non-trump winner.
			if h.TrumpRemaining(trump) > 0 {
				unbeatable = false
			}
		}
		if unbeatable {
			f[i] = 1.0
		}
		i++ // winner_is_unbeatable

		// i_can_beat_winner: I hold at least one card that beats the current winner.
		iCanBeat := false
		for _, hc := range hand {
			hcTR := card.TrumpRank(hc, trump)
			canBeat := false
			if winnerTR >= 0 {
				canBeat = hcTR > winnerTR
			} else {
				if hcTR >= 0 {
					canBeat = true
				} else {
					ledSuit := trick.LedSuit()
					if card.EffectiveSuit(hc, trump) == ledSuit {
						canBeat = card.NonTrumpRank(hc) > card.NonTrumpRank(winnerCard)
					}
				}
			}
			if canBeat {
				iCanBeat = true
				break
			}
		}
		if iCanBeat {
			f[i] = 1.0
		}
		i++ // i_can_beat_winner

		if partnerWinning && unbeatable {
			f[i] = 1.0
		}
		i++ // partner_winning_unbeatable
	} else {
		i += 3 // winner_is_unbeatable, i_can_beat_winner, partner_winning_unbeatable all 0 when leading
	}

	// --- Off-suit field knowledge (40–44) ---
	// Enumerate the 3 non-trump suits in suit-index order (0-3, skip trump).
	var offSuits [3]card.Suit
	offIdx := 0
	for s := card.Suit(0); s < 4; s++ {
		if s != trump {
			offSuits[offIdx] = s
			offIdx++
		}
	}
	for oi := 0; oi < 3; oi++ {
		total := offSuitCardCount(offSuits[oi], trump)
		played := h.CardsPlayedInSuit(offSuits[oi], trump)
		rem := total - played
		if rem < 0 {
			rem = 0
		}
		f[i] = float32(rem) / 12.0
		i++ // offsuit1/2/3_remaining
	}

	if cardsInTrick > 0 {
		ledCard := trick.Cards[0].Card
		ledSuit := card.EffectiveSuit(ledCard, trump)
		ledTR := card.TrumpRank(ledCard, trump)
		if ledTR >= 0 {
			f[i] = float32(ledTR) / card.TrumpRankRight
		} else {
			f[i] = float32(card.NonTrumpRank(ledCard)) / card.NonTrumpRankAce
		}
		i++ // led_card_rank_norm

		total := offSuitCardCount(ledSuit, trump)
		played := h.CardsPlayedInSuit(ledSuit, trump)
		rem := total - played
		if rem < 0 {
			rem = 0
		}
		f[i] = float32(rem) / 12.0
		i++ // led_suit_remaining
	} else {
		i += 2 // led_card_rank_norm, led_suit_remaining both 0 when leading
	}

	// --- Void knowledge (45–50) ---
	partners := game.Partners(seat)
	for oi := 0; oi < 3; oi++ {
		s := offSuits[oi]
		voidCount := 0
		for _, p := range partners {
			if h.IsVoidInSuit(p, s) {
				voidCount++
			}
		}
		f[i] = float32(voidCount) / float32(len(partners))
		i++ // partner_void_1/2/3
	}
	for oi := 0; oi < 3; oi++ {
		s := offSuits[oi]
		voidCount := 0
		for opp := 0; opp < 6; opp++ {
			if opp != seat && game.TeamOf(opp) != myTeam && h.IsVoidInSuit(opp, s) {
				voidCount++
			}
		}
		f[i] = float32(voidCount) / 4.0
		i++ // opp_void_count_1/2/3
	}

	// --- Trick dynamics (51–55) ---
	partnerPlayed := false
	for _, p := range game.Partners(seat) {
		for _, pc := range trick.Cards[:trick.NCards] {
			if pc.Seat == p {
				partnerPlayed = true
				break
			}
		}
	}
	if partnerPlayed {
		f[i] = 1.0
	}
	i++ // partner_has_played

	if cardsInTrick > 0 {
		ledSuit := card.EffectiveSuit(trick.Cards[0].Card, trump)
		if ledSuit == trump {
			f[i] = 1.0
		}
		i++ // suit_led_is_trump

		trumpInTrick := 0
		for _, pc := range trick.Cards[:trick.NCards] {
			if card.TrumpRank(pc.Card, trump) >= 0 {
				trumpInTrick++
			}
		}
		f[i] = float32(trumpInTrick) / 6.0
		i++ // trump_in_current_trick

		// top_of_led_suit_in_hand: do I hold the highest remaining card of the led suit?
		ledEffSuit := card.EffectiveSuit(trick.Cards[0].Card, trump)
		topInLedSuit := false
		if card.TrumpRank(trick.Cards[0].Card, trump) >= 0 {
			// Led suit is trump — check if I hold the top trump
			for _, hc := range hand {
				if card.TrumpRank(hc, trump) >= 0 && h.IsTopTrump(hc, trump) {
					topInLedSuit = true
					break
				}
			}
		} else {
			// Off-suit led — find highest remaining card of that suit in hand
			myBest := -1
			for _, hc := range hand {
				if card.EffectiveSuit(hc, trump) == ledEffSuit {
					r := card.NonTrumpRank(hc)
					if r > myBest {
						myBest = r
					}
				}
			}
			if myBest >= 0 {
				// Check if any higher card of the led suit is still unseen
				isTop := true
			topCheck:
				for _, r := range []card.Rank{card.Nine, card.Ten, card.Jack, card.Queen, card.King, card.Ace} {
					higher := card.Card{Suit: ledEffSuit, Rank: r}
					if card.NonTrumpRank(higher) > myBest && card.TrumpRank(higher, trump) < 0 {
						for ci := 0; ci < 2; ci++ {
							if !h.IsSeen(card.Card{Suit: ledEffSuit, Rank: r, CopyIndex: ci}) {
								isTop = false
								break topCheck
							}
						}
					}
				}
				topInLedSuit = isTop
			}
		}
		if topInLedSuit {
			f[i] = 1.0
		}
		i++ // top_of_led_suit_in_hand

		f[i] = float32((trick.Winner()-seat+6)%6) / 5.0
		i++ // winner_seat_rel_norm
	} else {
		i += 4 // suit_led_is_trump, trump_in_current_trick, top_of_led_suit_in_hand, winner_seat_rel_norm all 0 when leading
	}

	// --- Per-seat void flags (56–70) ---
	for relPos := 1; relPos <= 5; relPos++ {
		otherSeat := (seat + relPos) % 6
		for oi := 0; oi < 3; oi++ {
			if h.IsVoidInSuit(otherSeat, offSuits[oi]) {
				f[i] = 1.0
			}
			i++
		}
	}

	// --- Trick leader identity (71–72) ---
	if cardsInTrick > 0 {
		leaderSeat := trick.Cards[0].Seat
		if leaderSeat == bidderSeat {
			f[i] = 1.0
		}
		i++ // led_by_bidder
		for _, p := range game.Partners(seat) {
			if leaderSeat == p {
				f[i] = 1.0
				break
			}
		}
		i++ // led_by_partner
	} else {
		i += 2 // led_by_bidder, led_by_partner both 0 when leading
	}

	// --- Recent trump density (73) ---
	{
		recentTricks := state.TrickNumber
		if recentTricks > 2 {
			recentTricks = 2
		}
		if recentTricks > 0 {
			recentCards := recentTricks * activePlayers
			played := h.PlayedSlice()
			if len(played) >= recentCards {
				start := len(played) - recentCards
				trumpInRecent := 0
				for _, c := range played[start:] {
					if card.TrumpRank(c, trump) >= 0 {
						trumpInRecent++
					}
				}
				f[i] = float32(trumpInRecent) / float32(recentCards)
			}
		}
		i++ // trump_rate_recent
	}

	_ = i // i == ContextFeatureLen (74) here
	return f
}

// offSuitCardCount returns the total cards of the given effective suit in a pinochle deck.
// Partner suit loses its J to trump, so has 10 cards; other off-suits have 12.
func offSuitCardCount(suit card.Suit, trump card.Suit) int {
	if suit == card.PartnerSuit(trump) {
		return 10
	}
	return 12
}

// AppendCard appends the 6 per-card features to a context vector, returning the full row.
// hand is the player's full hand at the time of the decision (for card_suit_count).
// trick is the current trick in progress (for can_beat_winner and is_overtrump).
func AppendCard(ctx [ContextFeatureLen]float32, c card.Card, trump card.Suit, hand []card.Card, trick *game.Trick, history *game.HandHistory) [TotalFeatureLen]float32 {
	var f [TotalFeatureLen]float32
	copy(f[:ContextFeatureLen], ctx[:])

	i := ContextFeatureLen
	isTrump := card.TrumpRank(c, trump) >= 0

	if isTrump {
		f[i] = 1.0
	}
	i++ // card_is_trump

	if isTrump {
		f[i] = float32(card.TrumpRank(c, trump)) / card.TrumpRankRight
	} else {
		f[i] = float32(card.NonTrumpRank(c)) / card.NonTrumpRankAce
	}
	i++ // card_rank_norm

	if isTrump && history.IsTopTrump(c, trump) {
		f[i] = 1.0
	}
	i++ // card_is_top

	// can_beat_winner and is_overtrump
	if trick == nil || trick.NCards == 0 {
		// Leading: always "wins" by default (sets the bar).
		f[i] = 1.0
		i++ // can_beat_winner
		i++ // is_overtrump (0 — not relevant when leading)
	} else {
		winnerSeat := trick.Winner()
		var winnerCard card.Card
		for _, pc := range trick.Cards[:trick.NCards] {
			if pc.Seat == winnerSeat {
				winnerCard = pc.Card
				break
			}
		}
		winnerTrumpRank := card.TrumpRank(winnerCard, trump)
		cTrumpRank := card.TrumpRank(c, trump)

		canBeat := false
		if winnerTrumpRank >= 0 {
			// Current winner is trump: only a higher trump beats it.
			canBeat = cTrumpRank > winnerTrumpRank
		} else {
			// Current winner is not trump.
			if cTrumpRank >= 0 {
				// Any trump beats a non-trump winner.
				canBeat = true
			} else {
				// Both non-trump: must be same led suit and higher rank.
				ledSuit := trick.LedSuit()
				if card.EffectiveSuit(c, trump) == ledSuit {
					canBeat = card.NonTrumpRank(c) > card.NonTrumpRank(winnerCard)
				}
			}
		}
		if canBeat {
			f[i] = 1.0
		}
		i++ // can_beat_winner

		// Overtrump: spending trump to beat trump.
		if cTrumpRank >= 0 && winnerTrumpRank >= 0 {
			f[i] = 1.0
		}
		i++ // is_overtrump
	}

	// card_suit_count: how many cards of this card's effective suit are in hand.
	effSuit := card.EffectiveSuit(c, trump)
	suitCount := 0
	for _, hc := range hand {
		if card.EffectiveSuit(hc, trump) == effSuit {
			suitCount++
		}
	}
	f[i] = float32(suitCount) / 8.0
	i++ // card_suit_count

	// card_is_top_in_suit: for non-trump, 1 if every higher-ranked card of this suit has been played.
	if !isTrump && history != nil {
		myRank := card.NonTrumpRank(c)
		isTop := true
	outer:
		for _, r := range []card.Rank{card.Nine, card.Ten, card.Jack, card.Queen, card.King, card.Ace} {
			higher := card.Card{Suit: c.Suit, Rank: r}
			if card.NonTrumpRank(higher) > myRank && card.TrumpRank(higher, trump) < 0 {
				for ci := 0; ci < 2; ci++ {
					if !history.IsSeen(card.Card{Suit: c.Suit, Rank: r, CopyIndex: ci}) {
						isTop = false
						break outer
					}
				}
			}
		}
		if isTop {
			f[i] = 1.0
		}
	}
	i++ // card_is_top_in_suit

	// card_suit_remaining: remaining cards of this card's effective suit globally / 12.
	if history != nil {
		total := offSuitCardCount(effSuit, trump)
		if isTrump {
			total = card.TotalTrumpCards
		}
		played := history.CardsPlayedInSuit(effSuit, trump)
		rem := total - played
		if rem < 0 {
			rem = 0
		}
		f[i] = float32(rem) / 12.0
	}
	i++ // card_suit_remaining

	_ = i
	return f
}
