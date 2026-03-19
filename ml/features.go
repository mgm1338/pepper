package ml

import (
	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

// Feature vector layout: ContextFeatureLen context features shared across all legal card
// choices at a decision point, followed by CardFeatureLen card-specific features appended
// once per candidate card. One training row = one (decision point, candidate card) pair.
const (
	ContextFeatureLen = 37
	CardFeatureLen    = 6
	TotalFeatureLen   = ContextFeatureLen + CardFeatureLen
)

// FeatureNames provides human-readable column names for the full feature vector.
// Indices 0–36 are context, 37–42 are per-card.
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

	// Per-card features (37–42)
	"card_is_trump",    // 1 if candidate card is trump
	"card_rank_norm",   // trump rank / 13 if trump, else off-suit rank / 6
	"card_is_top",      // 1 if candidate is the current top unplayed trump
	"can_beat_winner",  // 1 if this card beats the current trick leader
	"is_overtrump",     // 1 if card is trump and current winner is also trump
	"card_suit_count",  // cards of this suit in my hand / 8
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
	cardsInTrick := len(trick.Cards)

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

		for _, pc := range trick.Cards {
			if pc.Seat == bidderSeat {
				f[i] = 1.0
				break
			}
		}
		i++ // bidder_has_played

		for _, pc := range trick.Cards {
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

	_ = i // i == ContextFeatureLen here
	return f
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
	if trick == nil || len(trick.Cards) == 0 {
		// Leading: always "wins" by default (sets the bar).
		f[i] = 1.0
		i++ // can_beat_winner
		i++ // is_overtrump (0 — not relevant when leading)
	} else {
		winnerSeat := trick.Winner()
		var winnerCard card.Card
		for _, pc := range trick.Cards {
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

	_ = i
	return f
}
