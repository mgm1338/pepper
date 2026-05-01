package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/strategy"
	"github.com/max/pepper/ml"
)

// bidStateFromJSON converts a BidStateJSON + seat + hand into a game.BidState.
func bidStateFromJSON(seat int, hand []card.Card, s BidStateJSON) game.BidState {
	return game.BidState{
		Hand:            hand,
		CurrentHigh:     s.CurrentHigh,
		DealerSeat:      s.DealerSeat,
		HighSeat:        s.HighSeat,
		SeatsLeft:       s.SeatsLeft,
		PassesSoFar:     s.PassesSoFar,
		PartnerHasBid:   s.PartnerHasBid,
		PartnerBidLevel: s.PartnerBidLevel,
		Scores:          s.Scores,
		Seat:            seat,
		AllBids:         s.SeatBidLevels,
	}
}

// --- Shared response types ---

type CardScore struct {
	Card  string  `json:"card"`
	Score float32 `json:"score"`
	Rank  int     `json:"rank"`
}

type BidScore struct {
	Bid   int     `json:"bid"`
	Label string  `json:"label"`
	Score float32 `json:"score"`
	Rank  int     `json:"rank"`
}

type AttributionEntry struct {
	Feature string  `json:"feature"`
	Value   float32 `json:"value"`   // raw normalized value the model saw
	Impact  float32 `json:"impact"`  // gradient×input in score_delta units
}

// --- /evaluate/play ---

type EvaluatePlayRequest struct {
	Seat       int            `json:"seat"`
	ValidPlays []string       `json:"valid_plays"`
	State      TrickStateJSON `json:"state"`
	Chosen     string         `json:"chosen"` // card the player picked
}

type EvaluatePlayResponse struct {
	// MLP did not load — cannot evaluate.
	Unavailable bool `json:"unavailable,omitempty"`

	// Optimal card according to MLP.
	Optimal CardScore `json:"optimal"`

	// The card the player chose.
	Chosen  CardScore `json:"chosen"`
	EVLost  float32   `json:"ev_lost"` // optimal.Score - chosen.Score (≥0)

	// All valid cards ranked by MLP score.
	AllCards []CardScore `json:"all_cards"`

	// Gradient×input attribution for the chosen card.
	// Shows why the model rated it the way it did.
	ChosenAttribution []AttributionEntry `json:"chosen_attribution"`

	// Attribution for the optimal card (omitted when chosen == optimal).
	OptimalAttribution []AttributionEntry `json:"optimal_attribution,omitempty"`

	// Balanced's recommended card for comparison.
	BalancedCard string `json:"balanced_card"`
}

func handleEvaluatePlay(w http.ResponseWriter, r *http.Request) {
	var req EvaluatePlayRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	trump := parseSuit(req.State.Trump)
	trick := game.NewTrick(req.State.Leader, trump)
	for _, pc := range req.State.Trick {
		trick.Add(parseCard(pc.Card), pc.Seat)
	}
	var history game.HandHistory
	for _, trickCodes := range req.State.History {
		history.Record(parseCards(trickCodes))
	}
	hand := parseCards(req.State.Hand)
	validPlays := parseCards(req.ValidPlays)
	chosenCard := parseCard(req.Chosen)

	state := game.TrickState{
		Trick:       trick,
		Trump:       trump,
		Seat:        req.Seat,
		BidderSeat:  req.State.BidderSeat,
		BidAmount:   req.State.BidAmount,
		TrickNumber: req.State.TrickNumber,
		TricksTaken: req.State.TricksTaken,
		Scores:      req.State.Scores,
		Hand:        hand,
		History:     &history,
	}

	// Balanced recommendation (always available).
	balancedBot := strategy.NewStandard(strategy.Balanced)
	balancedChosen := balancedBot.Play(req.Seat, validPlays, &state)

	if cardModel == nil {
		writeJSON(w, EvaluatePlayResponse{
			Unavailable:  true,
			BalancedCard: cardCode(balancedChosen),
		})
		return
	}

	isBidder := game.TeamOf(req.Seat) == game.TeamOf(req.State.BidderSeat)
	sign := float32(1)
	if !isBidder {
		sign = -1
	}

	// Score all valid cards.
	ctx := ml.ExtractContext(req.Seat, hand, state, 6)
	n := len(validPlays)
	featFlat := cardModel.BatchFeatBuf
	feats := make([][ml.TotalFeatureLen]float32, n)
	for i, c := range validPlays {
		f := ml.AppendCard(ctx, c, trump, hand, state.Trick, state.History)
		feats[i] = f
		copy(featFlat[i*ml.TotalFeatureLen:], f[:])
	}
	rawScores := make([]float32, n)
	cardModel.ScoreBatch(n, featFlat[:n*ml.TotalFeatureLen], rawScores)

	// Build ranked list (from this player's perspective).
	type indexed struct {
		idx   int
		score float32
	}
	ranked := make([]indexed, n)
	for i, s := range rawScores {
		ranked[i] = indexed{i, s * sign}
	}
	sort.Slice(ranked, func(a, b int) bool { return ranked[a].score > ranked[b].score })

	allCards := make([]CardScore, n)
	for rank, r := range ranked {
		allCards[rank] = CardScore{
			Card:  cardCode(validPlays[r.idx]),
			Score: r.score,
			Rank:  rank + 1,
		}
	}

	// Find chosen and optimal.
	optimalEntry := allCards[0]
	var chosenEntry CardScore
	chosenIdx := -1
	for i, c := range validPlays {
		if c.Equal(chosenCard) {
			chosenIdx = i
			break
		}
	}
	if chosenIdx < 0 {
		// Chosen card not in valid plays — treat as rank last.
		chosenEntry = CardScore{Card: req.Chosen, Score: 0, Rank: n}
	} else {
		chosenScore := rawScores[chosenIdx] * sign
		for _, cs := range allCards {
			if cs.Card == cardCode(chosenCard) {
				chosenEntry = cs
				break
			}
		}
		_ = chosenScore
	}

	evLost := optimalEntry.Score - chosenEntry.Score
	if evLost < 0 {
		evLost = 0
	}

	// Attribution for chosen card.
	var chosenAttr []AttributionEntry
	if chosenIdx >= 0 {
		raw := cardModel.AttributePlay(feats[chosenIdx], isBidder)
		chosenAttr = sortedAttribution(raw)
	}

	// Attribution for optimal card (only when different from chosen).
	var optimalAttr []AttributionEntry
	optimalIdx := ranked[0].idx
	if optimalIdx != chosenIdx {
		raw := cardModel.AttributePlay(feats[optimalIdx], isBidder)
		optimalAttr = sortedAttribution(raw)
	}

	writeJSON(w, EvaluatePlayResponse{
		Optimal:            optimalEntry,
		Chosen:             chosenEntry,
		EVLost:             evLost,
		AllCards:           allCards,
		ChosenAttribution:  chosenAttr,
		OptimalAttribution: optimalAttr,
		BalancedCard:       cardCode(balancedChosen),
	})
}

// --- /evaluate/bid ---

type EvaluateBidRequest struct {
	Seat   int          `json:"seat"`
	Hand   []string     `json:"hand"`
	State  BidStateJSON `json:"bid_state"`
	Chosen int          `json:"chosen"` // 0=pass, 4-7=bid, 8=pepper
}

type EvaluateBidResponse struct {
	Unavailable bool `json:"unavailable,omitempty"`

	Optimal BidScore `json:"optimal"`
	Chosen  BidScore `json:"chosen"`
	EVLost  float32  `json:"ev_lost"`

	AllBids []BidScore `json:"all_bids"`

	ChosenAttribution  []AttributionEntry `json:"chosen_attribution"`
	OptimalAttribution []AttributionEntry `json:"optimal_attribution,omitempty"`

	// Balanced's view of the hand.
	Balanced struct {
		BestSuit      string  `json:"best_suit"`
		HandTricks    float64 `json:"hand_tricks"`
		PartnerTricks float64 `json:"partner_tricks"`
		TotalEstimate float64 `json:"total_estimate"`
		Bid           int     `json:"bid"`
	} `json:"balanced"`
}

func handleEvaluateBid(w http.ResponseWriter, r *http.Request) {
	var req EvaluateBidRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	hand := parseCards(req.Hand)
	state := bidStateFromJSON(req.Seat, hand, req.State)

	// Balanced analysis (always available).
	balancedStrat := strategy.NewStandard(strategy.Balanced)
	analysis := balancedStrat.AnalyzeBid(req.Seat, &state)

	var resp EvaluateBidResponse
	resp.Balanced.BestSuit = suitCode(analysis.BestSuit)
	resp.Balanced.HandTricks = analysis.HandTricks
	resp.Balanced.PartnerTricks = analysis.PartnerTricks
	resp.Balanced.TotalEstimate = analysis.TotalEstimate
	resp.Balanced.Bid = analysis.Bid

	if bidModel == nil {
		resp.Unavailable = true
		writeJSON(w, resp)
		return
	}

	// Score all valid bids.
	ctx := ml.BidContext(
		req.Seat, hand,
		req.State.DealerSeat, req.State.CurrentHigh, req.State.HighSeat,
		req.State.SeatsLeft, req.State.Scores,
		req.State.PassesSoFar, req.State.PartnerHasBid, req.State.PartnerBidLevel,
		req.State.SeatBidLevels,
	)

	validBids := ml.ValidBidLevels(req.State.CurrentHigh)
	n := len(validBids)
	featFlat := bidModel.BatchFeatBuf
	feats := make([][ml.BidTotalLen]float32, n)
	for i, bidLevel := range validBids {
		f := ml.AppendBidAction(ctx, bidLevel, req.State.CurrentHigh)
		feats[i] = f
		copy(featFlat[i*ml.BidTotalLen:], f[:])
	}
	rawScores := make([]float32, n)
	bidModel.ScoreBatch(n, featFlat[:n*ml.BidTotalLen], rawScores)

	type indexed struct {
		idx   int
		score float32
	}
	ranked := make([]indexed, n)
	for i, s := range rawScores {
		ranked[i] = indexed{i, s}
	}
	sort.Slice(ranked, func(a, b int) bool { return ranked[a].score > ranked[b].score })

	allBids := make([]BidScore, n)
	for rank, r := range ranked {
		allBids[rank] = BidScore{
			Bid:   validBids[r.idx],
			Label: bidLabel(validBids[r.idx]),
			Score: r.score,
			Rank:  rank + 1,
		}
	}

	optimalEntry := allBids[0]

	var chosenEntry BidScore
	chosenIdx := -1
	for i, b := range validBids {
		if b == req.Chosen {
			chosenIdx = i
			break
		}
	}
	if chosenIdx < 0 {
		chosenEntry = BidScore{Bid: req.Chosen, Label: bidLabel(req.Chosen), Score: 0, Rank: n}
	} else {
		for _, bs := range allBids {
			if bs.Bid == req.Chosen {
				chosenEntry = bs
				break
			}
		}
	}

	evLost := optimalEntry.Score - chosenEntry.Score
	if evLost < 0 {
		evLost = 0
	}

	var chosenAttr []AttributionEntry
	if chosenIdx >= 0 {
		raw := bidModel.AttributeBid(feats[chosenIdx])
		chosenAttr = sortedAttribution(raw)
	}

	var optimalAttr []AttributionEntry
	optimalIdx := ranked[0].idx
	if optimalIdx != chosenIdx {
		raw := bidModel.AttributeBid(feats[optimalIdx])
		optimalAttr = sortedAttribution(raw)
	}

	resp.Optimal = optimalEntry
	resp.Chosen = chosenEntry
	resp.EVLost = evLost
	resp.AllBids = allBids
	resp.ChosenAttribution = chosenAttr
	resp.OptimalAttribution = optimalAttr

	writeJSON(w, resp)
}

// sortedAttribution converts raw []FeatureAttribution to []AttributionEntry
// sorted by absolute impact descending.
func sortedAttribution(raw []ml.FeatureAttribution) []AttributionEntry {
	out := make([]AttributionEntry, len(raw))
	for i, fa := range raw {
		out[i] = AttributionEntry{
			Feature: fa.Feature,
			Value:   fa.Value,
			Impact:  fa.Impact,
		}
	}
	sort.Slice(out, func(a, b int) bool {
		ia, ib := out[a].Impact, out[b].Impact
		if ia < 0 {
			ia = -ia
		}
		if ib < 0 {
			ib = -ib
		}
		return ia > ib
	})
	return out
}

func bidLabel(bid int) string {
	switch bid {
	case game.PassBid:
		return "pass"
	case game.PepperBid:
		return "pepper"
	default:
		return fmt.Sprintf("%d", bid)
	}
}
