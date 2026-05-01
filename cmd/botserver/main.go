// Package main runs the bot HTTP microservice.
// pepper2 calls this to get bot decisions for bids, trump selection, and card play.
//
// POST /bid        { "hand": [...], "bid_state": {...} }          → { "bid": 5 }
// POST /trump      { "seat": 0, "hand": [...] }                   → { "suit": "H" }
// POST /play       { "seat": 0, "valid_plays": [...], "state": {...} } → { "card": "JH" }
//
// All action endpoints accept optional bot config fields:
//   "play_style": 0.0–1.0  (0 = pure Balanced, 1 = pure MLP card play)
//   "bid_style":  0.0–1.0  (0 = pure Balanced, 1 = pure MLP bidding)
//   "slop":       0.0–1.0  (0 = no mistakes, 1 = maximum randomness)
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/mlstrategy"
	"github.com/max/pepper/internal/strategy"
	"github.com/max/pepper/ml"
)

var mlpStrat  game.Strategy
var cardModel *ml.MLP
var bidModel  *ml.BidMLP

func botForConfig(cfg BotConfigJSON) game.Strategy {
	return strategy.NewBot(strategy.BotConfig{
		PlayStyle:   cfg.PlayStyle,
		BidStyle:    cfg.BidStyle,
		Slop:        cfg.Slop,
		TrumpMemory: cfg.TrumpMemory,
		AceMemory:   cfg.AceMemory,
	}, mlpStrat)
}

func main() {
	addr         := flag.String("addr", ":9090", "listen address")
	modelPath    := flag.String("model", "", "path to model_weights.json (enables MLP play)")
	bidModelPath := flag.String("bid-model", "", "path to bid_model_weights.json (enables MLP bidding)")
	flag.Parse()

	if *modelPath != "" {
		model, err := ml.LoadMLP(*modelPath)
		if err != nil {
			log.Fatalf("load MLP: %v", err)
		}
		cardModel = model
		ms := mlstrategy.NewMLPStrategy(model, strategy.Balanced)
		if *bidModelPath != "" {
			bm, err := ml.LoadBidMLP(*bidModelPath)
			if err != nil {
				log.Fatalf("load bid MLP: %v", err)
			}
			bidModel = bm
			ms.WithBidModel(bm)
			log.Printf("bid MLP loaded from %s", *bidModelPath)
		}
		mlpStrat = ms
		log.Printf("MLP strategy loaded from %s", *modelPath)
	} else {
		log.Printf("no --model provided; play_style and bid_style will fall back to Balanced")
	}

	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/bid", handleBid)
	http.HandleFunc("/trump", handleTrump)
	http.HandleFunc("/play", handlePlay)
	http.HandleFunc("/advice", handleAdvice)
	http.HandleFunc("/evaluate/play", handleEvaluatePlay)
	http.HandleFunc("/evaluate/bid", handleEvaluateBid)
	http.HandleFunc("/game/new", handleGameNew)
	http.HandleFunc("/game/state", handleGameState)
	http.HandleFunc("/game/bid", handleGameBid)
	http.HandleFunc("/game/trump", handleGameTrump)
	http.HandleFunc("/game/discard", handleGameDiscard)
	http.HandleFunc("/game/play", handleGamePlay)
	http.HandleFunc("/game/next-hand", handleGameNextHand)

	log.Printf("bot server listening on %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, nil))
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, "ok")
}

// --- Bid ---

type BotConfigJSON struct {
	PlayStyle   float64                   `json:"play_style"`
	BidStyle    float64                   `json:"bid_style"`
	Slop        float64                   `json:"slop"`
	TrumpMemory strategy.TrumpMemoryLevel `json:"trump_memory"`
	AceMemory   strategy.AceMemoryLevel   `json:"ace_memory"`
}

type BidRequest struct {
	Seat  int          `json:"seat"`
	Hand  []string     `json:"hand"`
	State BidStateJSON `json:"bid_state"`
	BotConfigJSON
}

type BidStateJSON struct {
	CurrentHigh     int    `json:"current_high"`
	DealerSeat      int    `json:"dealer_seat"`
	HighSeat        int    `json:"high_seat"`        // -1 if no bids yet
	SeatsLeft       int    `json:"seats_left"`
	PassesSoFar     int    `json:"passes_so_far"`
	PartnerHasBid   bool   `json:"partner_has_bid"`
	PartnerBidLevel int    `json:"partner_bid_level"`
	Scores          [2]int `json:"scores"`
	SeatBidLevels   [6]int `json:"seat_bid_levels"`  // bid placed by each seat (0 = pass/not yet bid)
}

type BidResponse struct {
	Bid int `json:"bid"`
}

func handleBid(w http.ResponseWriter, r *http.Request) {
	var req BidRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	hand := parseCards(req.Hand)
	state := bidStateFromJSON(req.Seat, hand, req.State)
	bot := botForConfig(req.BotConfigJSON)
	bid := bot.Bid(req.Seat, &state)
	log.Printf("BID seat=%d play=%.2f bid=%.2f slop=%.2f hand=%v currentHigh=%d → %d",
		req.Seat, req.PlayStyle, req.BidStyle, req.Slop, req.Hand, req.State.CurrentHigh, bid)
	writeJSON(w, BidResponse{Bid: bid})
}

// --- Trump ---

type TrumpRequest struct {
	Seat int      `json:"seat"`
	Hand []string `json:"hand"`
	BotConfigJSON
}

type TrumpResponse struct {
	Suit string `json:"suit"`
}

func handleTrump(w http.ResponseWriter, r *http.Request) {
	var req TrumpRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	hand := parseCards(req.Hand)
	bot := botForConfig(req.BotConfigJSON)
	suit := bot.ChooseTrump(req.Seat, hand)
	writeJSON(w, TrumpResponse{Suit: suitCode(suit)})
}

// --- Play ---

type PlayRequest struct {
	Seat       int            `json:"seat"`
	ValidPlays []string       `json:"valid_plays"`
	State      TrickStateJSON `json:"state"`
	BotConfigJSON
}

type TrickStateJSON struct {
	Trump       string           `json:"trump"`
	BidderSeat  int              `json:"bidder_seat"`
	BidAmount   int              `json:"bid_amount"`
	TrickNumber int              `json:"trick_number"`
	TricksTaken [6]int           `json:"tricks_taken"`
	Scores      [2]int           `json:"scores"`
	Hand        []string         `json:"hand"`
	Trick       []PlayedCardJSON `json:"trick"`
	Leader      int              `json:"leader"`
	History     [][]string       `json:"history"`
}

type PlayedCardJSON struct {
	Card string `json:"card"`
	Seat int    `json:"seat"`
}

type PlayResponse struct {
	Card string `json:"card"`
}

func handlePlay(w http.ResponseWriter, r *http.Request) {
	var req PlayRequest
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

	bot := botForConfig(req.BotConfigJSON)
	chosen := bot.Play(req.Seat, validPlays, &state)
	log.Printf("PLAY seat=%d play=%.2f bid=%.2f slop=%.2f trump=%s → %s",
		req.Seat, req.PlayStyle, req.BidStyle, req.Slop, req.State.Trump, cardCode(chosen))
	writeJSON(w, PlayResponse{Card: cardCode(chosen)})
}

// --- Advice (Ask Grandpa) ---

type AdviceRequest struct {
	Type       string          `json:"type"` // "bid" or "play"
	Seat       int             `json:"seat"`
	Hand       []string        `json:"hand"`
	BidState   *BidStateJSON   `json:"bid_state,omitempty"`
	ValidPlays []string        `json:"valid_plays,omitempty"`
	State      *TrickStateJSON `json:"state,omitempty"`
}

type AdviceResponse struct {
	Suggestion  string `json:"suggestion"`
	Explanation string `json:"explanation"`
}

func handleAdvice(w http.ResponseWriter, r *http.Request) {
	var req AdviceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Advice always uses the strongest config: full MLP, no slop.
	expert := botForConfig(BotConfigJSON{PlayStyle: 1.0, BidStyle: 1.0, TrumpMemory: strategy.TrumpMemoryFull, AceMemory: strategy.AceMemoryAces})

	switch req.Type {
	case "bid":
		if req.BidState == nil {
			http.Error(w, "bid_state required for bid advice", http.StatusBadRequest)
			return
		}
		hand := parseCards(req.Hand)
		state := game.BidState{
			Hand:        hand,
			CurrentHigh: req.BidState.CurrentHigh,
			DealerSeat:  req.BidState.DealerSeat,
			Scores:      req.BidState.Scores,
		}
		bid := expert.Bid(req.Seat, &state)
		suggestion := fmt.Sprintf("%d", bid)
		explanation := "Pass"
		if bid == game.PepperBid {
			suggestion = "pepper"
			explanation = "Grandpa says go for pepper!"
		} else if bid > 0 {
			explanation = fmt.Sprintf("Grandpa says bid %d", bid)
		} else {
			explanation = "Grandpa says pass on this one"
		}
		log.Printf("ADVICE bid seat=%d → %s", req.Seat, suggestion)
		writeJSON(w, AdviceResponse{Suggestion: suggestion, Explanation: explanation})

	case "play":
		if req.State == nil || len(req.ValidPlays) == 0 {
			http.Error(w, "state and valid_plays required for play advice", http.StatusBadRequest)
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
		chosen := expert.Play(req.Seat, validPlays, &state)
		code := cardCode(chosen)
		log.Printf("ADVICE play seat=%d → %s", req.Seat, code)
		writeJSON(w, AdviceResponse{
			Suggestion:  code,
			Explanation: fmt.Sprintf("Grandpa says play the %s", code),
		})

	default:
		http.Error(w, "type must be 'bid' or 'play'", http.StatusBadRequest)
	}
}

// --- Helpers ---

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func cardCode(c card.Card) string {
	ranks := map[card.Rank]string{
		card.Nine: "9", card.Ten: "T", card.Jack: "J",
		card.Queen: "Q", card.King: "K", card.Ace: "A",
	}
	suits := map[card.Suit]string{
		card.Hearts: "H", card.Diamonds: "D",
		card.Clubs: "C", card.Spades: "S",
	}
	return ranks[c.Rank] + suits[c.Suit]
}

func suitCode(s card.Suit) string {
	switch s {
	case card.Hearts:
		return "H"
	case card.Diamonds:
		return "D"
	case card.Clubs:
		return "C"
	default:
		return "S"
	}
}

func parseCard(code string) card.Card {
	if len(code) < 2 {
		return card.Card{}
	}
	rankMap := map[byte]card.Rank{
		'9': card.Nine, 'T': card.Ten, 'J': card.Jack,
		'Q': card.Queen, 'K': card.King, 'A': card.Ace,
	}
	suitMap := map[byte]card.Suit{
		'H': card.Hearts, 'D': card.Diamonds,
		'C': card.Clubs, 'S': card.Spades,
	}
	return card.Card{Rank: rankMap[code[0]], Suit: suitMap[code[1]]}
}

func parseCards(codes []string) []card.Card {
	cards := make([]card.Card, 0, len(codes))
	for _, c := range codes {
		cards = append(cards, parseCard(c))
	}
	return cards
}

func parseSuit(s string) card.Suit {
	switch s {
	case "H":
		return card.Hearts
	case "D":
		return card.Diamonds
	case "C":
		return card.Clubs
	default:
		return card.Spades
	}
}
