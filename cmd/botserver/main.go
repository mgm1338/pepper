// Package main runs the bot HTTP microservice.
// pepper2 calls this to get bot decisions for bids, trump selection, and card play.
//
// POST /bid        { "hand": [...], "bid_state": {...} }          → { "bid": 5 }
// POST /trump      { "seat": 0, "hand": [...] }                   → { "suit": "H" }
// POST /play       { "seat": 0, "valid_plays": [...], "state": {...} } → { "card": "JH" }
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

var strat game.Strategy = strategy.NewStandard(strategy.Balanced)

func main() {
	addr          := flag.String("addr", ":9090", "listen address")
	modelPath     := flag.String("model", "", "path to model_weights.json (enables MLP play)")
	bidModelPath  := flag.String("bid-model", "", "path to bid_model_weights.json (enables MLP bidding)")
	flag.Parse()

	if *modelPath != "" {
		model, err := ml.LoadMLP(*modelPath)
		if err != nil {
			log.Fatalf("load MLP: %v", err)
		}
		mlpStrat := mlstrategy.NewMLPStrategy(model, strategy.Balanced)
		if *bidModelPath != "" {
			bidModel, err := ml.LoadBidMLP(*bidModelPath)
			if err != nil {
				log.Fatalf("load bid MLP: %v", err)
			}
			mlpStrat.WithBidModel(bidModel)
			log.Printf("bid MLP loaded from %s", *bidModelPath)
		}
		strat = mlpStrat
		log.Printf("MLP strategy loaded from %s", *modelPath)
	} else {
		strat = strategy.NewStandard(strategy.Balanced)
		log.Printf("using rule-based Balanced strategy (no --model provided)")
	}

	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/bid", handleBid)
	http.HandleFunc("/trump", handleTrump)
	http.HandleFunc("/play", handlePlay)

	log.Printf("bot server listening on %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, nil))
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, "ok")
}

// --- Bid ---

type BidRequest struct {
	Seat  int          `json:"seat"`
	Hand  []string     `json:"hand"`
	State BidStateJSON `json:"bid_state"`
}

type BidStateJSON struct {
	CurrentHigh int    `json:"current_high"`
	DealerSeat  int    `json:"dealer_seat"`
	Scores      [2]int `json:"scores"`
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
	state := game.BidState{
		Hand:        hand,
		CurrentHigh: req.State.CurrentHigh,
		DealerSeat:  req.State.DealerSeat,
		Scores:      req.State.Scores,
	}
	bid := strat.Bid(req.Seat, state)
	log.Printf("BID seat=%d hand=%v currentHigh=%d dealer=%d → %d",
		req.Seat, req.Hand, req.State.CurrentHigh, req.State.DealerSeat, bid)
	writeJSON(w, BidResponse{Bid: bid})
}

// --- Trump ---

type TrumpRequest struct {
	Seat int      `json:"seat"`
	Hand []string `json:"hand"`
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
	suit := strat.ChooseTrump(req.Seat, hand)
	writeJSON(w, TrumpResponse{Suit: suitCode(suit)})
}

// --- Play ---

type PlayRequest struct {
	Seat       int           `json:"seat"`
	ValidPlays []string      `json:"valid_plays"`
	State      TrickStateJSON `json:"state"`
}

type TrickStateJSON struct {
	Trump       string          `json:"trump"`
	BidderSeat  int             `json:"bidder_seat"`
	BidAmount   int             `json:"bid_amount"`
	TrickNumber int             `json:"trick_number"`
	TricksTaken [6]int          `json:"tricks_taken"`
	Scores      [2]int          `json:"scores"`
	Hand        []string        `json:"hand"`
	Trick       []PlayedCardJSON `json:"trick"`
	Leader      int             `json:"leader"`
	// History: each element is the flat list of card codes played in one completed trick.
	// PHP reconstructs this by exclusion: all deck cards minus current hands minus current trick.
	History     [][]string      `json:"history"`
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

	// Reconstruct hand history from previously played cards.
	var history game.HandHistory
	for _, trickCodes := range req.State.History {
		history.Record(parseCards(trickCodes))
	}

	hand := parseCards(req.State.Hand)

	// Use the caller-supplied valid plays. The external system tracks copy
	// indices correctly; recomputing here would misidentify already-played
	// duplicates as still-available (since parseCard always assigns CopyIndex=0).
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

	chosen := strat.Play(req.Seat, validPlays, state)
	log.Printf("PLAY seat=%d trump=%s trick=%v valid=%v hand=%v → %s",
		req.Seat, req.State.Trump, req.ValidPlays, req.State.Trick, req.State.Hand, cardCode(chosen))
	writeJSON(w, PlayResponse{Card: cardCode(chosen)})
}

// --- Helpers ---

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// cardCode returns a 2-char code like "JH", "AC", "9S".
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
