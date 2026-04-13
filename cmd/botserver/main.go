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
	"math/rand"
	"net/http"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/mlstrategy"
	"github.com/max/pepper/internal/strategy"
	"github.com/max/pepper/ml"
)

// peterStrat holds the special "Peter" personality (created at startup).
var peterStrat game.Strategy

// level45Strat is the half step between levels 4 and 5 (sharp play, Balanced bid).
var level45Strat game.Strategy

// level46Strat is the bidder variant of 4.5 (sharp bid, Balanced play).
var level46Strat game.Strategy

// personalityStrats holds named personalities (scientist, aggressive, cautious, risky).
var personalityStrats map[string]game.Strategy

// stratForRequest returns the strategy for a given level and personality.
// Level may be 1-5 or 45 (meaning 4.5).
func stratForRequest(level int, personality string) game.Strategy {
	if personality != "" {
		if personality == "peter" {
			return peterStrat
		}
		if s, ok := personalityStrats[personality]; ok {
			return s
		}
	}
	if level == 45 {
		return level45Strat
	}
	if level == 46 {
		return level46Strat
	}
	if level < 1 || level > 5 {
		level = defaultLevel
	}
	return strategies[level-1]
}

// strategies holds one strategy per difficulty level (index 0 = level 1, index 4 = level 5).
var strategies [5]game.Strategy
var defaultLevel int = 5

func main() {
	addr          := flag.String("addr", ":9090", "listen address")
	modelPath     := flag.String("model", "", "path to model_weights.json (enables MLP play)")
	bidModelPath  := flag.String("bid-model", "", "path to bid_model_weights.json (enables MLP bidding)")
	level         := flag.Int("level", 5, "default difficulty level (1-5)")
	flag.Parse()

	defaultLevel = *level
	if defaultLevel < 1 { defaultLevel = 1 }
	if defaultLevel > 5 { defaultLevel = 5 }

	var mlpStrat game.Strategy
	if *modelPath != "" {
		model, err := ml.LoadMLP(*modelPath)
		if err != nil {
			log.Fatalf("load MLP: %v", err)
		}
		ms := mlstrategy.NewMLPStrategy(model, strategy.Balanced)
		if *bidModelPath != "" {
			bidModel, err := ml.LoadBidMLP(*bidModelPath)
			if err != nil {
				log.Fatalf("load bid MLP: %v", err)
			}
			ms.WithBidModel(bidModel)
			log.Printf("bid MLP loaded from %s", *bidModelPath)
		}
		mlpStrat = ms
		log.Printf("MLP strategy loaded from %s", *modelPath)
	} else {
		log.Printf("no --model provided; levels 4-5 will play as Balanced")
	}

	// Build all 5 difficulty levels + special personalities.
	for i := 1; i <= 5; i++ {
		strategies[i-1] = strategy.NewDifficulty(i, mlpStrat, rand.New(rand.NewSource(int64(i*1000))))
	}
	peterStrat = strategy.NewPeter(mlpStrat, rand.New(rand.NewSource(9999)))
	level45Strat = strategy.NewLevel45(mlpStrat, rand.New(rand.NewSource(4545)))
	level46Strat = strategy.NewLevel46(mlpStrat, rand.New(rand.NewSource(4646)))
	personalityStrats = map[string]game.Strategy{
		"scientist":  strategy.NewPersonality("scientist", mlpStrat, rand.New(rand.NewSource(5001))),
		"aggressive": strategy.NewPersonality("aggressive", mlpStrat, rand.New(rand.NewSource(5002))),
		"cautious":   strategy.NewPersonality("cautious", mlpStrat, rand.New(rand.NewSource(5003))),
		"risky":      strategy.NewPersonality("risky", mlpStrat, rand.New(rand.NewSource(5004))),
	}
	log.Printf("difficulty levels 1-5 + 4.5 ready (default: %d), personalities: peter, scientist, aggressive, cautious, risky", defaultLevel)

	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/bid", handleBid)
	http.HandleFunc("/trump", handleTrump)
	http.HandleFunc("/play", handlePlay)
	http.HandleFunc("/advice", handleAdvice)

	log.Printf("bot server listening on %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, nil))
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, "ok")
}

// --- Bid ---

type BidRequest struct {
	Seat        int          `json:"seat"`
	Hand        []string     `json:"hand"`
	State       BidStateJSON `json:"bid_state"`
	Level       int          `json:"level,omitempty"`
	Personality string       `json:"personality,omitempty"`
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
	bid := stratForRequest(req.Level, req.Personality).Bid(req.Seat, state)
	log.Printf("BID seat=%d level=%d hand=%v currentHigh=%d dealer=%d → %d",
		req.Seat, req.Level, req.Hand, req.State.CurrentHigh, req.State.DealerSeat, bid)
	writeJSON(w, BidResponse{Bid: bid})
}

// --- Trump ---

type TrumpRequest struct {
	Seat        int      `json:"seat"`
	Hand        []string `json:"hand"`
	Level       int      `json:"level,omitempty"`
	Personality string   `json:"personality,omitempty"`
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
	suit := stratForRequest(req.Level, req.Personality).ChooseTrump(req.Seat, hand)
	writeJSON(w, TrumpResponse{Suit: suitCode(suit)})
}

// --- Play ---

type PlayRequest struct {
	Seat        int            `json:"seat"`
	ValidPlays  []string       `json:"valid_plays"`
	State       TrickStateJSON `json:"state"`
	Level       int            `json:"level,omitempty"`
	Personality string         `json:"personality,omitempty"`
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

	chosen := stratForRequest(req.Level, req.Personality).Play(req.Seat, validPlays, state)
	log.Printf("PLAY seat=%d level=%d trump=%s trick=%v valid=%v hand=%v → %s",
		req.Seat, req.Level, req.State.Trump, req.ValidPlays, req.State.Trick, req.State.Hand, cardCode(chosen))
	writeJSON(w, PlayResponse{Card: cardCode(chosen)})
}

// --- Advice (Ask Grandpa) ---

type AdviceRequest struct {
	Type       string         `json:"type"` // "bid" or "play"
	Seat       int            `json:"seat"`
	Hand       []string       `json:"hand"`
	BidState   *BidStateJSON  `json:"bid_state,omitempty"`
	ValidPlays []string       `json:"valid_plays,omitempty"`
	State      *TrickStateJSON `json:"state,omitempty"`
}

type AdviceResponse struct {
	Suggestion string `json:"suggestion"` // bid number or card code
	Explanation string `json:"explanation"`
}

func handleAdvice(w http.ResponseWriter, r *http.Request) {
	var req AdviceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	expert := stratForRequest(5, "") // always level 5 for grandpa

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
		bid := expert.Bid(req.Seat, state)
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
		chosen := expert.Play(req.Seat, validPlays, state)
		code := cardCode(chosen)
		log.Printf("ADVICE play seat=%d → %s", req.Seat, code)
		writeJSON(w, AdviceResponse{
			Suggestion: code,
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
