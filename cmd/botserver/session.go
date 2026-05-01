package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"sync"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/strategy"
	"github.com/max/pepper/ml"
)

// --- Phase ---

type Phase string

const (
	PhaseBidding       Phase = "bidding"
	PhaseTrump         Phase = "trump"
	PhasePepperDiscard Phase = "pepper_discard"
	PhasePlaying       Phase = "playing"
	PhaseHandOver      Phase = "hand_over"
	PhaseGameOver      Phase = "game_over"
)

// --- JSON types ---

type BidEntry struct {
	Seat   int  `json:"seat"`
	Bid    int  `json:"bid"`
	Forced bool `json:"forced,omitempty"`
}

type LastTrickJSON struct {
	Cards  []PlayedCardJSON `json:"cards"`
	Winner int              `json:"winner"`
}

type HandResultJSON struct {
	BidderSeat  int    `json:"bidder_seat"`
	BidAmount   int    `json:"bid_amount"`
	IsPepper    bool   `json:"is_pepper"`
	IsStuck     bool   `json:"is_stuck"`
	MadeBid     bool   `json:"made_bid"`
	TricksTaken [2]int `json:"tricks_taken"`
	ScoreDelta  [2]int `json:"score_delta"`
	NewScores   [2]int `json:"new_scores"`
}

// AdvisoryJSON is the single-pick recommendation shown during the human's turn.
// Source is "mlp" when a model is loaded, "balanced" otherwise.
// Call /evaluate/play or /evaluate/bid for the full ranked breakdown.
type AdvisoryJSON struct {
	Type   string `json:"type"`   // "bid" | "play" | "trump" | "discard"
	Source string `json:"source"` // "mlp" | "balanced"

	// play / trump
	Card  string  `json:"card,omitempty"`
	Score float32 `json:"score,omitempty"` // MLP score; omitted for balanced

	// bid
	Bid      int     `json:"bid,omitempty"`
	BidLabel string  `json:"bid_label,omitempty"`
	BidScore float32 `json:"bid_score,omitempty"`

	// top-N features driving this recommendation (MLP only)
	Attribution []AttributionEntry `json:"attribution,omitempty"`

	// pepper_discard only: cards received from partners
	ReceivedCards []string `json:"received_cards,omitempty"`
}

type SessionStateJSON struct {
	Phase     Phase  `json:"phase"`
	Scores    [2]int `json:"scores"`
	HumanSeat int    `json:"human_seat"`
	Dealer    int    `json:"dealer"`
	Round     int    `json:"round"`
	HumanHand []string `json:"human_hand"`

	ActiveSeat  int        `json:"active_seat"`
	BidsPlaced  []BidEntry `json:"bids_placed"`
	CurrentHigh int        `json:"current_high"`

	BidderSeat int    `json:"bidder_seat"`
	BidAmount  int    `json:"bid_amount"`
	IsPepper   bool   `json:"is_pepper"`
	IsStuck    bool   `json:"is_stuck"`
	Trump      string `json:"trump,omitempty"`
	SittingOut []int  `json:"sitting_out,omitempty"`

	TrickNumber  int              `json:"trick_number"`
	TricksTaken  [6]int           `json:"tricks_taken"`
	CurrentTrick []PlayedCardJSON `json:"current_trick"`
	LastTrick    *LastTrickJSON   `json:"last_trick,omitempty"`

	Advisory   *AdvisoryJSON   `json:"advisory,omitempty"`
	HandResult *HandResultJSON `json:"hand_result,omitempty"`
	Winner     int             `json:"winner"` // -1 if game not over
}

// --- Session ---

type Session struct {
	mu sync.Mutex

	humanSeat int
	botCfg    BotConfigJSON
	gs        *game.GameState
	hands     [6][]card.Card
	bots      [6]game.Strategy
	rng       *rand.Rand
	phase    Phase
	advisory *AdvisoryJSON

	// Bidding state
	bidStep       int
	bidsPlaced    []BidEntry
	currentHigh   int
	highSeat      int
	passesSoFar   int
	partnerHasBid [6]bool
	seatBidLevel  [6]int
	bidResult     game.BidResult

	// Post-bid state
	trump           card.Suit
	trumpSet        bool
	pepperActive    bool
	sittingOut      [2]int
	activeSeatSet   []int
	pepperReceived  [2]card.Card

	// Playing state
	trickNum     int
	trickStep    int
	leader       int
	currentTrick *game.Trick
	tricksTaken  [6]int
	history      game.HandHistory
	lastTrick    *LastTrickJSON

	handResult *HandResultJSON
	winner     int
}

// --- Global session ---

var (
	globalSession *Session
	sessionMu     sync.Mutex
)

func getSession() (*Session, error) {
	sessionMu.Lock()
	s := globalSession
	sessionMu.Unlock()
	if s == nil {
		return nil, fmt.Errorf("no active game session")
	}
	return s, nil
}

// --- Construction ---

func newSession(humanSeat int, cfg BotConfigJSON, seed int64) *Session {
	if seed == 0 {
		seed = 42
	}
	s := &Session{
		humanSeat: humanSeat,
		botCfg:    cfg,
		gs:        game.NewGame(0),
		rng:       rand.New(rand.NewSource(seed)),
		highSeat:  -1,
		bidStep:   1,
		winner:    -1,
	}
	for i := range s.bots {
		if i != humanSeat {
			s.bots[i] = botForConfig(cfg)
		}
	}
	s.hands = card.Deal(s.rng)
	s.phase = PhaseBidding
	s.advanceUntilHuman()
	return s
}

// --- Advance loop ---

func (s *Session) advanceUntilHuman() {
	for {
		if s.step() {
			return
		}
	}
}

// step performs one game action (bot move or phase transition).
// Returns true if the game is now waiting on human input or has ended.
func (s *Session) step() bool {
	switch s.phase {
	case PhaseBidding:
		return s.stepBidding()
	case PhaseTrump:
		return s.stepTrump()
	case PhasePepperDiscard:
		return s.stepPepperDiscard()
	case PhasePlaying:
		return s.stepPlaying()
	default:
		return true
	}
}

func (s *Session) stepBidding() bool {
	seat := (s.gs.Dealer + s.bidStep) % 6

	// Stuck dealer: everyone else passed, dealer is forced to bid 3.
	if seat == s.gs.Dealer && s.highSeat == -1 {
		s.bidsPlaced = append(s.bidsPlaced, BidEntry{Seat: seat, Bid: game.StuckBid, Forced: true})
		s.bidResult = game.BidResult{Winner: seat, Amount: game.StuckBid, IsStuck: true}
		s.bidStep++
		s.finalizeBidding()
		return false
	}

	if seat == s.humanSeat {
		if s.advisory == nil {
			s.advisory = s.computeBidAdvisory()
		}
		return true
	}

	bs := s.buildBidState(seat)
	bid := s.bots[seat].Bid(seat, &bs)
	s.recordBid(seat, bid)
	s.bidStep++

	if bid == game.PepperBid {
		s.bidResult = game.BidResult{Winner: seat, Amount: game.PepperBid, IsPepper: true}
		s.finalizeBidding()
		return false
	}

	if s.bidStep > 6 {
		s.finalizeBidding()
	}
	return false
}

func (s *Session) stepTrump() bool {
	caller := s.bidResult.Winner
	if caller == s.humanSeat {
		if s.advisory == nil {
			s.advisory = s.computeTrumpAdvisory()
		}
		return true
	}
	trump := s.bots[caller].ChooseTrump(caller, s.hands[caller])
	s.applyTrumpInternal(trump)
	return false
}

func (s *Session) stepPepperDiscard() bool {
	caller := s.bidResult.Winner
	if caller == s.humanSeat {
		if s.advisory == nil {
			s.advisory = s.computeDiscardAdvisory()
		}
		return true
	}
	// Shouldn't reach here; bot callers are handled in applyTrumpInternal.
	s.autoDiscard()
	s.startPlaying()
	return false
}

func (s *Session) stepPlaying() bool {
	// Trick complete — finalize it.
	if s.trickStep >= len(s.activeSeatSet) {
		s.completeTrick()
		return s.phase != PhasePlaying
	}

	seat := s.currentPlaySeat()
	if seat == s.humanSeat {
		if s.advisory == nil {
			s.advisory = s.computePlayAdvisory()
		}
		return true
	}

	valid := game.ValidPlays(s.hands[seat], s.currentTrick, s.trump)
	ts := s.buildTrickState(seat)
	chosen := s.bots[seat].Play(seat, valid, &ts)
	s.playCard(seat, chosen)
	return false
}

// --- Phase transitions ---

func (s *Session) finalizeBidding() {
	if !s.bidResult.IsPepper && !s.bidResult.IsStuck {
		s.bidResult = game.BidResult{
			Winner: s.highSeat,
			Amount: s.currentHigh,
		}
	}
	s.phase = PhaseTrump
	s.advisory = nil
}

func (s *Session) applyTrumpInternal(trump card.Suit) {
	s.trump = trump
	s.trumpSet = true

	if s.bidResult.IsPepper {
		s.doPepperGive()
		if s.bidResult.Winner == s.humanSeat {
			s.phase = PhasePepperDiscard
		} else {
			s.autoDiscard()
			s.startPlaying()
		}
	} else {
		s.startPlaying()
	}
	s.advisory = nil
}

func (s *Session) doPepperGive() {
	caller := s.bidResult.Winner
	partners := game.Partners(caller)
	var received [2]card.Card
	for i, p := range partners {
		given := s.bots[p].GivePepper(p, s.hands[p], s.trump)
		received[i] = given
		s.hands[p] = removeCardFromHand(s.hands[p], given)
	}
	s.hands[caller] = append(s.hands[caller], received[0], received[1])
	s.pepperReceived = received
	s.pepperActive = true
	s.sittingOut = partners
}

func (s *Session) autoDiscard() {
	caller := s.bidResult.Winner
	discards := s.bots[caller].PepperDiscard(caller, s.hands[caller], s.trump, s.pepperReceived)
	for _, d := range discards {
		s.hands[caller] = removeCardFromHand(s.hands[caller], d)
	}
}

func (s *Session) startPlaying() {
	s.activeSeatSet = buildActiveSeatSetSession(s.pepperActive, s.bidResult.Winner, s.sittingOut)
	s.leader = s.bidResult.Winner
	s.currentTrick = game.NewTrick(s.leader, s.trump)
	s.trickNum = 0
	s.trickStep = 0
	s.tricksTaken = [6]int{}
	s.history.Reset()
	s.phase = PhasePlaying
	s.advisory = nil
}

func (s *Session) playCard(seat int, c card.Card) {
	s.currentTrick.Add(c, seat)
	s.hands[seat] = removeCardFromHand(s.hands[seat], c)
	s.trickStep++
}

func (s *Session) completeTrick() {
	winner := s.currentTrick.Winner()
	s.tricksTaken[winner]++
	s.lastTrick = &LastTrickJSON{
		Cards:  playedCardsToJSON(s.currentTrick.Cards[:s.currentTrick.NCards]),
		Winner: winner,
	}
	s.history.RecordTrick(s.currentTrick.Cards[:s.currentTrick.NCards])
	s.trickNum++

	if s.trickNum == game.TotalTricks {
		s.completeHand()
		return
	}

	s.leader = winner
	s.currentTrick = game.NewTrick(winner, s.trump)
	s.trickStep = 0
}

func (s *Session) completeHand() {
	var tricksByTeam [2]int
	for seat, count := range s.tricksTaken {
		tricksByTeam[game.TeamOf(seat)] += count
	}
	result := game.ScoreHand(s.bidResult.Winner, s.bidResult.Amount, s.bidResult.IsPepper, tricksByTeam)
	newScores := [2]int{
		s.gs.Scores[0] + result.ScoreDelta[0],
		s.gs.Scores[1] + result.ScoreDelta[1],
	}
	s.handResult = &HandResultJSON{
		BidderSeat:  result.BidderSeat,
		BidAmount:   result.BidAmount,
		IsPepper:    result.IsPepper,
		IsStuck:     result.IsStuck,
		MadeBid:     result.MadeBid,
		TricksTaken: result.TricksTaken,
		ScoreDelta:  result.ScoreDelta,
		NewScores:   newScores,
	}
	s.phase = PhaseHandOver
	s.advisory = nil
}

func (s *Session) doNextHand() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.phase != PhaseHandOver {
		return fmt.Errorf("not in hand_over phase")
	}

	s.gs.Scores = s.handResult.NewScores

	if over, winner := s.gs.IsOver(); over {
		s.phase = PhaseGameOver
		s.winner = int(winner)
		return nil
	}

	s.gs.NextDealer()
	s.resetForNewHand()
	s.hands = card.Deal(s.rng)
	s.phase = PhaseBidding
	s.advanceUntilHuman()
	return nil
}

func (s *Session) resetForNewHand() {
	s.bidStep = 1
	s.bidsPlaced = nil
	s.currentHigh = 0
	s.highSeat = -1
	s.passesSoFar = 0
	s.partnerHasBid = [6]bool{}
	s.seatBidLevel = [6]int{}
	s.bidResult = game.BidResult{}
	s.trumpSet = false
	s.pepperActive = false
	s.sittingOut = [2]int{}
	s.activeSeatSet = nil
	s.pepperReceived = [2]card.Card{}
	s.trickNum = 0
	s.trickStep = 0
	s.leader = 0
	s.currentTrick = nil
	s.tricksTaken = [6]int{}
	s.history.Reset()
	s.lastTrick = nil
	s.handResult = nil
	s.advisory = nil
}

// --- Human actions ---

func (s *Session) applyBid(bid int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.phase != PhaseBidding {
		return fmt.Errorf("not in bidding phase")
	}
	seat := (s.gs.Dealer + s.bidStep) % 6
	if seat != s.humanSeat {
		return fmt.Errorf("not human's turn to bid")
	}

	s.advisory = nil
	s.recordBid(seat, bid)
	s.bidStep++

	if bid == game.PepperBid {
		s.bidResult = game.BidResult{Winner: seat, Amount: game.PepperBid, IsPepper: true}
		s.finalizeBidding()
	} else if s.bidStep > 6 {
		s.finalizeBidding()
	}

	s.advanceUntilHuman()
	return nil
}

func (s *Session) applyTrump(trump card.Suit) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.phase != PhaseTrump {
		return fmt.Errorf("not in trump phase")
	}
	if s.bidResult.Winner != s.humanSeat {
		return fmt.Errorf("not human's turn to pick trump")
	}

	s.advisory = nil
	s.applyTrumpInternal(trump)
	s.advanceUntilHuman()
	return nil
}

func (s *Session) applyDiscard(discards []card.Card) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.phase != PhasePepperDiscard {
		return fmt.Errorf("not in pepper_discard phase")
	}
	if s.bidResult.Winner != s.humanSeat {
		return fmt.Errorf("not human's discard turn")
	}

	// Resolve each discard by suit+rank from the current hand.
	hand := s.hands[s.humanSeat]
	resolved := make([]card.Card, 0, len(discards))
	for _, d := range discards {
		actual, ok := findCardBySuitRank(hand, d)
		if !ok {
			return fmt.Errorf("card %s not in hand", cardCode(d))
		}
		resolved = append(resolved, actual)
		hand = removeCardFromHand(hand, actual)
	}

	s.advisory = nil
	for _, d := range resolved {
		s.hands[s.humanSeat] = removeCardFromHand(s.hands[s.humanSeat], d)
	}
	s.startPlaying()
	s.advanceUntilHuman()
	return nil
}

func (s *Session) applyPlay(c card.Card) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.phase != PhasePlaying {
		return fmt.Errorf("not in playing phase")
	}
	seat := s.currentPlaySeat()
	if seat != s.humanSeat {
		return fmt.Errorf("not human's turn to play")
	}

	valid := game.ValidPlays(s.hands[seat], s.currentTrick, s.trump)
	// Match by suit+rank (CopyIndex is unknown to client).
	actual, ok := findCardBySuitRank(valid, c)
	if !ok {
		return fmt.Errorf("card %s is not a valid play", cardCode(c))
	}

	s.advisory = nil
	s.playCard(seat, actual)

	if s.trickStep >= len(s.activeSeatSet) {
		s.completeTrick()
	}
	if s.phase == PhasePlaying {
		s.advanceUntilHuman()
	}
	return nil
}

// --- Bid state helpers ---

func (s *Session) recordBid(seat, bid int) {
	s.bidsPlaced = append(s.bidsPlaced, BidEntry{Seat: seat, Bid: bid})
	if bid == game.PassBid {
		s.passesSoFar++
		return
	}
	if bid == game.PepperBid {
		return
	}
	minRequired := game.MinBid
	if s.currentHigh >= game.MinBid {
		minRequired = s.currentHigh + 1
	}
	if bid >= minRequired {
		s.currentHigh = bid
		s.highSeat = seat
		s.partnerHasBid[seat] = true
		s.seatBidLevel[seat] = bid
	}
}

func (s *Session) buildBidState(seat int) game.BidState {
	myTeam := game.TeamOf(seat)
	anyPartnerBid := false
	partnerBidLevel := 0
	for ts := 0; ts < 6; ts++ {
		if ts != seat && game.TeamOf(ts) == myTeam && s.partnerHasBid[ts] {
			anyPartnerBid = true
			if s.seatBidLevel[ts] > partnerBidLevel {
				partnerBidLevel = s.seatBidLevel[ts]
			}
		}
	}
	i := ((seat - s.gs.Dealer) + 6) % 6
	seatsLeft := 6 - i
	return game.BidState{
		Hand:            s.hands[seat],
		Seat:            seat,
		DealerSeat:      s.gs.Dealer,
		CurrentHigh:     s.currentHigh,
		HighSeat:        s.highSeat,
		SeatsLeft:       seatsLeft,
		Scores:          s.gs.Scores,
		PassesSoFar:     s.passesSoFar,
		PartnerHasBid:   anyPartnerBid,
		PartnerBidLevel: partnerBidLevel,
	}
}

func (s *Session) buildTrickState(seat int) game.TrickState {
	return game.TrickState{
		Trick:       s.currentTrick,
		Trump:       s.trump,
		Seat:        seat,
		BidderSeat:  s.bidResult.Winner,
		BidAmount:   s.bidResult.Amount,
		TrickNumber: s.trickNum,
		TricksTaken: s.tricksTaken,
		Scores:      s.gs.Scores,
		History:     &s.history,
		Hand:        s.hands[seat],
	}
}

func (s *Session) currentPlaySeat() int {
	li := s.indexInActive(s.leader)
	return s.activeSeatSet[(li+s.trickStep)%len(s.activeSeatSet)]
}

func (s *Session) indexInActive(seat int) int {
	for i, v := range s.activeSeatSet {
		if v == seat {
			return i
		}
	}
	return 0
}

// --- Advisory computation ---

func (s *Session) computeBidAdvisory() *AdvisoryJSON {
	seat := s.humanSeat
	hand := s.hands[seat]
	bidState := s.buildBidState(seat)

	if bidModel != nil {
		ctx := ml.BidContext(
			seat, hand,
			s.gs.Dealer, s.currentHigh, s.highSeat,
			bidState.SeatsLeft, s.gs.Scores,
			s.passesSoFar, bidState.PartnerHasBid, bidState.PartnerBidLevel, s.seatBidLevel,
		)
		validBids := ml.ValidBidLevels(s.currentHigh)
		n := len(validBids)
		featFlat := bidModel.BatchFeatBuf
		feats := make([][ml.BidTotalLen]float32, n)
		for i, b := range validBids {
			f := ml.AppendBidAction(ctx, b, s.currentHigh)
			feats[i] = f
			copy(featFlat[i*ml.BidTotalLen:], f[:])
		}
		rawScores := make([]float32, n)
		bidModel.ScoreBatch(n, featFlat[:n*ml.BidTotalLen], rawScores)

		bestIdx, bestScore := 0, rawScores[0]
		for i, sc := range rawScores {
			if sc > bestScore {
				bestIdx, bestScore = i, sc
			}
		}
		raw := bidModel.AttributeBid(feats[bestIdx])
		return &AdvisoryJSON{
			Type:        "bid",
			Source:      "mlp",
			Bid:         validBids[bestIdx],
			BidLabel:    bidLabel(validBids[bestIdx]),
			BidScore:    bestScore,
			Attribution: sortedAttribution(raw),
		}
	}

	balancedStrat := strategy.NewStandard(strategy.Balanced)
	bid := balancedStrat.Bid(seat, &bidState)
	return &AdvisoryJSON{
		Type:     "bid",
		Source:   "balanced",
		Bid:      bid,
		BidLabel: bidLabel(bid),
	}
}

func (s *Session) computePlayAdvisory() *AdvisoryJSON {
	seat := s.humanSeat
	hand := s.hands[seat]
	valid := game.ValidPlays(hand, s.currentTrick, s.trump)
	state := s.buildTrickState(seat)

	if cardModel != nil {
		isBidder := game.TeamOf(seat) == game.TeamOf(s.bidResult.Winner)
		sign := float32(1)
		if !isBidder {
			sign = -1
		}

		ctx := ml.ExtractContext(seat, hand, state, 6)
		n := len(valid)
		featFlat := cardModel.BatchFeatBuf
		feats := make([][ml.TotalFeatureLen]float32, n)
		for i, c := range valid {
			f := ml.AppendCard(ctx, c, s.trump, hand, s.currentTrick, &s.history)
			feats[i] = f
			copy(featFlat[i*ml.TotalFeatureLen:], f[:])
		}
		rawScores := make([]float32, n)
		cardModel.ScoreBatch(n, featFlat[:n*ml.TotalFeatureLen], rawScores)

		bestIdx, bestScore := 0, rawScores[0]*sign
		for i, sc := range rawScores {
			if adj := sc * sign; adj > bestScore {
				bestIdx, bestScore = i, adj
			}
		}
		raw := cardModel.AttributePlay(feats[bestIdx], isBidder)
		return &AdvisoryJSON{
			Type:        "play",
			Source:      "mlp",
			Card:        cardCode(valid[bestIdx]),
			Score:       bestScore,
			Attribution: sortedAttribution(raw),
		}
	}

	balancedStrat := strategy.NewStandard(strategy.Balanced)
	chosen := balancedStrat.Play(seat, valid, &state)
	return &AdvisoryJSON{
		Type:   "play",
		Source: "balanced",
		Card:   cardCode(chosen),
	}
}

func (s *Session) computeTrumpAdvisory() *AdvisoryJSON {
	seat := s.humanSeat
	balancedStrat := strategy.NewStandard(strategy.Balanced)
	trump := balancedStrat.ChooseTrump(seat, s.hands[seat])
	return &AdvisoryJSON{
		Type:   "trump",
		Source: "balanced",
		Card:   suitCode(trump),
	}
}

func (s *Session) computeDiscardAdvisory() *AdvisoryJSON {
	return &AdvisoryJSON{
		Type:          "discard",
		Source:        "balanced",
		ReceivedCards: []string{cardCode(s.pepperReceived[0]), cardCode(s.pepperReceived[1])},
	}
}

// --- State serialization ---

func (s *Session) getState() SessionStateJSON {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buildStateJSON()
}

func (s *Session) buildStateJSON() SessionStateJSON {
	state := SessionStateJSON{
		Phase:       s.phase,
		Scores:      s.gs.Scores,
		HumanSeat:   s.humanSeat,
		Dealer:      s.gs.Dealer,
		Round:       s.gs.Round,
		HumanHand:   cardsToStrings(s.hands[s.humanSeat]),
		ActiveSeat:  s.currentActiveSeat(),
		BidsPlaced:  s.bidsPlaced,
		CurrentHigh: s.currentHigh,
		BidderSeat:  s.bidResult.Winner,
		BidAmount:   s.bidResult.Amount,
		IsPepper:    s.bidResult.IsPepper,
		IsStuck:     s.bidResult.IsStuck,
		TrickNumber: s.trickNum,
		TricksTaken: s.tricksTaken,
		Advisory:    s.advisory,
		HandResult:  s.handResult,
		Winner:      s.winner,
	}
	if s.trumpSet {
		state.Trump = suitCode(s.trump)
	}
	if s.pepperActive {
		state.SittingOut = []int{s.sittingOut[0], s.sittingOut[1]}
	}
	if s.currentTrick != nil {
		state.CurrentTrick = playedCardsToJSON(s.currentTrick.Cards[:s.currentTrick.NCards])
	}
	state.LastTrick = s.lastTrick
	return state
}

func (s *Session) currentActiveSeat() int {
	switch s.phase {
	case PhaseBidding:
		if s.bidStep <= 6 {
			return (s.gs.Dealer + s.bidStep) % 6
		}
	case PhaseTrump, PhasePepperDiscard:
		return s.bidResult.Winner
	case PhasePlaying:
		if s.currentTrick != nil && s.trickStep < len(s.activeSeatSet) {
			return s.currentPlaySeat()
		}
	}
	return -1
}

// --- HTTP handlers ---

type NewGameRequest struct {
	Seat      int           `json:"seat"`
	BotConfig BotConfigJSON `json:"bot_config"`
	Seed      int64         `json:"seed"`
}

func handleGameNew(w http.ResponseWriter, r *http.Request) {
	var req NewGameRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.Seat < 0 || req.Seat > 5 {
		http.Error(w, "seat must be 0–5", http.StatusBadRequest)
		return
	}

	sess := newSession(req.Seat, req.BotConfig, req.Seed)

	sessionMu.Lock()
	globalSession = sess
	sessionMu.Unlock()

	writeJSON(w, sess.getState())
}

func handleGameState(w http.ResponseWriter, r *http.Request) {
	sess, err := getSession()
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	writeJSON(w, sess.getState())
}

type GameBidRequest struct {
	Bid int `json:"bid"`
}

func handleGameBid(w http.ResponseWriter, r *http.Request) {
	sess, err := getSession()
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	var req GameBidRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := sess.applyBid(req.Bid); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, sess.getState())
}

type GameTrumpRequest struct {
	Suit string `json:"suit"`
}

func handleGameTrump(w http.ResponseWriter, r *http.Request) {
	sess, err := getSession()
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	var req GameTrumpRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := sess.applyTrump(parseSuit(req.Suit)); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, sess.getState())
}

type GameDiscardRequest struct {
	Cards []string `json:"cards"`
}

func handleGameDiscard(w http.ResponseWriter, r *http.Request) {
	sess, err := getSession()
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	var req GameDiscardRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	discards := parseCards(req.Cards)
	if len(discards) != 2 {
		http.Error(w, "must discard exactly 2 cards", http.StatusBadRequest)
		return
	}
	if err := sess.applyDiscard(discards); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, sess.getState())
}

type GamePlayRequest struct {
	Card string `json:"card"`
}

func handleGamePlay(w http.ResponseWriter, r *http.Request) {
	sess, err := getSession()
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	var req GamePlayRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := sess.applyPlay(parseCard(req.Card)); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, sess.getState())
}

func handleGameNextHand(w http.ResponseWriter, r *http.Request) {
	sess, err := getSession()
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	if err := sess.doNextHand(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, sess.getState())
}

// --- Local helpers ---

func cardsToStrings(cards []card.Card) []string {
	result := make([]string, len(cards))
	for i, c := range cards {
		result[i] = cardCode(c)
	}
	return result
}

func playedCardsToJSON(cards []game.PlayedCard) []PlayedCardJSON {
	result := make([]PlayedCardJSON, len(cards))
	for i, pc := range cards {
		result[i] = PlayedCardJSON{Card: cardCode(pc.Card), Seat: pc.Seat}
	}
	return result
}

func removeCardFromHand(hand []card.Card, target card.Card) []card.Card {
	for i, c := range hand {
		if c.Equal(target) {
			result := make([]card.Card, len(hand)-1)
			copy(result, hand[:i])
			copy(result[i:], hand[i+1:])
			return result
		}
	}
	return hand
}

// findCardBySuitRank returns the first card in the slice matching suit+rank,
// ignoring CopyIndex (which the client cannot know).
func findCardBySuitRank(cards []card.Card, target card.Card) (card.Card, bool) {
	for _, c := range cards {
		if c.Suit == target.Suit && c.Rank == target.Rank {
			return c, true
		}
	}
	return card.Card{}, false
}

func buildActiveSeatSetSession(pepperActive bool, callerSeat int, sittingOut [2]int) []int {
	if !pepperActive {
		seats := make([]int, 6)
		for i := range seats {
			seats[i] = i
		}
		return seats
	}
	sitting := map[int]bool{sittingOut[0]: true, sittingOut[1]: true}
	var active []int
	for i := 0; i < 6; i++ {
		if !sitting[i] {
			active = append(active, i)
		}
	}
	return active
}
