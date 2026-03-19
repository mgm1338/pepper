package strategy

import (
	"math/rand"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

// PlayProfile holds situation-specific card-play parameters.
// The StandardStrategy selects among Normal, Deficit, and Endgame profiles
// at each play decision based on the current game state.
type PlayProfile struct {
	// LeadHigh overrides all normal lead priority: always lead the highest
	// available card (trump first if on bidding team, off-suit first if defending).
	// Useful for deficit catch-up or endgame all-in situations.
	LeadHigh bool

	// ── Bidding team lead ──────────────────────────────────────────────────

	// PullTrumpWithRight: when leading as the bidding team and holding the right
	// bower, lead it immediately to start extracting opponents' trump.
	PullTrumpWithRight bool

	// PullTrumpMinCount: keep leading trump on subsequent tricks while holding
	// at least this many trump cards and opponents still appear to hold trump.
	// Lower values pull trump more aggressively; higher values switch to cashing
	// off-suit winners sooner. Range 0–7.
	PullTrumpMinCount int

	// CashAcesEarly: on the bidding team, cash off-suit aces immediately after
	// the right bower rather than continuing trump extraction.
	CashAcesEarly bool

	// VoidHunting: when leading, play from a short suit to try to establish a
	// void so your partner can ruff later. Applied to both bidding and defensive leads.
	VoidHunting bool

	// ── Both teams follow ──────────────────────────────────────────────────

	// DuckAndCover: when following a trick your partner is already winning,
	// play your lowest legal card to preserve high cards for later tricks.
	DuckAndCover bool

	// OvertrumpPartner: when partner is winning with trump and you hold a higher
	// trump, over-trump to consolidate bidding team trump. Overrides DuckAndCover
	// for this specific case. Only applied on bidding team.
	OvertrumpPartner bool

	// ── Defensive lead ─────────────────────────────────────────────────────

	// DefensiveLeadRight: cash the right bower immediately when leading on defense.
	DefensiveLeadRight bool

	// DefensiveSaveRight: even when DefensiveLeadRight or trump-lead thresholds
	// would trigger, hold the right bower for a counter-punch.
	DefensiveSaveRight bool

	// DefensiveAvoidLeadingIntoHand: do not lead trump when the bidder plays
	// immediately after us in seat order.
	DefensiveAvoidLeadingIntoHand bool

	// DefensiveTrumpWithRightThreshold: lead trump on defense when holding the right
	// bower AND at least this many total trump. 0 = lead any time. Range 0–5.
	DefensiveTrumpWithRightThreshold int

	// DefensiveTrumpLeadMin: lead trump on defense when holding at least this many
	// trump total regardless of right bower. 0 = disabled. Range 0–5.
	DefensiveTrumpLeadMin int

	// DefensiveLeadKing: lead an off-suit king when no ace is available.
	DefensiveLeadKing bool

	// DefensiveLeadHigh: lead the highest non-trump on defense rather than ace-first.
	DefensiveLeadHigh bool

	// ── Defensive follow ───────────────────────────────────────────────────

	// DefensiveHighFollow: when following on defense, play the highest winning
	// card rather than the cheapest winner.
	DefensiveHighFollow bool

	// DefensiveSacrificeLead: when following on defense and unable to win,
	// play highest non-trump to pressure bidder's partners.
	DefensiveSacrificeLead bool
}

// Config defines the tunable parameters for a StandardStrategy.
// Different configs represent different player personalities.
type Config struct {
	Name string

	// ── Bidding ────────────────────────────────────────────────────────────────

	// PartnerTricksEstimate is how many tricks you expect your partner to contribute
	// when you add up whether to bid. Higher values make you more willing to enter
	// the auction on borderline hands. Evolved alongside BidPadding — a high estimate
	// paired with a negative pad produces finer-grained bid control than either alone.
	// Range 0.5–2.5.
	PartnerTricksEstimate float64

	// BidPadding is a flat offset added to your computed bid after all other adjustments.
	// 0 = bid honestly. -1 = shade one lower (useful when PartnerTricksEstimate is inflated).
	// 1–2 = bluff higher to steal the hand. Range -1 to 2.
	BidPadding int

	// Bid5Threshold: bid 5 instead of flooring to 4 whenever your total trick estimate
	// meets or exceeds this value (and is below 6). Lower = jump to 5 more readily,
	// stealing the hand and forcing opponents to beat 5. Higher = only bid 5 when you
	// genuinely expect to take 5 tricks. Range 4.0–5.2.
	Bid5Threshold float64

	// Bid6Threshold: bid 6 instead of flooring to 5 whenever your total trick estimate
	// meets or exceeds this value (and is below 7). Works the same as Bid5Threshold but
	// for the 6-bid jump. When both thresholds are set, Bid6Threshold takes priority if
	// estimate qualifies. Range 5.0–6.2. 0 = disabled.
	Bid6Threshold float64

	// ScoreDeficitFactor controls catch-up aggression in bidding. For each point you
	// trail the opponents, this amount is added to your trick estimate. A value of 0.03
	// means being 10 points behind adds 0.3 to your estimate, making you more likely
	// to enter the auction on a marginal hand. Range 0.0–0.05.
	ScoreDeficitFactor float64

	// ScoreSurplusFactor controls protect-the-lead conservatism in bidding. For each
	// point you lead the opponents, this amount is subtracted from your trick estimate.
	// A value of 0.04 means being 10 points ahead cuts 0.4 from your estimate, nudging
	// you to pass and let the opponents dig themselves into a hole. Range 0.0–0.05.
	ScoreSurplusFactor float64

	// ScoreCloseoutBonus is added to your trick estimate when your team is within 16
	// points of the winning score (64). The +16 swing from any single bid win or a
	// pepper can close the game — this encodes the urgency to get into the auction
	// and finish it rather than playing safe. Range 0.0–0.5.
	ScoreCloseoutBonus float64

	// SeatPositionBias adds a small bonus to your trick estimate based on how late you
	// are in the bidding order. The first seat to bid (left of dealer) gets no bonus.
	// Each seat closer to the dealer adds one more unit of this bias. The dealer, who
	// has seen everyone pass and knows the hand is weaker overall, gets the full 5x bonus
	// and can open more loosely. Range 0.0–0.15.
	SeatPositionBias float64

	// OvercallBias is added to your trick estimate whenever an opponent already holds
	// the high bid. It encodes the competitive instinct to fight for hands rather than
	// concede them — even on cards you would normally pass. A value of 0.30 means you
	// will enter nearly any auction where you have four tricks of your own. Range 0.0–0.5.
	OvercallBias float64

	// OpeningBidFactor multiplies PartnerTricksEstimate when you are the first to bid
	// (no prior bid in the auction). Opening the auction is a statement of strength —
	// you have no information about opponents' hands. A value < 1.0 makes opening bids
	// tighter; > 1.0 makes them looser. 1.0 = no adjustment. Range 0.5–1.5.
	OpeningBidFactor float64

	// ── Trump suit scoring weights ─────────────────────────────────────────────
	// These weights control how much each trump card contributes to the trick estimate
	// used for both suit selection and bid sizing. They were previously hardcoded and
	// are now evolvable so the optimizer can calibrate them from self-play.

	// RightBowerScore is the trick-estimate value for the right bower. Range 1.0–2.5.
	RightBowerScore float64

	// LeftBowerScore is the trick-estimate value for the left bower. Range 0.8–2.0.
	LeftBowerScore float64

	// AceKingScore is the trick-estimate value for an ace or king of trump. Range 0.5–1.5.
	AceKingScore float64

	// LowTrumpScore is the trick-estimate value for any other trump card (Q, J, 10, 9).
	// Range 0.1–0.8.
	LowTrumpScore float64

	// ── Trump suit selection ───────────────────────────────────────────────────

	// MajorSuitBonus is added to the trick estimate when evaluating spades or hearts
	// as a potential trump suit. In a pinochle deck the card distribution is symmetric,
	// but spades and hearts tend to produce slightly cleaner trump pulls because the
	// partner suits (clubs/diamonds) leave fewer stranded off-suit cards. The evolver
	// consistently converged on ~0.24 across independent runs. Range 0.0–0.3.
	MajorSuitBonus float64

	// TrumpLengthBonus is added to the suit score for each trump card held beyond
	// the base of 5. A longer trump suit is disproportionately powerful: each extra
	// card is a near-guaranteed trick after the high trump are drawn. A value of 0.1
	// means holding 7 trump adds 0.2 to the suit evaluation score. Range 0.0–0.2.
	TrumpLengthBonus float64

	// ── Pepper calling ────────────────────────────────────────────────────────

	// PepperRequireBothRights requires that you hold both right bowers (one per deck
	// in a pinochle set) before calling pepper. Without both rights, the opponents can
	// take a trick on the very first lead. Almost always true in strong play.
	PepperRequireBothRights bool

	// PepperMinLeftBowers is the minimum number of left bowers you must hold to call
	// pepper. 0 = rights alone are enough. 1 = need at least one left to back them up.
	// 2 = need the full bower set. Higher thresholds reduce pepper calls to only the
	// most dominant hands. Range 0–2.
	PepperMinLeftBowers int

	// PepperMinTrump is the total trump card count required to call pepper, including
	// bowers. A higher threshold means you only pepper with a near-complete trump suit,
	// reducing the chance of being set for -16. Range 5–9.
	PepperMinTrump int

	// PepperDiscardKeepAces controls what you throw away after receiving your partners'
	// trump cards in a pepper hand. When true, aces are protected from discard — an
	// off-suit ace is almost a guaranteed trick and should rarely be thrown. When false,
	// the discard picks purely by lowest rank, which can give up aces needlessly.
	PepperDiscardKeepAces bool

	// ── Card play — situation-selected profiles ─────────────────────────────────

	// Normal is the default play profile used in most game states.
	Normal PlayProfile

	// Deficit is used when the bidding team needs more than DeficitRatio of the
	// remaining tricks to make their bid. Represents a catch-up situation.
	Deficit PlayProfile

	// Endgame is used when fewer than EndgameTrickThreshold tricks remain in the
	// hand. Takes priority over Deficit when both conditions are met.
	Endgame PlayProfile

	// DeficitRatio is the threshold for switching to the Deficit profile.
	// Triggers when bidding team needs more than this fraction of remaining tricks.
	// Range 0.3–0.9. Default 0.5.
	DeficitRatio float64

	// EndgameTrickThreshold is how many tricks remaining triggers the Endgame profile.
	// Compared to (8 - TrickNumber). 0 = disabled. Range 0–4.
	EndgameTrickThreshold int
}

// conservativeProfile is the shared play profile for the Conservative preset.
var conservativeProfile = PlayProfile{
	LeadHigh:                         false,
	PullTrumpWithRight:               true,
	PullTrumpMinCount:                0,
	CashAcesEarly:                    false,
	VoidHunting:                      false,
	DuckAndCover:                     true,
	OvertrumpPartner:                 false,
	DefensiveLeadRight:               false,
	DefensiveSaveRight:               true,
	DefensiveAvoidLeadingIntoHand:    true,
	DefensiveTrumpWithRightThreshold: 0,
	DefensiveTrumpLeadMin:            0,
	DefensiveLeadKing:                false,
	DefensiveLeadHigh:                false,
	DefensiveHighFollow:              false,
	DefensiveSacrificeLead:           false,
}

// aggressiveNormalProfile is the Normal play profile for the Aggressive preset.
var aggressiveNormalProfile = PlayProfile{
	LeadHigh:                         false,
	PullTrumpWithRight:               true,
	PullTrumpMinCount:                5,
	CashAcesEarly:                    true,
	VoidHunting:                      true,
	DuckAndCover:                     true,
	OvertrumpPartner:                 true,
	DefensiveLeadRight:               true,
	DefensiveSaveRight:               false,
	DefensiveAvoidLeadingIntoHand:    false,
	DefensiveTrumpWithRightThreshold: 3,
	DefensiveTrumpLeadMin:            3,
	DefensiveLeadKing:                true,
	DefensiveLeadHigh:                true,
	DefensiveHighFollow:              true,
	DefensiveSacrificeLead:           true,
}

// Preset configs representing common player types.
var (
	Conservative = Config{
		Name:                  "Conservative",
		PartnerTricksEstimate: 1.0,
		BidPadding:            0,
		Bid5Threshold:         5.0,
		Bid6Threshold:         0,
		ScoreDeficitFactor:    0.0,
		ScoreSurplusFactor:    0.01,
		ScoreCloseoutBonus:    0.1,
		SeatPositionBias:      0.0,
		OvercallBias:          0.0,
		OpeningBidFactor:      0.9,
		RightBowerScore:       1.5,
		LeftBowerScore:        1.2,
		AceKingScore:          1.0,
		LowTrumpScore:         0.5,
		MajorSuitBonus:        0.0,
		TrumpLengthBonus:      0.0,
		PepperRequireBothRights: true,
		PepperMinLeftBowers:     1,
		PepperMinTrump:          7,
		PepperDiscardKeepAces:   true,
		// Conservative stays conservative regardless of situation.
		Normal:                conservativeProfile,
		Deficit:               conservativeProfile,
		Endgame:               conservativeProfile,
		DeficitRatio:          0.5,
		EndgameTrickThreshold: 0,
	}

	Aggressive = Config{
		Name:                  "Aggressive",
		PartnerTricksEstimate: 1.5,
		BidPadding:            0,
		Bid5Threshold:         4.2,
		Bid6Threshold:         5.5,
		ScoreDeficitFactor:    0.02,
		ScoreSurplusFactor:    0.0,
		ScoreCloseoutBonus:    0.3,
		SeatPositionBias:      0.05,
		OvercallBias:          0.2,
		OpeningBidFactor:      1.1,
		RightBowerScore:       1.5,
		LeftBowerScore:        1.2,
		AceKingScore:          1.0,
		LowTrumpScore:         0.5,
		MajorSuitBonus:        0.1,
		TrumpLengthBonus:      0.08,
		PepperRequireBothRights: true,
		PepperMinLeftBowers:     0,
		PepperMinTrump:          6,
		PepperDiscardKeepAces:   true,
		Normal: aggressiveNormalProfile,
		Deficit: PlayProfile{
			LeadHigh:                         true, // full aggression when behind
			PullTrumpWithRight:               true,
			PullTrumpMinCount:                5,
			CashAcesEarly:                    true,
			VoidHunting:                      true,
			DuckAndCover:                     true,
			OvertrumpPartner:                 true,
			DefensiveLeadRight:               true,
			DefensiveSaveRight:               false,
			DefensiveAvoidLeadingIntoHand:    false,
			DefensiveTrumpWithRightThreshold: 3,
			DefensiveTrumpLeadMin:            3,
			DefensiveLeadKing:                true,
			DefensiveLeadHigh:                true,
			DefensiveHighFollow:              true,
			DefensiveSacrificeLead:           true,
		},
		Endgame: PlayProfile{
			LeadHigh:                         true, // was EndgameLeadHigh=true
			PullTrumpWithRight:               true,
			PullTrumpMinCount:                5,
			CashAcesEarly:                    true,
			VoidHunting:                      true,
			DuckAndCover:                     false, // endgame skips duck-and-cover
			OvertrumpPartner:                 true,
			DefensiveLeadRight:               true,
			DefensiveSaveRight:               false,
			DefensiveAvoidLeadingIntoHand:    false,
			DefensiveTrumpWithRightThreshold: 3,
			DefensiveTrumpLeadMin:            3,
			DefensiveLeadKing:                true,
			DefensiveLeadHigh:                true,
			DefensiveHighFollow:              true,
			DefensiveSacrificeLead:           true,
		},
		DeficitRatio:          0.5,
		EndgameTrickThreshold: 2,
	}

	Balanced = Config{
		Name: "Balanced",
		// Cross-validated: seeds 123, 456, 789 — 34-param space, ~66M hands total.
		// PartnerTricksEstimate works in tandem with OpeningBidFactor: when opening the
		// auction (no prior bids), effective partner estimate = 1.16 * 0.73 ≈ 0.85,
		// keeping opening bids honest. Overcalling uses 1.16 directly.
		PartnerTricksEstimate: 0.63,  // v2 evolver seed 123 (was 1.16 — partner overestimated)
		BidPadding:            -1,    // confirmed all seeds
		Bid5Threshold:         4.92,  // v2 evolver seed 123 (was 4.65)
		Bid6Threshold:         6.19,  // v2 evolver seed 123 (was 5.5 — 6-bids were too frequent)
		ScoreDeficitFactor:    0.026, // v2 evolver seed 123 (was 0.011)
		ScoreSurplusFactor:    0.045, // v2 evolver seed 123 (was 0.012)
		ScoreCloseoutBonus:    0.33,  // v2 evolver seed 123 (was 0.34 — stable)
		SeatPositionBias:      0.09,  // v2 evolver seed 123 (was 0.025)
		OvercallBias:          0.05,  // v2 evolver seed 123 (was 0.20 — overcalling was too aggressive)
		OpeningBidFactor:      0.67,  // v2 evolver seed 123 (was 0.73)
		RightBowerScore:       1.36,  // v2 evolver seed 123 (was 2.0 hardcoded — badly overvalued)
		LeftBowerScore:        0.80,  // v2 evolver seed 123 (was 1.5 hardcoded)
		AceKingScore:          0.63,  // v2 evolver seed 123 (was 1.0 hardcoded)
		LowTrumpScore:         0.67,  // v2 evolver seed 123 (was 0.5 hardcoded)
		MajorSuitBonus:        0.28,  // v2 evolver seed 123 (was 0.19)
		TrumpLengthBonus:      0.14,  // v2 evolver seed 123 (was 0.13 — stable)

		PepperRequireBothRights: true,
		PepperMinLeftBowers:     0,    // v2 evolver seed 123
		PepperMinTrump:          6,    // v2 evolver seed 123
		PepperDiscardKeepAces:   false,

		// Play profiles: v2 evolver seed 123 (200 candidates, 20-gen refinement).
		Normal: PlayProfile{
			LeadHigh:                         false,
			PullTrumpWithRight:               true,
			PullTrumpMinCount:                4,
			CashAcesEarly:                    true,
			VoidHunting:                      false,
			DuckAndCover:                     true,
			OvertrumpPartner:                 false,
			DefensiveLeadRight:               true,
			DefensiveSaveRight:               false,
			DefensiveAvoidLeadingIntoHand:    true,
			DefensiveTrumpWithRightThreshold: 1,
			DefensiveTrumpLeadMin:            0,
			DefensiveLeadKing:                false,
			DefensiveLeadHigh:                false,
			DefensiveHighFollow:              false,
			DefensiveSacrificeLead:           false,
		},
		Deficit: PlayProfile{
			LeadHigh:                         true,
			PullTrumpWithRight:               true,
			PullTrumpMinCount:                4,
			CashAcesEarly:                    true,
			VoidHunting:                      true,
			DuckAndCover:                     true,
			OvertrumpPartner:                 false,
			DefensiveLeadRight:               true,
			DefensiveSaveRight:               false,
			DefensiveAvoidLeadingIntoHand:    true,
			DefensiveTrumpWithRightThreshold: 1,
			DefensiveTrumpLeadMin:            4,
			DefensiveLeadKing:                false,
			DefensiveLeadHigh:                false,
			DefensiveHighFollow:              true,
			DefensiveSacrificeLead:           false,
		},
		Endgame: PlayProfile{
			LeadHigh:                         false,
			PullTrumpWithRight:               true,
			PullTrumpMinCount:                7,
			CashAcesEarly:                    false,
			VoidHunting:                      true,
			DuckAndCover:                     true,
			OvertrumpPartner:                 true,
			DefensiveLeadRight:               false,
			DefensiveSaveRight:               true,
			DefensiveAvoidLeadingIntoHand:    true,
			DefensiveTrumpWithRightThreshold: 0,
			DefensiveTrumpLeadMin:            1,
			DefensiveLeadKing:                false,
			DefensiveLeadHigh:                true,
			DefensiveHighFollow:              false,
			DefensiveSacrificeLead:           false,
		},
		DeficitRatio:          0.73, // v2 evolver seed 123 (was 0.50)
		EndgameTrickThreshold: 1,    // v2 evolver seed 123 (stable)
	}

	// BalancedV1 is the previous Balanced preset, cross-validated across seeds 42/99/777
	// using the 22-param space. Kept for head-to-head benchmarking against Balanced.
	BalancedV1 = Config{
		Name:                  "BalancedV1",
		PartnerTricksEstimate: 1.97,
		BidPadding:            -1,
		Bid5Threshold:         4.50,
		Bid6Threshold:         0,
		ScoreDeficitFactor:    0.035,
		ScoreSurplusFactor:    0.030,
		ScoreCloseoutBonus:    0.0,
		SeatPositionBias:      0.05,
		OvercallBias:          0.25,
		OpeningBidFactor:      1.0,
		RightBowerScore:       1.5,
		LeftBowerScore:        1.2,
		AceKingScore:          1.0,
		LowTrumpScore:         0.5,
		MajorSuitBonus:        0.21,
		TrumpLengthBonus:      0.0,
		PepperRequireBothRights: true,
		PepperMinLeftBowers:     0,
		PepperMinTrump:          6,
		PepperDiscardKeepAces:   false,
		// V1 had TrickDeficitAggression=false, EndgameTrickThreshold=0.
		// All three profiles are the same: no situational adaptation.
		Normal: PlayProfile{
			LeadHigh:                         false,
			PullTrumpWithRight:               true,
			PullTrumpMinCount:                6,
			CashAcesEarly:                    false,
			VoidHunting:                      false,
			DuckAndCover:                     true,
			OvertrumpPartner:                 false,
			DefensiveLeadRight:               false,
			DefensiveSaveRight:               false,
			DefensiveAvoidLeadingIntoHand:    false,
			DefensiveTrumpWithRightThreshold: 0,
			DefensiveTrumpLeadMin:            0,
			DefensiveLeadKing:                false,
			DefensiveLeadHigh:                false,
			DefensiveHighFollow:              false,
			DefensiveSacrificeLead:           false,
		},
		Deficit: PlayProfile{
			LeadHigh:                         false, // V1 had no deficit aggression
			PullTrumpWithRight:               true,
			PullTrumpMinCount:                6,
			CashAcesEarly:                    false,
			VoidHunting:                      false,
			DuckAndCover:                     true,
			OvertrumpPartner:                 false,
			DefensiveLeadRight:               false,
			DefensiveSaveRight:               false,
			DefensiveAvoidLeadingIntoHand:    false,
			DefensiveTrumpWithRightThreshold: 0,
			DefensiveTrumpLeadMin:            0,
			DefensiveLeadKing:                false,
			DefensiveLeadHigh:                false,
			DefensiveHighFollow:              false,
			DefensiveSacrificeLead:           false,
		},
		Endgame: PlayProfile{
			LeadHigh:                         false,
			PullTrumpWithRight:               true,
			PullTrumpMinCount:                6,
			CashAcesEarly:                    false,
			VoidHunting:                      false,
			DuckAndCover:                     true,
			OvertrumpPartner:                 false,
			DefensiveLeadRight:               false,
			DefensiveSaveRight:               false,
			DefensiveAvoidLeadingIntoHand:    false,
			DefensiveTrumpWithRightThreshold: 0,
			DefensiveTrumpLeadMin:            0,
			DefensiveLeadKing:                false,
			DefensiveLeadHigh:                false,
			DefensiveHighFollow:              false,
			DefensiveSacrificeLead:           false,
		},
		DeficitRatio:          0.5,
		EndgameTrickThreshold: 0,
	}
)

// StandardStrategy implements game.Strategy using a Config.
type StandardStrategy struct {
	cfg Config
}

func NewStandard(cfg Config) *StandardStrategy {
	return &StandardStrategy{cfg: cfg}
}

// --- Bidding ---

func (s *StandardStrategy) Bid(seat int, state game.BidState) int {
	_, eval := s.bestTrumpWithBias(state.Hand)

	partnerEst := s.cfg.PartnerTricksEstimate

	// Scale partner estimate based on opening (no prior bids) vs overcalling.
	if state.CurrentHigh == 0 {
		partnerEst *= s.cfg.OpeningBidFactor
	} else {
		// An opponent (or partner) holds the current high bid.
		// Since we can't distinguish from BidState alone, add OvercallBias
		// whenever there is a bid to fight for.
		partnerEst += s.cfg.OvercallBias
	}

	total := eval + partnerEst

	// Seat position bonus: later seats can open slightly wider.
	bidOrder := (seat - state.DealerSeat + 6) % 6
	total += float64(bidOrder) * s.cfg.SeatPositionBias

	// Score adjustments.
	myTeam := game.TeamOf(seat)
	myScore := state.Scores[myTeam]
	theirScore := state.Scores[1-myTeam]

	deficit := float64(theirScore - myScore)
	if deficit > 0 {
		total += deficit * s.cfg.ScoreDeficitFactor
	} else {
		surplus := float64(myScore - theirScore)
		total -= surplus * s.cfg.ScoreSurplusFactor
	}

	if myScore >= 48 || theirScore >= 48 {
		total += s.cfg.ScoreCloseoutBonus
	}

	// Convert estimate to a bid value.
	bid := int(total) + s.cfg.BidPadding
	if bid < 4 {
		bid = 4
	}

	// Apply jump bid thresholds.
	if s.cfg.Bid6Threshold > 0 && total >= s.cfg.Bid6Threshold && bid < 6 {
		bid = 6
	} else if s.cfg.Bid5Threshold > 0 && total >= s.cfg.Bid5Threshold && bid < 5 {
		bid = 5
	}

	if bid < 4 {
		bid = 4
	}

	// Cannot beat the current high bid — pass.
	if state.CurrentHigh > 0 && bid <= state.CurrentHigh {
		return 0 // pass
	}

	// Check pepper eligibility before committing to any bid.
	if s.canCallPepper(state.Hand) {
		return game.PepperBid
	}

	if bid > 8 {
		bid = 8
	}
	return bid
}

func (s *StandardStrategy) canCallPepper(hand []card.Card) bool {
	for suit := card.Suit(0); suit < 4; suit++ {
		rights, lefts, total := 0, 0, 0
		for _, c := range hand {
			tr := card.TrumpRank(c, suit)
			if tr < 0 {
				continue
			}
			total++
			if card.IsRightBower(c, suit) {
				rights++
			} else if card.IsLeftBower(c, suit) {
				lefts++
			}
		}
		if s.cfg.PepperRequireBothRights && rights < 2 {
			continue
		}
		if lefts < s.cfg.PepperMinLeftBowers {
			continue
		}
		if total >= s.cfg.PepperMinTrump {
			return true
		}
	}
	return false
}

// --- Trump selection ---

func (s *StandardStrategy) ChooseTrump(seat int, hand []card.Card) card.Suit {
	best, _ := s.bestTrumpWithBias(hand)
	return best
}

func (s *StandardStrategy) bestTrumpWithBias(hand []card.Card) (card.Suit, float64) {
	bestSuit := card.Spades
	bestScore := -1.0
	for suit := card.Suit(0); suit < 4; suit++ {
		sc := s.suitScore(hand, suit) + rand.Float64()*0.01
		if sc > bestScore {
			bestScore = sc
			bestSuit = suit
		}
	}
	return bestSuit, bestScore
}

func (s *StandardStrategy) suitScore(hand []card.Card, trump card.Suit) float64 {
	var trumpCards []card.Card
	for _, c := range hand {
		if card.TrumpRank(c, trump) >= 0 {
			trumpCards = append(trumpCards, c)
		}
	}
	score := 0.0
	for _, c := range trumpCards {
		tr := card.TrumpRank(c, trump)
		switch {
		case tr >= 12: // right bower
			score += s.cfg.RightBowerScore
		case tr >= 11: // left bower
			score += s.cfg.LeftBowerScore
		case tr >= 9: // ace, king
			score += s.cfg.AceKingScore
		default: // queen, jack (non-bower), ten, nine
			score += s.cfg.LowTrumpScore
		}
	}
	// Length bonus for trump beyond the base 5.
	extra := len(trumpCards) - 5
	if extra > 0 {
		score += float64(extra) * s.cfg.TrumpLengthBonus
	}
	// Major suit bonus.
	if trump == card.Spades || trump == card.Hearts {
		score += s.cfg.MajorSuitBonus
	}
	return score
}

// --- Play dispatch ---

func (s *StandardStrategy) Play(seat int, validPlays []card.Card, state game.TrickState) card.Card {
	trump := state.Trump
	isPepper := state.BidAmount == game.PepperBid

	if isPepper && seat == state.BidderSeat {
		if len(state.Trick.Cards) == 0 {
			return s.pepperCallerLead(validPlays, trump)
		}
		return s.pepperCallerFollow(validPlays, state, trump)
	}

	if isPepper {
		if len(state.Trick.Cards) == 0 {
			return s.pepperOpponentLead(validPlays, trump)
		}
		return s.pepperOpponentFollow(validPlays, state, trump)
	}

	if len(state.Trick.Cards) == 0 {
		return s.chooseLead(seat, validPlays, state, trump)
	}
	return s.chooseFollow(seat, validPlays, state, trump)
}

// selectProfile returns the appropriate PlayProfile for the current game state.
// Endgame takes priority over Deficit when both conditions are satisfied.
func (s *StandardStrategy) selectProfile(state game.TrickState) *PlayProfile {
	tricksLeft := 8 - state.TrickNumber

	// Endgame check (highest priority).
	if s.cfg.EndgameTrickThreshold > 0 && tricksLeft <= s.cfg.EndgameTrickThreshold {
		return &s.cfg.Endgame
	}

	// Deficit check: only meaningful when bidding team has a real trick target.
	if state.BidAmount > 0 && tricksLeft > 0 {
		bidTeamTricks := 0
		for si := 0; si < 6; si++ {
			if game.TeamOf(si) == game.TeamOf(state.BidderSeat) {
				bidTeamTricks += state.TricksTaken[si]
			}
		}
		tricksNeeded := state.BidAmount - bidTeamTricks
		if tricksNeeded < 0 {
			tricksNeeded = 0
		}
		if float64(tricksNeeded) > s.cfg.DeficitRatio*float64(tricksLeft) {
			return &s.cfg.Deficit
		}
	}

	return &s.cfg.Normal
}

func (s *StandardStrategy) chooseLead(seat int, hand []card.Card, state game.TrickState, trump card.Suit) card.Card {
	isBiddingTeam := game.TeamOf(seat) == game.TeamOf(state.BidderSeat)
	trumpCards := filterTrump(hand, trump)
	p := s.selectProfile(state)

	// Universal override: lead highest available card regardless of role.
	if p.LeadHigh {
		if isBiddingTeam {
			if len(trumpCards) > 0 {
				return highestTrump(trumpCards, trump)
			}
			if c, ok := highestOffSuit(hand, trump); ok {
				return c
			}
		} else {
			if c, ok := highestOffSuit(hand, trump); ok {
				return c
			}
			if len(trumpCards) > 0 {
				return highestTrump(trumpCards, trump)
			}
		}
		return hand[0]
	}

	if isBiddingTeam {
		// Bidding team lead priority:
		// 1. Right bower — pull trump with your guaranteed top card.
		// 2. Cash aces early — before continuing trump extraction.
		// 3. Keep pulling trump while opponents appear to hold trump.
		// 4. Off-suit ace.
		// 5. Left bower — only if we also have the right.
		// 6. Short suit / void hunting.
		// 7. Lowest throwaway.

		hasRight := false
		hasLeft := false
		for _, c := range trumpCards {
			if card.IsRightBower(c, trump) {
				hasRight = true
			}
			if card.IsLeftBower(c, trump) {
				hasLeft = true
			}
		}

		// 1. Right bower.
		if p.PullTrumpWithRight && hasRight {
			for _, c := range trumpCards {
				if card.IsRightBower(c, trump) {
					return c
				}
			}
		}

		// 2. Cash aces early before continuing trump extraction.
		if p.CashAcesEarly {
			if c, ok := offSuitAce(hand, trump); ok {
				return c
			}
		}

		// 3. Keep pulling trump while opponents appear to still have trump.
		if len(trumpCards) > 0 {
			trumpRemaining := 14
			if state.History != nil {
				trumpRemaining = state.History.TrumpRemaining(trump)
			}
			opponentTrumpEst := trumpRemaining - len(trumpCards) - 1
			if opponentTrumpEst > 0 && len(trumpCards) >= p.PullTrumpMinCount {
				return lowestTrump(trumpCards, trump)
			}
		}

		// 4. Off-suit ace.
		if c, ok := offSuitAce(hand, trump); ok {
			return c
		}

		// 5. Left bower — only if we also have the right.
		if hasLeft && hasRight {
			for _, c := range trumpCards {
				if card.IsLeftBower(c, trump) {
					return c
				}
			}
		}

		// 6. Void hunting.
		if p.VoidHunting {
			if c, ok := shortSuitLead(hand, trump); ok {
				return c
			}
		}

		// 7. Lowest non-trump throwaway.
		if c, ok := lowestNonTrump(hand, trump); ok {
			return c
		}
		if len(trumpCards) > 0 {
			return lowestTrump(trumpCards, trump)
		}
		return hand[0]
	}

	// ── Defensive lead ────────────────────────────────────────────────────────

	// Determine if the bidder plays immediately after us (we'd be "leading into" them).
	// Position 1 means bidder plays next — a trump lead hands them exactly what they want.
	bidderRelPos := (state.BidderSeat - seat + 6) % 6
	avoidTrump := p.DefensiveAvoidLeadingIntoHand && bidderRelPos == 1

	// Scan trump holding for right bower.
	hasRightDefense := false
	for _, c := range trumpCards {
		if card.IsRightBower(c, trump) {
			hasRightDefense = true
			break
		}
	}

	// DefensiveLeadRight: cash right bower immediately (overridden by DefensiveSaveRight
	// and DefensiveAvoidLeadingIntoHand).
	if p.DefensiveLeadRight && !p.DefensiveSaveRight && hasRightDefense && !avoidTrump {
		for _, c := range trumpCards {
			if card.IsRightBower(c, trump) {
				return c
			}
		}
	}

	// DefensiveTrumpWithRightThreshold: lead trump when holding right + enough total trump.
	if hasRightDefense && !avoidTrump && !p.DefensiveSaveRight {
		if len(trumpCards) >= p.DefensiveTrumpWithRightThreshold {
			return highestTrump(trumpCards, trump)
		}
	}

	// DefensiveTrumpLeadMin: lead trump with a dominant holding regardless of right bower.
	if p.DefensiveTrumpLeadMin > 0 && len(trumpCards) >= p.DefensiveTrumpLeadMin && !avoidTrump {
		return highestTrump(trumpCards, trump)
	}

	// Lead card by rank preference.
	if p.DefensiveLeadHigh {
		if c, ok := highestOffSuit(hand, trump); ok {
			return c
		}
	} else {
		if c, ok := offSuitAce(hand, trump); ok {
			return c
		}
		if p.DefensiveLeadKing {
			if c, ok := offSuitKing(hand, trump); ok {
				return c
			}
		}
	}

	// Void hunting.
	if p.VoidHunting {
		if c, ok := shortSuitLead(hand, trump); ok {
			return c
		}
	}

	// Safe low card — preserve trump and high cards.
	if c, ok := lowestNonTrump(hand, trump); ok {
		return c
	}
	if len(trumpCards) > 0 {
		return lowestTrump(trumpCards, trump)
	}
	return hand[0]
}

func (s *StandardStrategy) chooseFollow(seat int, hand []card.Card, state game.TrickState, trump card.Suit) card.Card {
	trick := state.Trick
	currentWinner := trick.Winner()
	partnerWinning := game.TeamOf(currentWinner) == game.TeamOf(seat)
	isBiddingTeam := game.TeamOf(seat) == game.TeamOf(state.BidderSeat)
	p := s.selectProfile(state)

	// Duck and cover: if partner is winning, play lowest legal card.
	if p.DuckAndCover && partnerWinning {
		// OvertrumpPartner exception: if partner is winning with trump and we have higher
		// trump, over-trump to consolidate the bidding team's trump holdings.
		if p.OvertrumpPartner && isBiddingTeam {
			for _, pc := range trick.Cards {
				if pc.Seat == currentWinner {
					winTR := card.TrumpRank(pc.Card, trump)
					if winTR >= 0 {
						trumpCards := filterTrump(hand, trump)
						for _, c := range trumpCards {
							if card.TrumpRank(c, trump) > winTR {
								return highestTrump(trumpCards, trump)
							}
						}
					}
					break
				}
			}
		}
		return lowestCard(hand, trump)
	}

	// Defensive high follow: play highest winner to maximize euchre chances.
	if !isBiddingTeam && p.DefensiveHighFollow && !partnerWinning {
		if winning := highestWinner(hand, trick, trump); winning != nil {
			return *winning
		}
		return lowestCard(hand, trump)
	}

	// Try to win the trick with the lowest card that beats current winner.
	if winning := lowestWinner(hand, trick, trump); winning != nil {
		return *winning
	}

	// Can't win: sacrifice with highest non-trump to pressure the bidder, or throw off lowest.
	if !isBiddingTeam && p.DefensiveSacrificeLead {
		if c, ok := highestOffSuit(hand, trump); ok {
			return c
		}
	}

	return lowestCard(hand, trump)
}

// --- Pepper play ---

func (s *StandardStrategy) pepperCallerLead(hand []card.Card, trump card.Suit) card.Card {
	trumpCards := filterTrump(hand, trump)
	if len(trumpCards) > 0 {
		return highestTrump(trumpCards, trump)
	}
	if c, ok := highestOffSuit(hand, trump); ok {
		return c
	}
	return hand[0]
}

func (s *StandardStrategy) pepperCallerFollow(hand []card.Card, state game.TrickState, trump card.Suit) card.Card {
	trick := state.Trick
	winSeat := trick.Winner()
	var winCard card.Card
	for _, pc := range trick.Cards {
		if pc.Seat == winSeat {
			winCard = pc.Card
			break
		}
	}
	winTR := card.TrumpRank(winCard, trump)
	ledSuit := trick.LedSuit()

	var winners []card.Card
	for _, c := range hand {
		tr := card.TrumpRank(c, trump)
		effSuit := card.EffectiveSuit(c, trump)
		var beats bool
		if winTR >= 0 {
			beats = tr > winTR
		} else {
			if tr >= 0 {
				beats = true
			} else if effSuit == ledSuit && card.NonTrumpRank(c) > card.NonTrumpRank(winCard) {
				beats = true
			}
		}
		if beats {
			winners = append(winners, c)
		}
	}
	if len(winners) == 0 {
		return lowestCard(hand, trump)
	}
	best := winners[0]
	for _, c := range winners[1:] {
		cTR := card.TrumpRank(c, trump)
		bTR := card.TrumpRank(best, trump)
		if cTR >= 0 && bTR >= 0 {
			if cTR > bTR {
				best = c
			}
		} else if cTR < 0 && bTR < 0 {
			if card.NonTrumpRank(c) > card.NonTrumpRank(best) {
				best = c
			}
		} else if cTR >= 0 {
			best = c
		}
	}
	return best
}

func (s *StandardStrategy) pepperOpponentLead(hand []card.Card, trump card.Suit) card.Card {
	if c, ok := offSuitAce(hand, trump); ok {
		return c
	}
	if c, ok := highestOffSuit(hand, trump); ok {
		return c
	}
	trumpCards := filterTrump(hand, trump)
	if len(trumpCards) > 0 {
		return highestTrump(trumpCards, trump)
	}
	return hand[0]
}

func (s *StandardStrategy) pepperOpponentFollow(hand []card.Card, state game.TrickState, trump card.Suit) card.Card {
	trick := state.Trick
	winSeat := trick.Winner()
	var winCard card.Card
	for _, pc := range trick.Cards {
		if pc.Seat == winSeat {
			winCard = pc.Card
			break
		}
	}
	winTR := card.TrumpRank(winCard, trump)
	ledSuit := trick.LedSuit()

	var winners []card.Card
	for _, c := range hand {
		tr := card.TrumpRank(c, trump)
		effSuit := card.EffectiveSuit(c, trump)
		var beats bool
		if winTR >= 0 {
			beats = tr > winTR
		} else {
			if tr >= 0 {
				beats = true
			} else if effSuit == ledSuit && card.NonTrumpRank(c) > card.NonTrumpRank(winCard) {
				beats = true
			}
		}
		if beats {
			winners = append(winners, c)
		}
	}
	if len(winners) == 0 {
		return lowestCard(hand, trump)
	}
	best := winners[0]
	for _, c := range winners[1:] {
		cTR := card.TrumpRank(c, trump)
		bTR := card.TrumpRank(best, trump)
		if cTR >= 0 && bTR >= 0 {
			if cTR > bTR {
				best = c
			}
		} else if cTR < 0 && bTR < 0 {
			if card.NonTrumpRank(c) > card.NonTrumpRank(best) {
				best = c
			}
		} else if cTR >= 0 {
			best = c
		}
	}
	return best
}

// --- Pepper exchange ---

func (s *StandardStrategy) GivePepper(seat int, hand []card.Card, trump card.Suit) card.Card {
	best, found := game.BestTrump(hand, trump)
	if found {
		return best
	}
	return hand[0]
}

func (s *StandardStrategy) PepperDiscard(seat int, hand []card.Card, trump card.Suit, received [2]card.Card) [2]card.Card {
	if s.cfg.PepperDiscardKeepAces {
		nonTrumpNonAce := filterNonTrumpNonAce(hand, trump)
		if len(nonTrumpNonAce) >= 2 {
			sorted := sortByRankAsc(nonTrumpNonAce)
			return [2]card.Card{sorted[0], sorted[1]}
		}
	}
	nonTrump := filterNonTrump(hand, trump)
	if len(nonTrump) >= 2 {
		sorted := sortByRankAsc(nonTrump)
		return [2]card.Card{sorted[0], sorted[1]}
	}
	sorted := sortByTrumpRankAsc(hand, trump)
	return [2]card.Card{sorted[0], sorted[1]}
}

// --- Helpers ---

func filterTrump(hand []card.Card, trump card.Suit) []card.Card {
	var out []card.Card
	for _, c := range hand {
		if card.TrumpRank(c, trump) >= 0 {
			out = append(out, c)
		}
	}
	return out
}

func filterNonTrump(hand []card.Card, trump card.Suit) []card.Card {
	var out []card.Card
	for _, c := range hand {
		if card.TrumpRank(c, trump) < 0 {
			out = append(out, c)
		}
	}
	return out
}

func filterNonTrumpNonAce(hand []card.Card, trump card.Suit) []card.Card {
	var out []card.Card
	for _, c := range hand {
		if card.TrumpRank(c, trump) < 0 && c.Rank != card.Ace {
			out = append(out, c)
		}
	}
	return out
}

func highestTrump(trumpCards []card.Card, trump card.Suit) card.Card {
	best := trumpCards[0]
	for _, c := range trumpCards[1:] {
		if card.TrumpRank(c, trump) > card.TrumpRank(best, trump) {
			best = c
		}
	}
	return best
}

func highestOffSuit(hand []card.Card, trump card.Suit) (card.Card, bool) {
	var best *card.Card
	for i, c := range hand {
		if card.TrumpRank(c, trump) >= 0 {
			continue
		}
		if best == nil || card.NonTrumpRank(c) > card.NonTrumpRank(*best) {
			best = &hand[i]
		}
	}
	if best == nil {
		return card.Card{}, false
	}
	return *best, true
}

func lowestCard(hand []card.Card, trump card.Suit) card.Card {
	best := hand[0]
	bestTR := card.TrumpRank(best, trump)
	bestNR := card.NonTrumpRank(best)
	for _, c := range hand[1:] {
		tr := card.TrumpRank(c, trump)
		nr := card.NonTrumpRank(c)
		if bestTR >= 0 && tr < 0 {
			best = c
			bestTR = tr
			bestNR = nr
		} else if bestTR < 0 && tr >= 0 {
			// keep current non-trump
		} else if tr >= 0 && bestTR >= 0 {
			if tr < bestTR {
				best = c
				bestTR = tr
				bestNR = nr
			}
		} else {
			if nr < bestNR {
				best = c
				bestNR = nr
			}
		}
	}
	return best
}

func lowestWinner(hand []card.Card, trick *game.Trick, trump card.Suit) *card.Card {
	winSeat := trick.Winner()
	var winCard card.Card
	for _, pc := range trick.Cards {
		if pc.Seat == winSeat {
			winCard = pc.Card
			break
		}
	}
	winTR := card.TrumpRank(winCard, trump)
	winNR := card.NonTrumpRank(winCard)
	ledSuit := trick.LedSuit()

	var candidates []card.Card
	for _, c := range hand {
		tr := card.TrumpRank(c, trump)
		effSuit := card.EffectiveSuit(c, trump)
		beats := false
		if winTR >= 0 {
			beats = tr > winTR
		} else {
			if tr >= 0 {
				beats = true
			} else if effSuit == ledSuit && card.NonTrumpRank(c) > winNR {
				beats = true
			}
		}
		if beats {
			candidates = append(candidates, c)
		}
	}
	if len(candidates) == 0 {
		return nil
	}
	best := candidates[0]
	for _, c := range candidates[1:] {
		cTR := card.TrumpRank(c, trump)
		bTR := card.TrumpRank(best, trump)
		if cTR >= 0 && bTR >= 0 {
			if cTR < bTR {
				best = c
			}
		} else if cTR < 0 && bTR < 0 {
			if card.NonTrumpRank(c) < card.NonTrumpRank(best) {
				best = c
			}
		}
	}
	return &best
}

func highestWinner(hand []card.Card, trick *game.Trick, trump card.Suit) *card.Card {
	winSeat := trick.Winner()
	var winCard card.Card
	for _, pc := range trick.Cards {
		if pc.Seat == winSeat {
			winCard = pc.Card
			break
		}
	}
	winTR := card.TrumpRank(winCard, trump)
	winNR := card.NonTrumpRank(winCard)
	ledSuit := trick.LedSuit()

	var candidates []card.Card
	for _, c := range hand {
		tr := card.TrumpRank(c, trump)
		effSuit := card.EffectiveSuit(c, trump)
		beats := false
		if winTR >= 0 {
			beats = tr > winTR
		} else {
			if tr >= 0 {
				beats = true
			} else if effSuit == ledSuit && card.NonTrumpRank(c) > winNR {
				beats = true
			}
		}
		if beats {
			candidates = append(candidates, c)
		}
	}
	if len(candidates) == 0 {
		return nil
	}
	best := candidates[0]
	for _, c := range candidates[1:] {
		cTR := card.TrumpRank(c, trump)
		bTR := card.TrumpRank(best, trump)
		if cTR >= 0 && bTR >= 0 {
			if cTR > bTR {
				best = c
			}
		} else if cTR < 0 && bTR < 0 {
			if card.NonTrumpRank(c) > card.NonTrumpRank(best) {
				best = c
			}
		} else if cTR >= 0 {
			best = c
		}
	}
	return &best
}

func offSuitAce(hand []card.Card, trump card.Suit) (card.Card, bool) {
	for _, c := range hand {
		if c.Rank == card.Ace && card.TrumpRank(c, trump) < 0 {
			return c, true
		}
	}
	return card.Card{}, false
}

func offSuitKing(hand []card.Card, trump card.Suit) (card.Card, bool) {
	for _, c := range hand {
		if c.Rank == card.King && card.TrumpRank(c, trump) < 0 {
			return c, true
		}
	}
	return card.Card{}, false
}

func lowestNonTrump(hand []card.Card, trump card.Suit) (card.Card, bool) {
	var best *card.Card
	for i, c := range hand {
		if card.TrumpRank(c, trump) >= 0 {
			continue
		}
		if best == nil || card.NonTrumpRank(c) < card.NonTrumpRank(*best) {
			best = &hand[i]
		}
	}
	if best == nil {
		return card.Card{}, false
	}
	return *best, true
}

func lowestTrump(trumpCards []card.Card, trump card.Suit) card.Card {
	best := trumpCards[0]
	for _, c := range trumpCards[1:] {
		if card.TrumpRank(c, trump) < card.TrumpRank(best, trump) {
			best = c
		}
	}
	return best
}

func shortSuitLead(hand []card.Card, trump card.Suit) (card.Card, bool) {
	suitCounts := map[card.Suit][]card.Card{}
	for _, c := range hand {
		eff := card.EffectiveSuit(c, trump)
		if eff != trump {
			suitCounts[eff] = append(suitCounts[eff], c)
		}
	}
	var bestSuit card.Suit
	bestLen := 99
	found := false
	for s, cards := range suitCounts {
		if len(cards) > 0 && len(cards) < bestLen {
			bestLen = len(cards)
			bestSuit = s
			found = true
		}
	}
	if !found {
		return card.Card{}, false
	}
	cards := suitCounts[bestSuit]
	low := cards[0]
	for _, c := range cards[1:] {
		if card.NonTrumpRank(c) < card.NonTrumpRank(low) {
			low = c
		}
	}
	return low, true
}

func sortByRankAsc(hand []card.Card) []card.Card {
	cp := make([]card.Card, len(hand))
	copy(cp, hand)
	for i := 1; i < len(cp); i++ {
		for j := i; j > 0 && card.NonTrumpRank(cp[j]) < card.NonTrumpRank(cp[j-1]); j-- {
			cp[j], cp[j-1] = cp[j-1], cp[j]
		}
	}
	return cp
}

func sortByTrumpRankAsc(hand []card.Card, trump card.Suit) []card.Card {
	cp := make([]card.Card, len(hand))
	copy(cp, hand)
	for i := 1; i < len(cp); i++ {
		for j := i; j > 0 && card.TrumpRank(cp[j], trump) < card.TrumpRank(cp[j-1], trump); j-- {
			cp[j], cp[j-1] = cp[j-1], cp[j]
		}
	}
	return cp
}
