package ml

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/max/pepper/internal/strategy"
)

// PlayProfileGenome is the evolvable form of strategy.PlayProfile.
type PlayProfileGenome struct {
	LeadHigh                         bool
	PullTrumpWithRight               bool
	PullTrumpMinCount                int
	CashAcesEarly                    bool
	VoidHunting                      bool
	DuckAndCover                     bool
	OvertrumpPartner                 bool
	DefensiveLeadRight               bool
	DefensiveSaveRight               bool
	DefensiveAvoidLeadingIntoHand    bool
	DefensiveTrumpWithRightThreshold int
	DefensiveTrumpLeadMin            int
	DefensiveLeadKing                bool
	DefensiveLeadHigh                bool
	DefensiveHighFollow              bool
	DefensiveSacrificeLead           bool
}

func (p PlayProfileGenome) toProfile() strategy.PlayProfile {
	return strategy.PlayProfile{
		LeadHigh:                         p.LeadHigh,
		PullTrumpWithRight:               p.PullTrumpWithRight,
		PullTrumpMinCount:                p.PullTrumpMinCount,
		CashAcesEarly:                    p.CashAcesEarly,
		VoidHunting:                      p.VoidHunting,
		DuckAndCover:                     p.DuckAndCover,
		OvertrumpPartner:                 p.OvertrumpPartner,
		DefensiveLeadRight:               p.DefensiveLeadRight,
		DefensiveSaveRight:               p.DefensiveSaveRight,
		DefensiveAvoidLeadingIntoHand:    p.DefensiveAvoidLeadingIntoHand,
		DefensiveTrumpWithRightThreshold: p.DefensiveTrumpWithRightThreshold,
		DefensiveTrumpLeadMin:            p.DefensiveTrumpLeadMin,
		DefensiveLeadKing:                p.DefensiveLeadKing,
		DefensiveLeadHigh:                p.DefensiveLeadHigh,
		DefensiveHighFollow:              p.DefensiveHighFollow,
		DefensiveSacrificeLead:           p.DefensiveSacrificeLead,
	}
}

// Genome represents a strategy config as an evolvable parameter vector.
// Bidding, trump selection, and pepper parameters are flat fields.
// Card-play parameters are split across three situational profiles
// (Normal, Deficit, Endgame) selected at runtime by the strategy.
type Genome struct {
	// Bidding
	PartnerTricksEstimate float64
	BidPadding            int
	Bid5Threshold         float64
	Bid6Threshold         float64
	ScoreDeficitFactor    float64
	ScoreSurplusFactor    float64
	ScoreCloseoutBonus    float64
	SeatPositionBias      float64
	OvercallBias          float64
	OpeningBidFactor      float64

	// Trump suit scoring weights
	RightBowerScore float64
	LeftBowerScore  float64
	AceKingScore    float64
	LowTrumpScore   float64

	// Trump selection
	MajorSuitBonus   float64
	TrumpLengthBonus float64

	// Pepper calling
	PepperRequireBothRights bool
	PepperMinLeftBowers     int
	PepperMinTrump          int
	PepperDiscardKeepAces   bool

	// Card play — situational profiles
	Normal  PlayProfileGenome
	Deficit PlayProfileGenome
	Endgame PlayProfileGenome

	// Profile selection thresholds
	DeficitRatio          float64 // 0.3–0.9
	EndgameTrickThreshold int     // 0–4
}

// ToConfig converts a Genome to a strategy.Config.
func (g Genome) ToConfig(name string) strategy.Config {
	return strategy.Config{
		Name:                    name,
		PartnerTricksEstimate:   g.PartnerTricksEstimate,
		BidPadding:              g.BidPadding,
		Bid5Threshold:           g.Bid5Threshold,
		Bid6Threshold:           g.Bid6Threshold,
		ScoreDeficitFactor:      g.ScoreDeficitFactor,
		ScoreSurplusFactor:      g.ScoreSurplusFactor,
		ScoreCloseoutBonus:      g.ScoreCloseoutBonus,
		SeatPositionBias:        g.SeatPositionBias,
		OvercallBias:            g.OvercallBias,
		OpeningBidFactor:        g.OpeningBidFactor,
		RightBowerScore:         g.RightBowerScore,
		LeftBowerScore:          g.LeftBowerScore,
		AceKingScore:            g.AceKingScore,
		LowTrumpScore:           g.LowTrumpScore,
		MajorSuitBonus:          g.MajorSuitBonus,
		TrumpLengthBonus:        g.TrumpLengthBonus,
		PepperRequireBothRights: g.PepperRequireBothRights,
		PepperMinLeftBowers:     g.PepperMinLeftBowers,
		PepperMinTrump:          g.PepperMinTrump,
		PepperDiscardKeepAces:   g.PepperDiscardKeepAces,
		Normal:                  g.Normal.toProfile(),
		Deficit:                 g.Deficit.toProfile(),
		Endgame:                 g.Endgame.toProfile(),
		DeficitRatio:            g.DeficitRatio,
		EndgameTrickThreshold:   g.EndgameTrickThreshold,
	}
}

// profileString returns a compact representation of a PlayProfileGenome.
func profileString(p PlayProfileGenome) string {
	return fmt.Sprintf(
		"hi=%v pull(R=%v min=%d) cash=%v void=%v duck=%v otrump=%v "+
			"def(Rl=%v Rs=%v avoid=%v Rth=%d min=%d king=%v hi=%v hiFol=%v sac=%v)",
		p.LeadHigh,
		p.PullTrumpWithRight, p.PullTrumpMinCount,
		p.CashAcesEarly, p.VoidHunting,
		p.DuckAndCover, p.OvertrumpPartner,
		p.DefensiveLeadRight, p.DefensiveSaveRight, p.DefensiveAvoidLeadingIntoHand,
		p.DefensiveTrumpWithRightThreshold, p.DefensiveTrumpLeadMin,
		p.DefensiveLeadKing, p.DefensiveLeadHigh, p.DefensiveHighFollow, p.DefensiveSacrificeLead,
	)
}

// String returns a compact human-readable description.
func (g Genome) String() string {
	return fmt.Sprintf(
		"partner=%.2f pad=%d b5=%.2f b6=%.2f def=%.3f sur=%.3f cls=%.2f pos=%.2f ovr=%.2f opn=%.2f "+
			"scores(R=%.2f L=%.2f AK=%.2f lo=%.2f) maj=%.2f len=%.2f "+
			"pep(rr=%v ll>=%d tr>=%d ace=%v) "+
			"defRatio=%.2f endTh=%d\n"+
			"  Normal:  %s\n"+
			"  Deficit: %s\n"+
			"  Endgame: %s",
		g.PartnerTricksEstimate, g.BidPadding,
		g.Bid5Threshold, g.Bid6Threshold,
		g.ScoreDeficitFactor, g.ScoreSurplusFactor, g.ScoreCloseoutBonus,
		g.SeatPositionBias, g.OvercallBias, g.OpeningBidFactor,
		g.RightBowerScore, g.LeftBowerScore, g.AceKingScore, g.LowTrumpScore,
		g.MajorSuitBonus, g.TrumpLengthBonus,
		g.PepperRequireBothRights, g.PepperMinLeftBowers, g.PepperMinTrump, g.PepperDiscardKeepAces,
		g.DeficitRatio, g.EndgameTrickThreshold,
		profileString(g.Normal),
		profileString(g.Deficit),
		profileString(g.Endgame),
	)
}

// randomProfile generates a random PlayProfileGenome.
func randomProfile(rng *rand.Rand) PlayProfileGenome {
	return PlayProfileGenome{
		LeadHigh:                         rng.Float64() < 0.3,
		PullTrumpWithRight:               rng.Float64() < 0.8,
		PullTrumpMinCount:                rng.Intn(7),   // 0–6
		CashAcesEarly:                    rng.Float64() < 0.5,
		VoidHunting:                      rng.Float64() < 0.5,
		DuckAndCover:                     rng.Float64() < 0.75,
		OvertrumpPartner:                 rng.Float64() < 0.4,
		DefensiveLeadRight:               rng.Float64() < 0.5,
		DefensiveSaveRight:               rng.Float64() < 0.5,
		DefensiveAvoidLeadingIntoHand:    rng.Float64() < 0.5,
		DefensiveTrumpWithRightThreshold: rng.Intn(6),  // 0–5
		DefensiveTrumpLeadMin:            rng.Intn(6),  // 0–5
		DefensiveLeadKing:                rng.Float64() < 0.5,
		DefensiveLeadHigh:                rng.Float64() < 0.5,
		DefensiveHighFollow:              rng.Float64() < 0.5,
		DefensiveSacrificeLead:           rng.Float64() < 0.5,
	}
}

// RandomGenome generates a genome with uniformly random parameters.
func RandomGenome(rng *rand.Rand) Genome {
	return Genome{
		// Bidding
		PartnerTricksEstimate: 0.5 + rng.Float64()*2.0,  // 0.5–2.5
		BidPadding:            rng.Intn(3) - 1,           // -1, 0, 1
		Bid5Threshold:         4.0 + rng.Float64()*1.2,  // 4.0–5.2
		Bid6Threshold:         rng.Float64() * 6.2,       // 0 (off) or 5.0–6.2
		ScoreDeficitFactor:    rng.Float64() * 0.05,      // 0.0–0.05
		ScoreSurplusFactor:    rng.Float64() * 0.05,      // 0.0–0.05
		ScoreCloseoutBonus:    rng.Float64() * 0.5,       // 0.0–0.5
		SeatPositionBias:      rng.Float64() * 0.15,      // 0.0–0.15
		OvercallBias:          rng.Float64() * 0.5,       // 0.0–0.5
		OpeningBidFactor:      0.5 + rng.Float64()*1.0,  // 0.5–1.5

		// Trump suit scoring weights
		RightBowerScore: 1.0 + rng.Float64()*1.5, // 1.0–2.5
		LeftBowerScore:  0.8 + rng.Float64()*1.2, // 0.8–2.0
		AceKingScore:    0.5 + rng.Float64()*1.0, // 0.5–1.5
		LowTrumpScore:   0.1 + rng.Float64()*0.7, // 0.1–0.8

		// Trump selection
		MajorSuitBonus:   rng.Float64() * 0.3, // 0.0–0.3
		TrumpLengthBonus: rng.Float64() * 0.2, // 0.0–0.2

		// Pepper
		PepperRequireBothRights: rng.Float64() < 0.85,
		PepperMinLeftBowers:     rng.Intn(3),            // 0–2
		PepperMinTrump:          5 + rng.Intn(4),        // 5–8
		PepperDiscardKeepAces:   rng.Float64() < 0.5,

		// Card play — three independent profiles
		Normal:  randomProfile(rng),
		Deficit: randomProfile(rng),
		Endgame: randomProfile(rng),

		// Profile selection
		DeficitRatio:          0.3 + rng.Float64()*0.6, // 0.3–0.9
		EndgameTrickThreshold: rng.Intn(5),             // 0–4
	}
}

// mutateProfile returns a mutated copy of a PlayProfileGenome.
func mutateProfile(p PlayProfileGenome, rng *rand.Rand, strength float64) PlayProfileGenome {
	m := p
	flipBool := func(b bool, rate float64) bool {
		if rng.Float64() < rate*strength {
			return !b
		}
		return b
	}
	m.LeadHigh = flipBool(p.LeadHigh, 0.25)
	m.PullTrumpWithRight = flipBool(p.PullTrumpWithRight, 0.15)
	m.CashAcesEarly = flipBool(p.CashAcesEarly, 0.2)
	m.VoidHunting = flipBool(p.VoidHunting, 0.2)
	m.DuckAndCover = flipBool(p.DuckAndCover, 0.15)
	m.OvertrumpPartner = flipBool(p.OvertrumpPartner, 0.2)
	m.DefensiveLeadRight = flipBool(p.DefensiveLeadRight, 0.2)
	m.DefensiveSaveRight = flipBool(p.DefensiveSaveRight, 0.2)
	m.DefensiveAvoidLeadingIntoHand = flipBool(p.DefensiveAvoidLeadingIntoHand, 0.2)
	m.DefensiveLeadKing = flipBool(p.DefensiveLeadKing, 0.2)
	m.DefensiveLeadHigh = flipBool(p.DefensiveLeadHigh, 0.2)
	m.DefensiveHighFollow = flipBool(p.DefensiveHighFollow, 0.2)
	m.DefensiveSacrificeLead = flipBool(p.DefensiveSacrificeLead, 0.2)

	if rng.Float64() < 0.4*strength {
		m.PullTrumpMinCount = clampI(p.PullTrumpMinCount+randSign(rng), 0, 7)
	}
	if rng.Float64() < 0.4*strength {
		m.DefensiveTrumpWithRightThreshold = clampI(p.DefensiveTrumpWithRightThreshold+randSign(rng), 0, 5)
	}
	if rng.Float64() < 0.4*strength {
		m.DefensiveTrumpLeadMin = clampI(p.DefensiveTrumpLeadMin+randSign(rng), 0, 5)
	}
	return m
}

// Mutate returns a new genome with small random perturbations.
func (g Genome) Mutate(rng *rand.Rand, strength float64) Genome {
	m := g

	// Float params: Gaussian noise scaled by strength.
	m.PartnerTricksEstimate = clampF(g.PartnerTricksEstimate+rng.NormFloat64()*0.2*strength, 0.5, 2.5)
	m.Bid5Threshold = clampF(g.Bid5Threshold+rng.NormFloat64()*0.15*strength, 4.0, 5.2)
	m.Bid6Threshold = clampF(g.Bid6Threshold+rng.NormFloat64()*0.2*strength, 0.0, 6.2)
	m.ScoreDeficitFactor = clampF(g.ScoreDeficitFactor+rng.NormFloat64()*0.005*strength, 0.0, 0.05)
	m.ScoreSurplusFactor = clampF(g.ScoreSurplusFactor+rng.NormFloat64()*0.005*strength, 0.0, 0.05)
	m.ScoreCloseoutBonus = clampF(g.ScoreCloseoutBonus+rng.NormFloat64()*0.05*strength, 0.0, 0.5)
	m.SeatPositionBias = clampF(g.SeatPositionBias+rng.NormFloat64()*0.02*strength, 0.0, 0.15)
	m.OvercallBias = clampF(g.OvercallBias+rng.NormFloat64()*0.05*strength, 0.0, 0.5)
	m.OpeningBidFactor = clampF(g.OpeningBidFactor+rng.NormFloat64()*0.1*strength, 0.5, 1.5)
	m.RightBowerScore = clampF(g.RightBowerScore+rng.NormFloat64()*0.15*strength, 1.0, 2.5)
	m.LeftBowerScore = clampF(g.LeftBowerScore+rng.NormFloat64()*0.12*strength, 0.8, 2.0)
	m.AceKingScore = clampF(g.AceKingScore+rng.NormFloat64()*0.1*strength, 0.5, 1.5)
	m.LowTrumpScore = clampF(g.LowTrumpScore+rng.NormFloat64()*0.08*strength, 0.1, 0.8)
	m.MajorSuitBonus = clampF(g.MajorSuitBonus+rng.NormFloat64()*0.04*strength, 0.0, 0.3)
	m.TrumpLengthBonus = clampF(g.TrumpLengthBonus+rng.NormFloat64()*0.03*strength, 0.0, 0.2)
	m.DeficitRatio = clampF(g.DeficitRatio+rng.NormFloat64()*0.05*strength, 0.3, 0.9)

	// Int params: +/- 1 with probability proportional to strength.
	if rng.Float64() < 0.4*strength {
		m.BidPadding = clampI(g.BidPadding+randSign(rng), -1, 2)
	}
	if rng.Float64() < 0.4*strength {
		m.PepperMinLeftBowers = clampI(g.PepperMinLeftBowers+randSign(rng), 0, 2)
	}
	if rng.Float64() < 0.4*strength {
		m.PepperMinTrump = clampI(g.PepperMinTrump+randSign(rng), 4, 9)
	}
	if rng.Float64() < 0.4*strength {
		m.EndgameTrickThreshold = clampI(g.EndgameTrickThreshold+randSign(rng), 0, 4)
	}

	// Bool params: flip with low probability.
	flipBool := func(b bool, rate float64) bool {
		if rng.Float64() < rate*strength {
			return !b
		}
		return b
	}
	m.PepperRequireBothRights = flipBool(g.PepperRequireBothRights, 0.15)
	m.PepperDiscardKeepAces = flipBool(g.PepperDiscardKeepAces, 0.15)

	// Mutate each play profile independently.
	m.Normal = mutateProfile(g.Normal, rng, strength)
	m.Deficit = mutateProfile(g.Deficit, rng, strength)
	m.Endgame = mutateProfile(g.Endgame, rng, strength)

	return m
}

// crossoverProfile returns a new profile by randomly selecting fields from p or other.
func crossoverProfile(p, other PlayProfileGenome, rng *rand.Rand) PlayProfileGenome {
	pick := func() bool { return rng.Float64() < 0.5 }
	c := p
	if pick() { c.LeadHigh = other.LeadHigh }
	if pick() { c.PullTrumpWithRight = other.PullTrumpWithRight }
	if pick() { c.PullTrumpMinCount = other.PullTrumpMinCount }
	if pick() { c.CashAcesEarly = other.CashAcesEarly }
	if pick() { c.VoidHunting = other.VoidHunting }
	if pick() { c.DuckAndCover = other.DuckAndCover }
	if pick() { c.OvertrumpPartner = other.OvertrumpPartner }
	if pick() { c.DefensiveLeadRight = other.DefensiveLeadRight }
	if pick() { c.DefensiveSaveRight = other.DefensiveSaveRight }
	if pick() { c.DefensiveAvoidLeadingIntoHand = other.DefensiveAvoidLeadingIntoHand }
	if pick() { c.DefensiveTrumpWithRightThreshold = other.DefensiveTrumpWithRightThreshold }
	if pick() { c.DefensiveTrumpLeadMin = other.DefensiveTrumpLeadMin }
	if pick() { c.DefensiveLeadKing = other.DefensiveLeadKing }
	if pick() { c.DefensiveLeadHigh = other.DefensiveLeadHigh }
	if pick() { c.DefensiveHighFollow = other.DefensiveHighFollow }
	if pick() { c.DefensiveSacrificeLead = other.DefensiveSacrificeLead }
	return c
}

// Crossover returns a new genome by randomly taking each parameter from g or other.
func (g Genome) Crossover(other Genome, rng *rand.Rand) Genome {
	pick := func() bool { return rng.Float64() < 0.5 }
	c := g
	if pick() { c.PartnerTricksEstimate = other.PartnerTricksEstimate }
	if pick() { c.BidPadding = other.BidPadding }
	if pick() { c.Bid5Threshold = other.Bid5Threshold }
	if pick() { c.Bid6Threshold = other.Bid6Threshold }
	if pick() { c.ScoreDeficitFactor = other.ScoreDeficitFactor }
	if pick() { c.ScoreSurplusFactor = other.ScoreSurplusFactor }
	if pick() { c.ScoreCloseoutBonus = other.ScoreCloseoutBonus }
	if pick() { c.SeatPositionBias = other.SeatPositionBias }
	if pick() { c.OvercallBias = other.OvercallBias }
	if pick() { c.OpeningBidFactor = other.OpeningBidFactor }
	if pick() { c.RightBowerScore = other.RightBowerScore }
	if pick() { c.LeftBowerScore = other.LeftBowerScore }
	if pick() { c.AceKingScore = other.AceKingScore }
	if pick() { c.LowTrumpScore = other.LowTrumpScore }
	if pick() { c.MajorSuitBonus = other.MajorSuitBonus }
	if pick() { c.TrumpLengthBonus = other.TrumpLengthBonus }
	if pick() { c.PepperRequireBothRights = other.PepperRequireBothRights }
	if pick() { c.PepperMinLeftBowers = other.PepperMinLeftBowers }
	if pick() { c.PepperMinTrump = other.PepperMinTrump }
	if pick() { c.PepperDiscardKeepAces = other.PepperDiscardKeepAces }
	if pick() { c.DeficitRatio = other.DeficitRatio }
	if pick() { c.EndgameTrickThreshold = other.EndgameTrickThreshold }
	c.Normal = crossoverProfile(g.Normal, other.Normal, rng)
	c.Deficit = crossoverProfile(g.Deficit, other.Deficit, rng)
	c.Endgame = crossoverProfile(g.Endgame, other.Endgame, rng)
	return c
}

func clampF(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(hi, v))
}

func clampI(v, lo, hi int) int {
	if v < lo { return lo }
	if v > hi { return hi }
	return v
}

func randSign(rng *rand.Rand) int {
	if rng.Float64() < 0.5 {
		return 1
	}
	return -1
}
