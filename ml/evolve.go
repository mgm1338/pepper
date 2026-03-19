package ml

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/max/pepper/internal/strategy"
)

// EvolveConfig controls the evolutionary search.
type EvolveConfig struct {
	// Phase 1: random search
	Phase1Candidates int // number of random genomes to evaluate
	Phase1Games      int // games per candidate vs baseline
	Phase1Keep       int // top N to advance to phase 2

	// Phase 2: tournament
	Phase2Games   int // games per pair in round-robin
	Phase2Workers int // parallel workers

	// Phase 3: evolutionary refinement
	Phase3Generations  int     // number of generations
	Phase3MutationsEach int    // mutations per top genome per generation
	Phase3Games        int     // games per candidate per generation
	Phase3Keep         int     // survivors per generation
	Phase3MutStrength  float64 // mutation strength (0–1)
	Phase3Workers      int     // parallel workers

	Seed    int64
	OutFile string // CSV output path
}

// DefaultEvolveConfig returns sensible defaults.
func DefaultEvolveConfig() EvolveConfig {
	return EvolveConfig{
		Phase1Candidates: 200,
		Phase1Games:      25000,
		Phase1Keep:       20,

		Phase2Games:   50000,
		Phase2Workers: 12,

		Phase3Generations:   20,
		Phase3MutationsEach: 10,
		Phase3Games:         50000,
		Phase3Keep:          5,
		Phase3MutStrength:   1.0,
		Phase3Workers:       12,

		Seed:    time.Now().UnixNano(),
		OutFile: "evolve_results.csv",
	}
}

// Run executes all three phases and returns the best genome found.
func Run(cfg EvolveConfig) Genome {
	rng := rand.New(rand.NewSource(cfg.Seed))
	baseline := balancedGenome()

	w, err := os.Create(cfg.OutFile)
	if err != nil {
		panic(err)
	}
	defer w.Close()
	csvW := csv.NewWriter(w)
	csvW.Write([]string{
		"phase", "generation", "rank", "win_rate", "avg_advantage", "games",
		"partner_est", "bid_pad", "bid5_thresh", "score_deficit",
		"pepper_rr", "pepper_ll", "pepper_trump",
		"deficit_ratio", "endgame_th",
		"n_lead_high", "n_pull_right", "n_pull_min", "n_duck", "n_void",
		"d_lead_high", "e_lead_high",
	})

	writeResults := func(phase string, gen int, results []Result) {
		for _, r := range results {
			g := r.Genome
			csvW.Write([]string{
				phase, strconv.Itoa(gen), strconv.Itoa(r.Rank),
				fmt.Sprintf("%.4f", r.WinRate),
				fmt.Sprintf("%.4f", r.AvgAdvantage),
				strconv.Itoa(r.Games),
				fmt.Sprintf("%.3f", g.PartnerTricksEstimate),
				strconv.Itoa(g.BidPadding),
				fmt.Sprintf("%.3f", g.Bid5Threshold),
				fmt.Sprintf("%.4f", g.ScoreDeficitFactor),
				boolStr(g.PepperRequireBothRights),
				strconv.Itoa(g.PepperMinLeftBowers),
				strconv.Itoa(g.PepperMinTrump),
				fmt.Sprintf("%.2f", g.DeficitRatio),
				strconv.Itoa(g.EndgameTrickThreshold),
				boolStr(g.Normal.LeadHigh),
				boolStr(g.Normal.PullTrumpWithRight),
				strconv.Itoa(g.Normal.PullTrumpMinCount),
				boolStr(g.Normal.DuckAndCover),
				boolStr(g.Normal.VoidHunting),
				boolStr(g.Deficit.LeadHigh),
				boolStr(g.Endgame.LeadHigh),
			})
		}
		csvW.Flush()
	}

	// ── Phase 1: Random Search ─────────────────────────────────────────────
	fmt.Printf("\n╔══════════════════════════════════════════════╗\n")
	fmt.Printf("║  PHASE 1: Random Search (%d candidates)      ║\n", cfg.Phase1Candidates)
	fmt.Printf("╚══════════════════════════════════════════════╝\n")
	fmt.Printf("  Baseline: %s\n", baseline)
	fmt.Printf("  %d games per candidate, %d workers\n\n", cfg.Phase1Games, cfg.Phase2Workers)

	candidates := make([]Genome, cfg.Phase1Candidates)
	for i := range candidates {
		candidates[i] = RandomGenome(rng)
	}

	p1Results := EvaluateParallel(candidates, baseline, cfg.Phase1Games, cfg.Seed, cfg.Phase2Workers)
	SortByWinRate(p1Results)
	for i := range p1Results {
		p1Results[i].Rank = i + 1
	}

	fmt.Printf("  Top %d after Phase 1:\n", cfg.Phase1Keep)
	for i, r := range p1Results[:cfg.Phase1Keep] {
		fmt.Printf("  #%d  %s\n", i+1, r)
	}
	writeResults("1_random", 0, p1Results)

	if cfg.Phase1Keep == 0 {
		return p1Results[0].Genome
	}

	survivors := make([]Genome, cfg.Phase1Keep)
	for i, r := range p1Results[:cfg.Phase1Keep] {
		survivors[i] = r.Genome
	}

	// ── Phase 2: Tournament ────────────────────────────────────────────────
	fmt.Printf("\n╔══════════════════════════════════════════════╗\n")
	fmt.Printf("║  PHASE 2: Tournament (top %d round-robin)    ║\n", cfg.Phase1Keep)
	fmt.Printf("╚══════════════════════════════════════════════╝\n")
	fmt.Printf("  %d games per pair\n\n", cfg.Phase2Games)

	p2Results := Tournament(survivors, cfg.Phase2Games, cfg.Seed+10000, cfg.Phase2Workers)

	fmt.Printf("  Tournament standings:\n")
	for _, r := range p2Results {
		fmt.Printf("  #%d  %s\n", r.Rank, r)
	}
	writeResults("2_tournament", 0, p2Results)

	if cfg.Phase3Generations == 0 {
		return p2Results[0].Genome
	}

	keep := cfg.Phase3Keep
	if keep > len(p2Results) {
		keep = len(p2Results)
	}
	elites := make([]Genome, keep)
	for i := range elites {
		elites[i] = p2Results[i].Genome
	}

	// ── Phase 3: Evolutionary Refinement ──────────────────────────────────
	fmt.Printf("\n╔══════════════════════════════════════════════╗\n")
	fmt.Printf("║  PHASE 3: Evolutionary Refinement            ║\n")
	fmt.Printf("╚══════════════════════════════════════════════╝\n")
	fmt.Printf("  %d generations, %d mutations per elite, %d games per eval\n\n",
		cfg.Phase3Generations, cfg.Phase3MutationsEach, cfg.Phase3Games)

	mutStrength := cfg.Phase3MutStrength
	for gen := 1; gen <= cfg.Phase3Generations; gen++ {
		// Generate next population: elites + mutations + crossbreeds.
		var population []Genome
		population = append(population, elites...)
		for _, e := range elites {
			for m := 0; m < cfg.Phase3MutationsEach; m++ {
				population = append(population, e.Mutate(rng, mutStrength))
			}
		}
		// Add a few crossbreeds between top elites.
		for i := 0; i < len(elites); i++ {
			for j := i + 1; j < len(elites); j++ {
				population = append(population, elites[i].Crossover(elites[j], rng))
			}
		}

		// Evaluate against best known genome (elites[0]) as the opponent.
		opponent := elites[0]
		p3Results := EvaluateParallel(population, opponent, cfg.Phase3Games, cfg.Seed+int64(gen*100000), cfg.Phase3Workers)
		SortByWinRate(p3Results)
		for i := range p3Results {
			p3Results[i].Rank = i + 1
		}

		for i := range elites {
			elites[i] = p3Results[i].Genome
		}

		fmt.Printf("  Gen %d/%d — best: win=%.1f%%  %s\n",
			gen, cfg.Phase3Generations, p3Results[0].WinRate*100, p3Results[0].Genome)
		writeResults("3_evolve", gen, p3Results[:cfg.Phase3Keep*2])

		// Decay mutation strength over generations.
		mutStrength *= 0.85
	}

	best := elites[0]
	fmt.Printf("\n╔══════════════════════════════════════════════╗\n")
	fmt.Printf("║  BEST GENOME FOUND                           ║\n")
	fmt.Printf("╚══════════════════════════════════════════════╝\n")
	fmt.Printf("  %s\n", best)
	fmt.Printf("  Results written to %s\n", cfg.OutFile)

	return best
}

// balancedGenome returns the Balanced preset as a Genome for use as baseline.
func balancedGenome() Genome {
	c := strategy.Balanced
	profileFromStrategy := func(p strategy.PlayProfile) PlayProfileGenome {
		return PlayProfileGenome{
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
	return Genome{
		PartnerTricksEstimate:   c.PartnerTricksEstimate,
		BidPadding:              c.BidPadding,
		Bid5Threshold:           c.Bid5Threshold,
		Bid6Threshold:           c.Bid6Threshold,
		ScoreDeficitFactor:      c.ScoreDeficitFactor,
		ScoreSurplusFactor:      c.ScoreSurplusFactor,
		ScoreCloseoutBonus:      c.ScoreCloseoutBonus,
		SeatPositionBias:        c.SeatPositionBias,
		OvercallBias:            c.OvercallBias,
		OpeningBidFactor:        c.OpeningBidFactor,
		RightBowerScore:         c.RightBowerScore,
		LeftBowerScore:          c.LeftBowerScore,
		AceKingScore:            c.AceKingScore,
		LowTrumpScore:           c.LowTrumpScore,
		MajorSuitBonus:          c.MajorSuitBonus,
		TrumpLengthBonus:        c.TrumpLengthBonus,
		PepperRequireBothRights: c.PepperRequireBothRights,
		PepperMinLeftBowers:     c.PepperMinLeftBowers,
		PepperMinTrump:          c.PepperMinTrump,
		PepperDiscardKeepAces:   c.PepperDiscardKeepAces,
		Normal:                  profileFromStrategy(c.Normal),
		Deficit:                 profileFromStrategy(c.Deficit),
		Endgame:                 profileFromStrategy(c.Endgame),
		DeficitRatio:            c.DeficitRatio,
		EndgameTrickThreshold:   c.EndgameTrickThreshold,
	}
}

func boolStr(b bool) string {
	if b { return "1" }
	return "0"
}
