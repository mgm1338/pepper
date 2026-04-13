package game

import "github.com/max/pepper/internal/card"

const (
	PassBid   = 0 // player passes
	PepperBid = 8 // player calls pepper (one above the max regular bid of 7)
	StuckBid  = 3 // dealer forced bid when no one else bids
	MinBid    = 4 // minimum voluntary bid
)

// BidResult holds the outcome of a bidding round.
type BidResult struct {
	Winner    int       // seat index of the winning bidder
	Amount    int       // bid amount (3 if dealer stuck, 4–7 normal, 8 pepper)
	IsPepper  bool      // true if pepper was called
	IsStuck   bool      // true if dealer was forced to take it at 3
}

// BidState is the view passed to a strategy's Bid method.
type BidState struct {
	Hand        []card.Card // the bidding player's hand
	Seat        int         // this player's seat
	DealerSeat  int         // who is the dealer
	CurrentHigh int         // current highest bid (0 if no bids yet)
	HighSeat    int         // seat holding the current high bid (-1 if no bids yet)
	SeatsLeft   int         // seats remaining to bid after this one
	Scores      [2]int      // current game scores
}

// RunBidding executes one round of bidding using the provided strategy callbacks.
// bidFn(seat, state) returns the player's bid (0=pass, 99=pepper, else amount).
func RunBidding(
	hands [6][]card.Card,
	dealer int,
	scores [2]int,
	bidFn func(seat int, state BidState) int,
) BidResult {
	currentHigh := 0
	highSeat := -1
	isPepper := false

	// Bidding starts left of dealer, goes clockwise for one round.
	for i := 1; i <= 6; i++ {
		seat := (dealer + i) % 6

		// Dealer is last; if no one has bid yet, dealer is stuck at 3.
		isDealerTurn := seat == dealer
		if isDealerTurn && highSeat == -1 {
			return BidResult{
				Winner:   dealer,
				Amount:   StuckBid,
				IsStuck:  true,
			}
		}

		state := BidState{
			Hand:        hands[seat],
			Seat:        seat,
			DealerSeat:  dealer,
			CurrentHigh: currentHigh,
			HighSeat:    highSeat,
			SeatsLeft:   6 - i,
			Scores:      scores,
		}

		bid := bidFn(seat, state)

		if bid == PepperBid {
			return BidResult{
				Winner:   seat,
				Amount:   PepperBid,
				IsPepper: true,
			}
		}

		if bid == PassBid {
			continue
		}

		// Validate: must strictly exceed current high, and meet minimum.
		minRequired := MinBid
		if currentHigh >= MinBid {
			minRequired = currentHigh + 1
		}
		if bid >= minRequired {
			currentHigh = bid
			highSeat = seat
			isPepper = false
		}
		// Invalid bids (below minimum) are treated as passes.
	}

	return BidResult{
		Winner:   highSeat,
		Amount:   currentHigh,
		IsPepper: isPepper,
	}
}
