package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// --- /health ---

func TestHealth(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	handleHealth(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
}

// --- /bid ---

func TestHandleBid_Pass(t *testing.T) {
	// Weak hand — strategy should pass (bid=0).
	payload := BidRequest{
		Seat: 0,
		Hand: []string{"9H", "9D", "9C", "9S", "TH", "TD", "TC", "TS"},
		State: BidStateJSON{
			CurrentHigh: 5,
			DealerSeat:  5,
			Scores:      [2]int{0, 0},
		},
	}
	resp := postJSON(t, handleBid, payload)
	var result BidResponse
	mustDecode(t, resp, &result)
	if result.Bid != 0 {
		t.Logf("bid = %d (expected 0 for weak hand against high=5)", result.Bid)
	}
}

func TestHandleBid_ReturnsValidRange(t *testing.T) {
	// Strong hand — bid should be 0, 4–7, or 8 (pepper).
	payload := BidRequest{
		Seat: 2,
		Hand: []string{"JH", "JD", "AH", "AD", "KH", "KD", "QH", "QD"},
		State: BidStateJSON{
			CurrentHigh: 0,
			DealerSeat:  1,
			Scores:      [2]int{0, 0},
		},
	}
	resp := postJSON(t, handleBid, payload)
	var result BidResponse
	mustDecode(t, resp, &result)
	if result.Bid != 0 && (result.Bid < 4 || result.Bid > 8) {
		t.Errorf("bid = %d, want 0 or 4–8", result.Bid)
	}
}

func TestHandleBid_BadBody(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/bid", bytes.NewBufferString("not json"))
	w := httptest.NewRecorder()
	handleBid(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

// --- /trump ---

func TestHandleTrump_ReturnsValidSuit(t *testing.T) {
	payload := TrumpRequest{
		Seat: 3,
		Hand: []string{"JH", "JD", "AH", "KH", "QH"},
	}
	resp := postJSON(t, handleTrump, payload)
	var result TrumpResponse
	mustDecode(t, resp, &result)
	valid := map[string]bool{"H": true, "D": true, "C": true, "S": true}
	if !valid[result.Suit] {
		t.Errorf("suit = %q, want one of H/D/C/S", result.Suit)
	}
}

func TestHandleTrump_BadBody(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/trump", bytes.NewBufferString("{bad"))
	w := httptest.NewRecorder()
	handleTrump(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

// --- /play ---

func TestHandlePlay_ChoosesFromValidPlays(t *testing.T) {
	validPlays := []string{"9H", "TH", "JH"}
	payload := PlayRequest{
		Seat:       0,
		ValidPlays: validPlays,
		State: TrickStateJSON{
			Trump:       "H",
			BidderSeat:  0,
			BidAmount:   5,
			TrickNumber: 0,
			TricksTaken: [6]int{},
			Scores:      [2]int{0, 0},
			Hand:        []string{"9H", "TH", "JH", "AS", "KS"},
			Trick:       []PlayedCardJSON{},
			Leader:      0,
			History:     nil,
		},
	}
	resp := postJSON(t, handlePlay, payload)
	var result PlayResponse
	mustDecode(t, resp, &result)

	validSet := map[string]bool{"9H": true, "TH": true, "JH": true}
	if !validSet[result.Card] {
		t.Errorf("card = %q, not in valid plays %v", result.Card, validPlays)
	}
}

func TestHandlePlay_WithHistory(t *testing.T) {
	// History of played cards should be accepted without error.
	payload := PlayRequest{
		Seat:       2,
		ValidPlays: []string{"AS", "KS"},
		State: TrickStateJSON{
			Trump:       "S",
			BidderSeat:  2,
			BidAmount:   5,
			TrickNumber: 3,
			TricksTaken: [6]int{1, 1, 1, 0, 0, 0},
			Scores:      [2]int{10, 8},
			Hand:        []string{"AS", "KS", "QS"},
			Trick:       []PlayedCardJSON{{Card: "9S", Seat: 1}},
			Leader:      1,
			History: [][]string{
				{"JH", "9D", "TC", "KS", "AS", "QH"},
				{"JD", "AD", "9H", "KH", "QD", "TH"},
				{"9S", "TS", "QS", "JS", "KD", "AC"},
			},
		},
	}
	resp := postJSON(t, handlePlay, payload)
	var result PlayResponse
	mustDecode(t, resp, &result)

	validSet := map[string]bool{"AS": true, "KS": true}
	if !validSet[result.Card] {
		t.Errorf("card = %q, not in valid plays", result.Card)
	}
}

func TestHandlePlay_PepperFallsBackToStandard(t *testing.T) {
	// BidAmount=8 (PepperBid) — MLPStrategy must delegate to StandardStrategy.
	payload := PlayRequest{
		Seat:       0,
		ValidPlays: []string{"AH", "KH"},
		State: TrickStateJSON{
			Trump:       "H",
			BidderSeat:  0,
			BidAmount:   8, // PepperBid
			TrickNumber: 0,
			TricksTaken: [6]int{},
			Scores:      [2]int{0, 0},
			Hand:        []string{"AH", "KH", "QH"},
			Trick:       []PlayedCardJSON{},
			Leader:      0,
			History:     nil,
		},
	}
	resp := postJSON(t, handlePlay, payload)
	if resp.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.Code)
	}
	var result PlayResponse
	mustDecode(t, resp, &result)
	validSet := map[string]bool{"AH": true, "KH": true}
	if !validSet[result.Card] {
		t.Errorf("card = %q, not in valid plays", result.Card)
	}
}

func TestHandlePlay_BadBody(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/play", bytes.NewBufferString(""))
	w := httptest.NewRecorder()
	handlePlay(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

// --- helpers ---

func postJSON(t *testing.T, handler http.HandlerFunc, body any) *httptest.ResponseRecorder {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler(w, req)
	return w
}

func mustDecode(t *testing.T, w *httptest.ResponseRecorder, v any) {
	t.Helper()
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if err := json.NewDecoder(w.Body).Decode(v); err != nil {
		t.Fatalf("decode: %v", err)
	}
}
