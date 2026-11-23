"""Reel script checks (no API calls). Run from the repo root: python -m pytest tests/"""

import copy

import pandas as pd

from src.publish.reel_script import fallback_question, finish, problems
from src.publish.reel_story import Story

GOOD = {
    "beats": [
        {"scene": "hook", "text": "Giannis was No. 30 three weeks ago."},
        {"scene": "proof", "text": "Now he's No. 3. He's scoring 28.2 points a game with 12.6 rebounds and 5.5 assists, "
                                   "shooting .602 from the field."},
        {"scene": "context", "text": "Miami's 9-6, fourth in the East. Only Luka and Wemby are ahead of him."},
        {"scene": "ask", "text": "Giannis or Wemby: who finishes higher?"},
    ],
    "question": "Giannis or Wemby: who finishes higher?",
    "instagram_caption": "Giannis or Wemby: who finishes higher? Full board at stingercollectibles.com",
    "hashtags": ["#StingerNBA", "#NBAMVP", "#GiannisAntetokounmpo", "#MVPRace", "#HEATCulture"],
}


def script(**changes):
    d = copy.deepcopy(GOOD)
    for scene, text in changes.items():
        next(b for b in d["beats"] if b["scene"] == scene)["text"] = text
    return d


def story(kind="top5_climber"):
    return Story(kind, pd.Series({"player_name": "Giannis Antetokounmpo", "player_key": "g"}), 7, 3, "Up 4 spots", "")


def test_good_script_passes():
    assert problems(GOOD) == []


def test_long_hook():
    assert any("hook beat" in p for p in problems(script(hook="So our model has Giannis Antetokounmpo at No. 30 three weeks ago.")))


def test_our_model_in_hook():
    assert any("our model" in p for p in problems(script(hook="Our model loves Giannis now.")))


def test_ending_must_be_a_question():
    assert any("question mark" in p for p in problems(script(ask="Full board at stingercollectibles.com.")))


def test_filler_and_spoken_link():
    p = problems(script(proof="Here's what jumped out to me. I'll be honest, he's scoring 28.2 points a game with 12.6 rebounds.",
                        ask="Check out the full board. Who finishes higher?"))
    assert any("filler" in x and "i'll be honest" in x and "check out" in x and "full board" in x for x in p)


def test_probability_language():
    assert any("probability" in p for p in problems(script(context="He's the favorite now, with Luka and Wemby ahead of him.")))


def test_wordy_proof():
    wordy = ("His scoring barely moved, about twenty-eight a night. Miami did. They were 1-4 and twelfth in the East. "
             "They've gone 8-2 since and sit fourth. That's our model reacting to winning.")
    assert any("proof beat is 31 words" in p for p in problems(script(proof=wordy)))


def test_word_range():
    assert any("must total" in p for p in problems(script(proof="Scoring 28.2.")))


def test_finish_adds_fallback_question():
    d = finish(script(ask="That's the board this week."), story())
    assert d["beats"][-1]["text"].endswith("Is Giannis a real MVP contender?")
    assert d["question"] == "Is Giannis a real MVP contender?"
    assert (d["beats"][0]["start"], d["beats"][-1]["end"]) == (0, 20)


def test_fallback_questions_by_story():
    assert fallback_question(story("new_no1")) == "Is Giannis your MVP right now?"
    assert fallback_question(story("top5_drop")) == "Is Giannis out of the MVP race?"


def _rows():
    now = pd.Series({"rank": 3, "team": "MIA", "team_record": "9-6", "conf_rank": 4, "conference": "East",
                     "pts": 28.2, "reb": 12.6, "ast": 5.5, "blk": 1.3, "fg_pct": 0.602})
    then = pd.Series({"week": 2, "rank": 30, "team": "MIA", "team_record": "1-4", "conf_rank": 12, "conference": "East",
                      "pts": 27.6, "reb": 11.2, "ast": 4.6, "blk": 0.6, "fg_pct": 0.567})
    return now, then


def test_then_now_lines():
    from src.publish.reel_script import then_now_lines
    lines = then_now_lines(*_rows(), 5)
    assert lines[0].startswith("Then vs now (Week 2, 3 weeks ago")
    assert "- Team record: 1-4 -> 9-6 (8-2 since then)" in lines
    assert "- Conference rank: No. 12 -> No. 4 in the East" in lines
    assert "PPG 27.6 -> 28.2" in lines[-1]


def test_then_now_cards_lead_with_the_team():
    from src.publish.render_reel import then_now_cards
    cards = then_now_cards(*_rows(), ["pts", "reb", "blk", "fg_pct"])
    assert cards[:3] == [("MIA record", "1-4 → 9-6"), ("East rank", "No. 12 → No. 4"), ("PPG", "27.6 → 28.2")]
    assert cards[3][0] == "RPG"            # the model weighs his rebounding gain over the doubled (small) blocks


def test_stat_percentages_are_allowed():
    from src.publish.reel_script import probability_language
    for t in ("60% from the field", "shooting 38 percent from three", "a 68% completion rate",
              "completion percentage of 68", "FG% .602", "his K% jumped", "a .915 save percentage"):
        assert probability_language(t) == [], t


def test_win_percent_and_probability_are_not():
    from src.publish.reel_script import probability_language
    for t in ("a 60% chance at MVP", "his win percentage is .812", "Miami's win % is up", "70 percent of voters"):
        assert probability_language(t), t


def test_jeffs_week5_rewrite_passes():
    jeff = script(hook="No. 30 to No. 3 in three weeks.",
                  proof="His numbers barely moved. Twenty-eight a night, 60% from the field. "
                        "So what changed? Miami. 1-4 then. 8-2 since. That's our model reacting to winning.",
                  context="Only Luka and Wemby are ahead of him on our board.",
                  ask="Giannis or Wemby: who finishes higher?")
    assert problems(jeff) == []
