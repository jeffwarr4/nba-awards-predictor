"""Reel script checks (no API calls). Run from the repo root: python -m pytest tests/"""

import copy

import pandas as pd

from src.publish.reel_script import (
    caption_with_cta, driver, fallback_question, finish, milestone, probability_language, problems, proof_opener, story_frame, unanchored_time,
)
from src.publish.reel_story import Story

NAMES = ["Luka Dončić", "Victor Wembanyama", "Giannis Antetokounmpo", "Nikola Jokić", "Cade Cunningham",
         "Alperen Şengün", "Jayson Tatum", "Tyrese Maxey", "Karl-Anthony Towns", "Shai Gilgeous-Alexander",
         "Walker Kessler", "Austin Reaves"]
TEAMS = ["LAL", "SAS", "MIA", "DEN", "DET", "HOU", "BOS", "PHI", "NYK", "OKC", "UTA", "LAL"]
BOARD = pd.DataFrame({"rank": range(1, 13), "player_name": NAMES, "team": TEAMS, "player_key": [f"k{i}" for i in range(1, 13)]})


def row(name):
    return BOARD[BOARD["player_name"] == name].iloc[0]


def story(kind="top5_climber", name="Giannis Antetokounmpo", prev=7, cur=None):
    r = row(name)
    return Story(kind, r, prev, int(r["rank"]) if cur is None else cur, "Up 4 spots", "")


# Example 1 from the prompt (a team-driven climb), as edited to pass the checks
GOOD = {
    "beats": [
        {"scene": "hook", "text": "Giannis: No. 30 to No. 3 in three weeks."},
        {"scene": "proof", "text": "His numbers barely moved: twenty-eight points and nearly thirteen boards a night. "
                                   "The reason? Miami, 1-4 three weeks ago, has gone 8-2 since."},
        {"scene": "context", "text": "Only Luka and Wemby are ahead of him on our board."},
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


def check(d, s=None, ranks=None):
    s = s or story(prev=30)
    return problems(d, s, BOARD, {"ranks": ranks or {30, 3}})


def test_example_1_passes():
    assert check(GOOD) == []


def test_example_2_passes():
    # Example 2 (NFL) on an NBA-shaped board: a No. 10 player, milestone the top 5
    ex2 = {"beats": [
        {"scene": "hook", "text": "Shai Gilgeous-Alexander: No. 17 to No. 10 in three weeks."},
        {"scene": "proof", "text": "Here's why. His last three games: twelve touchdowns, two wins, and his "
                                   "season rating up from 93 to 101."},
        {"scene": "context", "text": "Five spots from our top five, and Cunningham is the one to catch."},
        {"scene": "ask", "text": "Can SGA crack the top 5?"}],
        "question": "Can SGA crack the top 5?", "instagram_caption": GOOD["instagram_caption"], "hashtags": GOOD["hashtags"]}
    assert check(ex2, story("bubble_breakthrough", "Shai Gilgeous-Alexander", prev=17), {17, 10}) == []


def test_this_weeks_nfl_script_fails_the_new_checks():
    # 2026 NFL Week 11 (v2), with an NBA No. 10 standing in for Dak: no name, unanchored then/now, rank repeated,
    # board leader named
    old = {"beats": [
        {"scene": "hook", "text": "No. 17 to No. 10 in one week."},
        {"scene": "proof", "text": "His passing got sharper. Rating about 93 then. About 101 now. Twelve touchdowns "
                                   "in three games. Dallas went 2-1 since. Our model rewards that."},
        {"scene": "context", "text": "He's No. 10 now. Luka Dončić holds No. 1."},
        {"scene": "ask", "text": "Can Shai climb into the top 5?"}],
        "question": "Can Shai climb into the top 5?", "instagram_caption": GOOD["instagram_caption"], "hashtags": GOOD["hashtags"]}
    p = " || ".join(check(old, story("bubble_breakthrough", "Shai Gilgeous-Alexander", prev=17), {17, 10}))
    assert "full name" in p and "hook must name him" in p
    assert "Rating about 93 then." in p and "Dallas went 2-1 since." in p
    assert "repeats the hook's rank (No. 10)" in p and "board leader" in p


def test_nickname_counts_as_the_full_name():
    assert check(script(hook="Giannis: No. 30 to No. 3 in three weeks.")) == []
    p = check(script(hook="No. 30 to No. 3 in three weeks."))
    assert any("hook must name him" in x for x in p)


def test_full_name_needed_without_a_nickname():
    s = story("top5_climber", "Nikola Jokić", prev=9)
    d = script(hook="Jokić: No. 9 to No. 4 in three weeks.", ask="Jokić or Wemby: who finishes higher?",
               context="Only three players sit ahead of him on our board.")
    d["question"] = d["beats"][-1]["text"]
    assert any("full name" in x for x in check(d, s, {9, 4}))
    d["beats"][0]["text"] = "Nikola Jokic: No. 9 to No. 4 in three weeks."     # accents don't matter
    assert check(d, s, {9, 4}) == []


def test_time_anchor_heuristic():
    assert unanchored_time("Miami, 1-4 three weeks ago, has gone 8-2 since.") == []
    assert unanchored_time("He's No. 3 now, up from No. 30 in three weeks.") == []
    assert unanchored_time("Is Giannis your MVP right now?") == []
    assert unanchored_time("They've gone 8-2 since. Rating about 93 then.") == ["They've gone 8-2 since.", "Rating about 93 then."]
    assert unanchored_time("No. 3, known for defense.") == []          # "No." and "known" aren't time words


def test_context_rank_and_leader():
    p = check(script(context="He's No. 3 behind Luka and Wemby."))
    assert any("repeats the hook's rank (No. 3)" in x for x in p)
    s = story("bubble_breakthrough", "Shai Gilgeous-Alexander", prev=17)          # No. 10: leader not allowed
    d = script(hook="SGA: No. 17 to No. 10 in three weeks.", context="Luka leads, and SGA is five spots out.",
               ask="Can SGA crack the top 5?")
    d["question"] = "Can SGA crack the top 5?"
    assert any("board leader (Luka Dončić)" in x for x in check(d, s, {17, 10}))
    from src.publish.reel_script import context_problems
    # a No. 1 story may name the leader; so may a No. 2-3 player (the milestone is No. 1)
    assert context_problems("Wemby is one spot behind Luka.", story("new_no1", "Luka Dončić", prev=3), BOARD, {3, 1}) == []
    assert context_problems("Only Luka and Wemby are ahead of him.", story(), BOARD, {30, 3}) == []


def test_stock_phrases_and_my_model():
    p = check(script(proof="You won't believe it. My model says his numbers barely moved. So what changed? Miami, "
                           "1-4 three weeks ago, has gone 8-2 since."))
    assert any("you won't believe" in x and "my model" in x for x in p)
    p = check(script(proof="Let's dive in. His numbers barely moved. So what changed? Miami, 1-4 three weeks ago, "
                           "has gone 8-2 since."))
    assert any("let's" in x for x in p)


def test_word_range_40_to_50():
    assert any("must total" in x for x in check(script(proof="So what changed? Miami.")))
    long = script(context="Only Luka Dončić and Victor Wembanyama are ahead of him on our board this week, both "
                          "of them by a real margin.")
    assert any("must total" in x for x in check(long))


def test_hook_and_proof_caps():
    assert any("hook beat" in x for x in check(script(hook="So, Giannis Antetokounmpo went all the way from No. 30 to No. 3.")))
    wordy = ("His scoring barely moved, about twenty-eight a night. Miami did. They were 1-4 three weeks ago and "
             "twelfth in the East, and they've gone 8-2 in the three weeks since.")
    assert any("proof beat is" in x for x in check(script(proof=wordy)))


def test_ending_must_be_a_question():
    assert any("question mark" in x for x in check(script(ask="Full board at stingercollectibles.com.")))


def test_finish_adds_milestone_question():
    s = story("bubble_breakthrough", "Shai Gilgeous-Alexander", prev=17)
    d = finish(script(ask="That's the board this week."), s, BOARD)
    assert d["question"] == "Can SGA crack the top 5?"
    assert d["beats"][-1]["text"].endswith("Can SGA crack the top 5?")
    assert (d["beats"][0]["start"], d["beats"][-1]["end"]) == (0, 20)


def test_fallback_questions_by_story():
    assert fallback_question(story("new_no1")) == "Is Giannis your MVP right now?"
    assert fallback_question(story("top5_drop")) == "Is Giannis out of the MVP race?"
    assert fallback_question(story("top5_climber"), BOARD) == "Can Giannis catch Luka?"
    assert fallback_question(story("top5_climber", "Cade Cunningham"), BOARD) == "Can Cade crack the top 3?"
    assert fallback_question(story("bubble_breakthrough", "Austin Reaves", prev=20), BOARD) == "Can Austin crack the top 10?"


def test_milestones():
    assert milestone(story("profile_no1", "Luka Dončić"), BOARD)["kind"] == "hold"
    m = milestone(story(), BOARD)
    assert (m["kind"], m["gap"], m["other"]["player_name"]) == ("no1", 2, "Luka Dončić")
    m = milestone(story("bubble_breakthrough", "Shai Gilgeous-Alexander"), BOARD)
    assert (m["target"], m["gap"], m["other"]["player_name"]) == (5, 5, "Cade Cunningham")


def _rows():
    now = pd.Series({"rank": 3, "team": "MIA", "team_record": "9-6", "conf_rank": 4, "conference": "East",
                     "games": 15, "pts": 28.2, "reb": 12.6, "ast": 5.5, "blk": 1.3, "fg_pct": 0.602,
                     "player_name": "Giannis Antetokounmpo"})
    then = pd.Series({"week": 2, "rank": 30, "team": "MIA", "team_record": "1-4", "conf_rank": 12, "conference": "East",
                      "games": 5, "pts": 27.6, "reb": 11.2, "ast": 4.6, "blk": 0.6, "fg_pct": 0.567})
    return now, then


def test_driver_player_team_unclear():
    now, then = _rows()
    # 2027 Week 5: points +0.6, rebounds +1.4, assists +0.9 a game (all under REEL_FLAT_STATS); Miami 1-4, then 8-2
    d = driver(now, then, 30)
    assert d["driver"] == "team" and d["team"]["window"] == "8-2"
    scoring = now.copy()
    scoring["pts"] = 31.0                                          # +3.4 points a game: his own production
    assert driver(scoring, then, 30)["driver"] == "player"
    hot_start = then.copy()
    hot_start["team_record"] = "4-1"                              # 4-1, then 5-5 in the window: no swing up
    assert driver(now, hot_start, 30)["driver"] == "unclear"
    assert driver(now, None, 7)["driver"] == "unclear"


def test_story_frame_window():
    now, then = _rows()
    s = Story("top5_climber", BOARD.iloc[2].combine_first(now), 7, 3, "No. 30 → No. 3 in 3 weeks", "")
    f = story_frame(s, BOARD, then, 5)
    assert f["window"] == "the last three weeks (Week 2 to Week 5), his last ten games"
    assert f["ranks"] == {30, 3} and f["start_rank"] == 30


def _rows_then_now_lines():
    from src.publish.reel_script import then_now_lines
    return then_now_lines(*_rows(), 5)


def test_then_now_lines():
    lines = _rows_then_now_lines()
    assert lines[0].startswith("Then vs now (Week 2, 3 weeks ago")
    assert "- Team record: 1-4 -> 9-6 (8-2 in the window)" in lines
    assert "- Conference rank: No. 12 -> No. 4 in the East" in lines
    assert "PPG 27.6 -> 28.2" in lines[-1]


def test_then_now_cards_lead_with_the_team():
    from src.publish.render_reel import then_now_cards
    cards = then_now_cards(*_rows(), ["pts", "reb", "blk", "fg_pct"])
    assert cards[:3] == [("MIA record", "1-4 → 9-6"), ("East rank", "No. 12 → No. 4"), ("PPG", "27.6 → 28.2")]
    assert cards[3][0] == "RPG"            # the model weighs his rebounding gain over the doubled (small) blocks


def test_stat_percentages_are_allowed():
    for t in ("60% from the field", "38% from three", "shooting 38 percent from three", "a 68% completion rate",
              "completion percentage of 68", "FG% .602", "his K% jumped", "a .915 save percentage",
              "a 92% save percentage", "save percentage"):
        assert probability_language(t) == [], t


def test_win_percent_and_probability_are_not():
    for t in ("a 60% chance at MVP", "he's likely to win it", "the odds moved", "his win percentage is .812",
              "Miami's win % is up", "70 percent of voters", "the favorite now", "a lock for MVP"):
        assert probability_language(t), t


def test_crossed_and_the_board_driver():
    from src.publish.reel_script import crossed, story_frame
    earlier = BOARD.assign(rank=[3, 2, 4, 1, 5, 6, 7, 8, 9, 10, 11, 12])            # Jokić No. 1, Luka No. 3 then
    s = story("new_no1", "Luka Dončić", prev=3)
    assert [c["name"] for c in crossed(s, BOARD, earlier, 3)] == ["Victor Wembanyama", "Nikola Jokić"]
    luka = BOARD.iloc[0].copy()
    luka["games"], luka["team_record"] = 10, "8-2"
    then = luka.copy()
    then["week"], then["rank"] = 4, 3
    s = Story("new_no1", luka, 3, 1, "New No. 1", "")
    assert story_frame(s, BOARD, then, 5, earlier)["driver"]["driver"] == "board"     # no games in the window


def test_proof_opener_rotates():
    four = [proof_opener(w) for w in range(1, 5)]
    assert len(set(four)) == 4                                   # a different lead-in each week, all four in turn
    assert proof_opener(5) == proof_opener(1)
    assert all(proof_opener(w) != proof_opener(w + 1) for w in range(1, 20))


def test_proof_opener_check():
    f = {"ranks": {30, 3}, "opener": "Here's why."}
    base = "His numbers barely moved. {} Miami, 1-4 three weeks ago, has gone 8-2 since."
    assert problems(script(proof=base.format("Here's why.")), story(prev=30), BOARD, f) == []
    p = problems(script(proof=base.format("")), story(prev=30), BOARD, f)               # no lead-in is fine
    assert not any("lead-in" in x for x in p)
    p = problems(script(proof=base.format("So what changed?")), story(prev=30), BOARD, f)
    assert any("lead-in this week is \"Here's why.\"" in x and '"So what changed?"' in x for x in p)
    p = problems(script(proof=base.format("The reason?")), story(prev=30), BOARD, f)
    assert any('"The reason?"' in x for x in p)


def test_story_frame_carries_the_opener():
    assert story_frame(story(prev=30), BOARD, None, 5)["opener"] == proof_opener(5)


def test_caption_site_line_is_its_own_sentence():
    cta = "Full board at stingercollectibles.com."
    assert caption_with_cta("Can SGA pass Tatum for the top 3? He jumped from No. 14 to No. 5 in one week, scoring "
                            "nearly thirty a game – Full board at stingercollectibles.com") == (
        "Can SGA pass Tatum for the top 3? He jumped from No. 14 to No. 5 in one week, scoring nearly thirty a "
        "game. " + cta)
    assert caption_with_cta("Can Aho get back into the top 3? He slid from No. 2 to No. 5 in one week, so see where "
                            "he stands: Full board at stingercollectibles.com").endswith("where he stands. " + cta)
    assert caption_with_cta("Can Giannis catch Luka? He went 30 to 3 on Miami's 8-2 run.") == (
        "Can Giannis catch Luka? He went 30 to 3 on Miami's 8-2 run. " + cta)
    assert caption_with_cta("Can Dak crack the top 5? Twelve touchdowns in three games.") .count(cta) == 1


def test_finish_rewrites_the_caption_ending():
    d = copy.deepcopy(GOOD)
    d["instagram_caption"] = "Giannis or Wemby: who finishes higher? He's up 27 spots: Full board at stingercollectibles.com"
    out = finish(d, story(prev=30), BOARD, {"ranks": {30, 3}})
    assert out["instagram_caption"] == ("Giannis or Wemby: who finishes higher? He's up 27 spots. "
                                        "Full board at stingercollectibles.com.")
    assert out["warnings"] == []
