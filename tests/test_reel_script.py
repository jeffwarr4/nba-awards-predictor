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
