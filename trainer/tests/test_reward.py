from types import SimpleNamespace

import numpy as np
import pytest

from cj4me.dataset import (
    FACT_CALL_AVAILABLE,
    FACT_CHOSE_CALL,
    FACT_CHOSE_RIICHI,
    FACT_DEAL_IN_ACTION,
    FACT_PLAYER_DEALT_IN,
    FACT_PLAYER_WON,
    FACT_PLAYER_WON_KOKUSHI,
    FACT_RIICHI_AVAILABLE,
    ROUND_END_EXHAUSTIVE_DRAW,
    TENPAI_YES,
)
from cj4me.reward import PERSONALITIES, compose_targets, personality_component


def reward_facts():
    flags = np.array(
        [
            0,
            FACT_PLAYER_DEALT_IN | FACT_DEAL_IN_ACTION,
            FACT_CALL_AVAILABLE | FACT_CHOSE_CALL,
            FACT_CALL_AVAILABLE | FACT_RIICHI_AVAILABLE,
            FACT_PLAYER_WON,
            0,
            FACT_PLAYER_WON | FACT_PLAYER_WON_KOKUSHI,
            FACT_RIICHI_AVAILABLE | FACT_CHOSE_RIICHI,
        ],
        dtype=np.uint16,
    )
    return SimpleNamespace(
        targets=np.full(8, 0.1, dtype=np.float32),
        fact_flags=flags,
        deal_in_points=np.array([0, 12000, 0, 0, 0, 0, 0, 0]),
        round_discard_counts=np.array([0, 0, 0, 0, 43, 43, 0, 0]),
        round_end_types=np.array(
            [0, 0, 0, 0, 1, ROUND_END_EXHAUSTIVE_DRAW, 1, 0]
        ),
        tenpai_statuses=np.array(
            [0, 0, 0, 0, 0, TENPAI_YES, 0, 0]
        ),
    )


def test_all_eight_personality_components():
    facts = reward_facts()

    assert PERSONALITIES == (
        "standard",
        "safe",
        "menzen",
        "call",
        "speed",
        "kokushi",
        "riichi",
        "dama",
    )
    np.testing.assert_array_equal(
        personality_component(facts, "standard"), np.zeros(8)
    )

    safe = personality_component(facts, "safe", reward_scale=8000)
    assert safe[1] == pytest.approx(-3.0)
    assert np.count_nonzero(safe) == 1

    menzen = personality_component(facts, "menzen")
    assert menzen[2] == -1.0
    assert np.count_nonzero(menzen) == 1

    call = personality_component(facts, "call")
    assert call[2] == 1.0
    assert call[3] == -1.0

    speed = personality_component(facts, "speed")
    assert speed[4] == pytest.approx(0.5)
    assert speed[5] == pytest.approx(0.125)
    assert speed[6] == pytest.approx(1.0)

    kokushi = personality_component(facts, "kokushi")
    assert kokushi[6] == 1.0
    assert np.count_nonzero(kokushi) == 1

    riichi = personality_component(facts, "riichi")
    dama = personality_component(facts, "dama")
    assert riichi[3] == -1.0
    assert riichi[7] == 1.0
    np.testing.assert_array_equal(dama, -riichi)


def test_composed_target_preserves_score_and_applies_weight():
    facts = reward_facts()

    standard = compose_targets(facts, "standard", personality_weight=0.05)
    safe = compose_targets(
        facts, "safe", personality_weight=0.05, reward_scale=8000
    )

    np.testing.assert_allclose(standard, facts.targets)
    assert safe[0] == pytest.approx(0.1)
    assert safe[1] == pytest.approx(-0.05)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"personality": "unknown"}, "unknown personality"),
        ({"personality": "safe", "reward_scale": 0}, "reward_scale"),
        (
            {"personality": "safe", "personality_weight": -1},
            "personality_weight",
        ),
    ],
)
def test_rejects_invalid_reward_configuration(kwargs, message):
    facts = reward_facts()

    with pytest.raises(ValueError, match=message):
        compose_targets(facts, **kwargs)
