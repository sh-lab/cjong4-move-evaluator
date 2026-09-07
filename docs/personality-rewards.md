# Personality teacher rewards

One dataset v3 file can produce all eight teacher targets. The stored default
target remains the point-return objective. Training reconstructs a preference
component from teacher-only facts and combines it as follows:

```text
teacher_target = stored_target + personality_weight * personality_component
```

The default `personality_weight` is `0.05`. This keeps personality as a
tie-break-like preference instead of replacing the point objective.

| Preset | Personality component |
| --- | --- |
| `standard` | `0` |
| `safe` | `-deal_in_points / reward_scale` on every record of the dealing-in player, with the same penalty once more on the causal action |
| `menzen` | `-1` when the selected action is chi, pon, or open-kan |
| `call` | `+1` when an available open call was selected, `-1` when it was passed |
| `speed` | on a win, `1 - round_discard_count / 86`; on exhaustive-draw tenpai, one quarter of that value |
| `kokushi` | `+1` when the player won with kokushi or kokushi 13-sided wait |
| `riichi` | `+1` when riichi was selected while available, `-1` when it was not selected |
| `dama` | the inverse of the `riichi` component |

`safe` uses `reward_scale` only to convert points to the same unit as the
stored return; its default is `8000`. All components are deterministic and are
applied identically to training and validation data.

Teacher facts are not NN features. Outcome-only facts such as deal-in, win,
and kokushi therefore shape the target without leaking hidden post-decision
information into inference.

Example:

```sh
python -m cj4me.train \
  --dataset train.cj4medata \
  --validation-dataset validation.cj4medata \
  --personality safe \
  --personality-weight 0.05 \
  --reward-scale 8000 \
  --output safe.pt
```
