# Dataset format v2

All integers and IEEE-754 float32 values are little-endian. C structure
layout and padding are not part of the format. Teacher facts are stored next
to each record but are never included in the 1803 neural-network inputs.

## Header

The header is exactly 32 bytes.

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 8 | ASCII `CJ4MEDA2` |
| 8 | 4 | format version (`2`) |
| 12 | 4 | feature schema version (`3`) |
| 16 | 4 | feature count (`1803`) |
| 20 | 4 | record count |
| 24 | 4 | record size (`7244`) |
| 28 | 4 | reserved, must be zero |

## Record

Every record is exactly 7244 bytes.

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 7212 | 1803 float32 features (feature schema v3) |
| 7212 | 4 | float32 default target |
| 7216 | 1 | absolute action player |
| 7217 | 1 | action type |
| 7218 | 2 | reserved record flags |
| 7220 | 4 | signed score delta from round start through settlement |
| 7224 | 4 | signed delta applied during settlement only |
| 7228 | 4 | nonnegative points paid by a player who dealt in, otherwise zero |
| 7232 | 4 | nonnegative settlement gain of a winner, otherwise zero |
| 7236 | 1 | public discard count when the action was selected |
| 7237 | 1 | final discard count of the round |
| 7238 | 1 | final count minus decision count |
| 7239 | 1 | `cj4_round_end_type` |
| 7240 | 1 | available open-call mask |
| 7241 | 1 | exhaustive-draw tenpai status |
| 7242 | 2 | teacher fact flags |

The default target remains the on-policy round return:

```text
score_delta / reward_scale
```

`score_delta` includes events such as a riichi deposit that happened before
settlement. `settlement_delta` isolates the final transfer, so win points and
deal-in points do not accidentally include an earlier riichi deposit.

## Available-call mask

| Bit | Meaning |
| ---: | --- |
| 0 | at least one chi action was available |
| 1 | at least one pon action was available |
| 2 | at least one open-kan action was available |

This mask is useful especially when the selected action is `PASS`. Ankan and
kakan are not open-call opportunities and are not included in this mask.

## Tenpai status

| Value | Meaning |
| ---: | --- |
| 0 | unknown or not an exhaustive draw |
| 1 | noten |
| 2 | tenpai |

For an ordinary exhaustive draw, v2 derives this exactly from the noten
payment and dealer continuation. Nagashi mangan has different payments, so
cjong4 3.3.0's public `cj4_is_shape_tenpai()` query resolves that case. This
avoids running the more expensive hand search for every ordinary draw while
still recording an exact value. The value is unknown only when the round did
not end in an exhaustive draw. Yaku and furiten are intentionally irrelevant
because the stored fact is shape tenpai.

## Teacher fact flags

| Bit | Name | Meaning |
| ---: | --- | --- |
| 0 | `WAS_MENZEN` | the acting player was menzen before the action |
| 1 | `OPENED_HAND` | the selected chi/pon/open-kan broke menzen |
| 2 | `CALL_AVAILABLE` | the available-call mask is nonzero |
| 3 | `CHOSE_CALL` | the selected action was chi/pon/open-kan |
| 4 | `RIICHI_AVAILABLE` | at least one riichi action was available |
| 5 | `CHOSE_RIICHI` | the selected action was riichi |
| 6 | `PLAYER_WON` | the acting player won the round |
| 7 | `PLAYER_DEALT_IN` | the acting player was the discarder in a ron |
| 8 | `DEAL_IN_ACTION` | this recorded action directly caused that ron |

Round facts such as `PLAYER_WON` and `PLAYER_DEALT_IN` are copied to every
record for that player in the round. `DEAL_IN_ACTION` is attached only to the
recorded discard, riichi, kakan, or ankan that caused ron. If the causal action
had only one legal choice and was therefore not recorded, no earlier action is
incorrectly marked as the cause.

These raw facts allow Python to rebuild targets for standard, defensive,
menzen, call-heavy, speed, value, riichi, and dama personalities without
regenerating self-play games.

An empty dataset has a valid header and a record count of zero. Readers reject
unknown versions, schema/count/record-size mismatches, invalid fact values,
nonzero reserved header fields, non-finite features or targets, truncation, and
trailing bytes. Dataset v1 remains documented for historical files but is not
accepted by the v2 reader.
