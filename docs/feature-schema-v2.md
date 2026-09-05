# Feature schema v2

`CJ4ME_FEATURE_SCHEMA_VERSION` is `2` and `CJ4ME_FEATURE_COUNT` is `1369`.
The encoder accepts `cj4_player_view`, public `cj4_rules`, and one offered
`cj4_action`. It never accepts `cj4_mahjong`. Rules are required so physical
tiles marked in `rules.aka_tiles` can be distinguished without hidden state.

All player-indexed regions use relative order:

```text
relative = (absolute + 4 - view.player) % 4
```

Base tile regions use `cj4_tile_get_type()`. Separate aka regions record
whether each physical tile is marked red by the active rules. Unknown or
invalid optional tiles use a zero valid flag and an all-zero type vector.

| Offset | Length | Region | Encoding |
| ---: | ---: | --- | --- |
| 0 | 34 | own hand | count by tile type / 4 |
| 34 | 136 | discards | relative player x tile-type count / 4 |
| 170 | 136 | tsumogiri discards | relative player x tile-type count / 4 |
| 306 | 136 | riichi declaration discards | relative player x tile-type count / 4 |
| 442 | 136 | discard order summary | normalized latest discard-history index for each relative player and tile type |
| 578 | 136 | exposed meld tiles | relative player x tile-type count / 4 |
| 714 | 20 | meld kinds | relative player x chi/pon/minkan/ankan/kakan count / 4 |
| 734 | 34 | visible dora indicators | count by tile type / 4 |
| 768 | 35 | draw tile | valid flag + tile-type one-hot |
| 803 | 35 | last discard | valid flag + tile-type one-hot |
| 838 | 35 | kan tile | valid flag + tile-type one-hot |
| 873 | 4 | scores | relative player score / 100000, clamped to [-1, 1] |
| 877 | 4 | current player | relative one-hot |
| 881 | 4 | dealer | relative one-hot |
| 885 | 4 | round wind | one-hot |
| 889 | 8 | phase | one-hot |
| 897 | 1 | honba | value / 16, clamped to [0, 1] |
| 898 | 1 | riichi sticks | value / 16, clamped to [0, 1] |
| 899 | 4 | riichi state | relative booleans |
| 903 | 3 | self flags | temporary furiten, riichi furiten, uninterrupted first turn |
| 906 | 11 | action kind | one-hot in `cj4_action_type` order |
| 917 | 4 | action player | relative one-hot |
| 921 | 35 | action primary tile | valid flag + tile-type one-hot |
| 956 | 34 | action tiles | count by tile type / 4 |
| 990 | 1 | action tile count | value / 4 |
| 991 | 34 | own-hand aka tiles | count by tile type / 4 |
| 1025 | 136 | discard aka tiles | relative player x tile-type count / 4 |
| 1161 | 136 | meld aka tiles | relative player x tile-type count / 4 |
| 1297 | 34 | dora-indicator aka tiles | count by tile type / 4 |
| 1331 | 1 | draw tile is aka | boolean |
| 1332 | 1 | last discard is aka | boolean |
| 1333 | 1 | kan tile is aka | boolean |
| 1334 | 1 | primary action tile is aka | boolean |
| 1335 | 34 | action aka tiles | count by tile type / 4 |

The discard-order summary is zero for unseen player/type pairs. For seen
pairs it is the public discard-history index transformed as
`2 * index / CJ4_DISCARD_HISTORY_MAX - 1`. The discard-count region
distinguishes an unseen pair from a first discard whose order value is `-1`.

Hands, all discards including called discards, melds, and dora indicators
are obtained through the public `cj4_location_collect_*()` functions.
Unknown locations are not interpreted or imputed.
