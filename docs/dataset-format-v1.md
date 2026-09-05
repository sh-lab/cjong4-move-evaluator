# Dataset format v1

All integers and IEEE-754 float32 values are little-endian. C structure
layout and padding are not part of the format.

## Header

The header is exactly 32 bytes.

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 8 | ASCII `CJ4MEDA1` |
| 8 | 4 | format version (`1`) |
| 12 | 4 | feature schema version (`2`) |
| 16 | 4 | feature count (`1369`) |
| 20 | 4 | record count |
| 24 | 4 | record size (`5484`) |
| 28 | 4 | reserved, must be zero |

## Record

Every record is exactly 5484 bytes.

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 5476 | 1369 float32 features |
| 5476 | 4 | float32 target |
| 5480 | 1 | absolute action player |
| 5481 | 1 | action type |
| 5482 | 2 | flags/reserved |

The target is the on-policy round return:

```text
(score_after_round - score_before_round) / reward_scale
```

An empty dataset has a valid header and a record count of zero. Readers must
reject unknown versions, schema/count/record-size mismatches, nonzero
reserved header fields, non-finite features or targets, truncation, and
trailing bytes. The C reader exposes a `failed` flag so callers can
distinguish clean exhaustion from a record decoding or I/O error.
