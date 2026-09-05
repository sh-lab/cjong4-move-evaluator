# Model format v2

All integers and IEEE-754 float32 values are little-endian. C structure layout
and padding are not part of the format. Model v2 is tied to feature schema v3.

## Architecture

```text
shared Tile Encoder: 13 -> 16 ReLU -> 8 ReLU, applied to 136 tiles
score network:       1123 -> 128 ReLU -> 64 ReLU -> 16 ReLU -> 1 linear
```

The 1123 score inputs are the 1088 concatenated tile embeddings followed by
33 state features and 2 action features.

## Header

The header is exactly 88 bytes.

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 8 | ASCII `CJ4MEM02` |
| 8 | 4 | model format version (`2`) |
| 12 | 4 | feature schema version (`3`) |
| 16 | 4 | kind (`1` float32, `2` INT8) |
| 20 | 4 | layer count (`6`) |
| 24 | 4 | physical tile count (`136`) |
| 28 | 4 | tile feature count (`13`) |
| 32 | 4 | Tile Encoder hidden count (`16`) |
| 36 | 4 | tile embedding count (`8`) |
| 40 | 4 | state feature count (`33`) |
| 44 | 4 | action feature count (`2`) |
| 48 | 4 | score-network input count (`1123`) |
| 52 | 4 | score hidden 1 (`128`) |
| 56 | 4 | score hidden 2 (`64`) |
| 60 | 4 | score hidden 3 (`16`) |
| 64 | 4 | output count (`1`) |
| 68 | 4 | tensor record count (`12` or `21`) |
| 72 | 8 | payload byte length |
| 80 | 4 | header size (`88`) |
| 84 | 4 | reserved, must be zero |

## Tensor records

Each tensor starts with a 16-byte record header: tensor ID, element type,
element count, and byte length as four uint32 values. Element types are
`1 = float32`, `2 = int8`, and `3 = int32`. Values are row-major.

### Float32 kind

IDs 1 through 12 contain weight and bias pairs in network order:

| IDs | Layer | Weight shape | Bias shape |
| ---: | --- | --- | --- |
| 1, 2 | Tile Encoder 1 | `[16][13]` | `[16]` |
| 3, 4 | Tile Encoder 2 | `[8][16]` | `[8]` |
| 5, 6 | Score 1 | `[128][1123]` | `[128]` |
| 7, 8 | Score 2 | `[64][128]` | `[64]` |
| 9, 10 | Score 3 | `[16][64]` | `[16]` |
| 11, 12 | Score output | `[1][16]` | `[1]` |

The canonical float32 file size is 614,460 bytes.

### INT8 kind

Each layer has three records: int8 weights, int32 biases, and per-output-channel
float32 weight scales. Their IDs are `101..118` in the same six-layer order.

| ID | Value | Count |
| ---: | --- | ---: |
| 119 | activation scales | 7 |
| 120 | requantization multipliers | 232 |
| 121 | requantization shifts | 232 |

Activation scales are ordered as tile input, tile hidden, score input, score
hidden 1, score hidden 2, score hidden 3, and final output. Tile Encoder layer 2
emits directly at the score-input scale so its output can be concatenated with
the separately quantized 35 context features.

The canonical INT8 file size is 157,484 bytes. Loaders reject incompatible
headers, tensor order/type/size mismatches, non-finite or non-positive scales,
invalid requantization parameters, unsafe int32 accumulator bounds, truncation,
and trailing bytes.
