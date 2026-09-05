# Model format v1

The model is a canonical byte stream. All integers and IEEE-754 binary32
values are little-endian; signed integers use two's-complement encoding. C
structure layout, alignment, padding, and host endianness are never part of
the format.

The only supported network is:

```text
1369 -> Dense 128 + ReLU
    -> Dense 64  + ReLU
    -> Dense 16  + ReLU
    -> Dense 1   + Linear
```

Every weight matrix is row-major `[output][input]`; each output neuron's
contiguous row is followed by the next output neuron's row.

## Header

The header is exactly 64 bytes.

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 8 | ASCII `CJ4MEM01` |
| 8 | 4 | format version (`1`) |
| 12 | 4 | feature schema version (`2`) |
| 16 | 4 | model kind: `1` float32, `2` INT8 |
| 20 | 4 | layer count (`4`) |
| 24 | 4 | dimension 0 (`1369`) |
| 28 | 4 | dimension 1 (`128`) |
| 32 | 4 | dimension 2 (`64`) |
| 36 | 4 | dimension 3 (`16`) |
| 40 | 4 | dimension 4 (`1`) |
| 44 | 4 | tensor count (`8` float32, `15` INT8) |
| 48 | 8 | payload byte length after this header |
| 56 | 4 | header byte length (`64`) |
| 60 | 4 | reserved (`0`) |

## Tensor record

Each tensor is encoded as a 16-byte tensor header followed immediately by
its data.

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 4 | tensor ID |
| 4 | 4 | element type: `1` float32, `2` int8, `3` int32 |
| 8 | 4 | element count |
| 12 | 4 | data byte length, exactly count times element size |
| 16 | variable | packed tensor data |

Tensor records must occur in the order listed below. There is no alignment
or padding between records. Consequently, the canonical total file sizes
are 738884 bytes for float32 and 188124 bytes for INT8.

## Float32 payload

All data elements have type float32. `Wn` is a weight matrix and `Bn` is a
bias vector.

| Order | ID | Name | Shape | Count |
| ---: | ---: | --- | --- | ---: |
| 1 | 1 | W1 | `[128][1369]` | 175232 |
| 2 | 2 | B1 | `[128]` | 128 |
| 3 | 3 | W2 | `[64][128]` | 8192 |
| 4 | 4 | B2 | `[64]` | 64 |
| 5 | 5 | W3 | `[16][64]` | 1024 |
| 6 | 6 | B3 | `[16]` | 16 |
| 7 | 7 | W4 | `[1][16]` | 16 |
| 8 | 8 | B4 | `[1]` | 1 |

All float values must be finite.

## INT8 payload

Weights are signed symmetric int8. Biases and accumulators are int32.
`WSn` contains one positive finite weight scale per output channel.

| Order | ID | Type | Name | Shape/count |
| ---: | ---: | --- | --- | ---: |
| 1 | 101 | int8 | W1 | `[128][1369]` |
| 2 | 102 | int32 | B1 | 128 |
| 3 | 103 | float32 | WS1 | 128 |
| 4 | 104 | int8 | W2 | `[64][128]` |
| 5 | 105 | int32 | B2 | 64 |
| 6 | 106 | float32 | WS2 | 64 |
| 7 | 107 | int8 | W3 | `[16][64]` |
| 8 | 108 | int32 | B3 | 16 |
| 9 | 109 | float32 | WS3 | 16 |
| 10 | 110 | int8 | W4 | `[1][16]` |
| 11 | 111 | int32 | B4 | 1 |
| 12 | 112 | float32 | WS4 | 1 |
| 13 | 113 | float32 | activation scales | 5 |
| 14 | 114 | int32 | requant multipliers | 208 |
| 15 | 115 | int32 | requant shifts | 208 |

The five positive finite activation scales are ordered as input, hidden 1,
hidden 2, hidden 3, and output. The output int32 represents
`real_value = output_int32 * output_scale`, so all actions evaluated by one
model have the same integer comparison scale. Exporters must set
`output_scale = hidden3_scale * WS4[0]`, rounded to float32. The loader
accepts only a relative difference of at most `1e-6`, using the larger of
the encoded and computed scales as the denominator.

The 208 multiplier and shift entries are concatenated in hidden-layer order:
128 entries for layer 1, 64 for layer 2, then 16 for layer 3. Multipliers
must be positive. Shifts are signed and must be in `[-62, 62]`.

For hidden output channel `o`, after the int32 bias and dot product:

1. Apply ReLU to the accumulator.
2. Compute `accumulator * multiplier[o]` in int64.
3. If shift is positive, divide by `2^shift`, rounding to nearest with ties
   away from zero. A zero shift changes nothing.
4. If shift is negative, multiply by `2^(-shift)`.
5. Saturate to int8; ReLU hidden outputs are therefore in `[0, 127]`.

Exporters normally approximate
`input_activation_scale * weight_scale[o] / output_activation_scale` with
the multiplier and signed power-of-two shift. Biases must already be
quantized in the corresponding accumulator scale. The final layer is not
requantized: its int32 accumulator and the common output scale are returned.

## Validation

The memory loaders require one exact byte span and reject null input,
truncation, trailing bytes, bad magic, unknown versions or kinds, schema or
dimension mismatch, wrong tensor order/type/count/length, non-finite or
non-positive scales, non-finite float parameters, invalid requantization
parameters, and any INT8 layer whose worst-case int8 dot product plus bias
could exceed int32. Destination models are caller-owned and are cleared on
failure. Inference uses caller-owned fixed-size scratch and performs no
allocation.

The public `cj4me_model_f32_load_file` and `cj4me_model_i8_load_file`
wrappers validate the seek result and require the exact canonical byte size
for the selected model kind before allocating and reading. They verify the
complete read and EOF, then delegate all format checks to the corresponding
memory loader. Temporary storage is freed on every path and the destination
is cleared on failure.
