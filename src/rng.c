#include "rng.h"

#include <stddef.h>

void cj4me_rng_seed(cj4me_rng *rng, uint64_t seed) {
  rng->state = seed ? seed : UINT64_C(0x9e3779b97f4a7c15);
}

uint32_t cj4me_rng_next(cj4me_rng *rng) {
  uint64_t x = rng->state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  rng->state = x;
  return (uint32_t)((x * UINT64_C(2685821657736338717)) >> 32);
}

uint32_t cj4me_rng_bounded(cj4me_rng *rng, uint32_t bound) {
  uint32_t threshold;
  uint32_t value;
  if (bound == 0u)
    return 0u;
  threshold = (uint32_t)(-bound) % bound;
  do {
    value = cj4me_rng_next(rng);
  } while (value < threshold);
  return value % bound;
}

float cj4me_rng_unit(cj4me_rng *rng) {
  return (float)(cj4me_rng_next(rng) >> 8) * (1.0f / 16777216.0f);
}

void cj4me_rng_shuffle_wall(cj4me_rng *rng,
                            cj4_tile_id wall[CJ4_TILE_ID_COUNT]) {
  for (uint16_t i = 0; i < CJ4_TILE_ID_COUNT; ++i)
    wall[i] = (cj4_tile_id)i;
  for (uint16_t i = CJ4_TILE_ID_COUNT - 1u; i > 0u; --i) {
    uint16_t j = (uint16_t)cj4me_rng_bounded(rng, (uint32_t)i + 1u);
    cj4_tile_id temporary = wall[i];
    wall[i] = wall[j];
    wall[j] = temporary;
  }
}
