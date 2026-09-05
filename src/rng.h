#ifndef CJ4ME_RNG_H
#define CJ4ME_RNG_H

#include <stdint.h>

#include "cjong4/core/tile.h"

typedef struct {
  uint64_t state;
} cj4me_rng;

void cj4me_rng_seed(cj4me_rng *rng, uint64_t seed);

uint32_t cj4me_rng_next(cj4me_rng *rng);

uint32_t cj4me_rng_bounded(cj4me_rng *rng, uint32_t bound);

float cj4me_rng_unit(cj4me_rng *rng);

void cj4me_rng_shuffle_wall(cj4me_rng *rng,
                            cj4_tile_id wall[CJ4_TILE_ID_COUNT]);

#endif /* CJ4ME_RNG_H */
