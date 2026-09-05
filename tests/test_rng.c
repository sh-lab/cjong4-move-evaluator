#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "../src/rng.h"
#include "../src/selfplay.h"
#include "cjong4/core/state_init.h"
#include "cjong4_move_evaluator/dataset.h"

static int files_equal(const char *first_path, const char *second_path) {
  FILE *first = fopen(first_path, "rb");
  FILE *second = fopen(second_path, "rb");
  int equal = first != NULL && second != NULL;

  while (equal) {
    unsigned char a[4096];
    unsigned char b[4096];
    size_t a_count = fread(a, 1u, sizeof(a), first);
    size_t b_count = fread(b, 1u, sizeof(b), second);
    if (a_count != b_count || memcmp(a, b, a_count) != 0)
      equal = 0;
    if (a_count < sizeof(a))
      break;
  }
  if (first)
    fclose(first);
  if (second)
    fclose(second);
  return equal;
}

void test_rng(void) {
  cj4me_rng first;
  cj4me_rng second;
  cj4_tile_id wall_a[CJ4_TILE_ID_COUNT];
  cj4_tile_id wall_b[CJ4_TILE_ID_COUNT];

  cj4me_rng_seed(&first, 123u);
  cj4me_rng_seed(&second, 123u);
  cj4me_rng_shuffle_wall(&first, wall_a);
  cj4me_rng_shuffle_wall(&second, wall_b);
  assert(memcmp(wall_a, wall_b, sizeof(wall_a)) == 0);
  assert(cj4_wall_is_valid(wall_a));

  {
    const char *path_a = "cj4me-selfplay-a.cj4medata";
    const char *path_b = "cj4me-selfplay-b.cj4medata";
    char error[128];
    cj4me_selfplay_config config = {.games = 1u,
                                    .seed = 99u,
                                    .epsilon = 1.0f,
                                    .reward_scale = 8000.0f,
                                    .max_steps_per_game = 10000u,
                                    .max_records_per_round = 4096u,
                                    .output_path = path_a};
    assert(cj4me_generate_dataset(&config, error, sizeof(error)));
    config.output_path = path_b;
    assert(cj4me_generate_dataset(&config, error, sizeof(error)));
    assert(files_equal(path_a, path_b));
    {
      cj4me_dataset_reader reader;
      assert(cj4me_dataset_reader_open(&reader, path_a));
      assert(reader.record_count > 0u);
      cj4me_dataset_reader_close(&reader);
    }
    assert(remove(path_a) == 0);
    assert(remove(path_b) == 0);
  }
}
