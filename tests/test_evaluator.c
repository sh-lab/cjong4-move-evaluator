#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "cjong4/core/location.h"
#include "cjong4_move_evaluator/evaluator.h"
#include "cjong4_move_evaluator/feature.h"

static cj4_player_view empty_view(void) {
  cj4_player_view view;
  memset(&view, 0, sizeof(view));
  memset(view.locations, CJ4_LOCATION_NONE, sizeof(view.locations));
  view.draw_tile = CJ4_TILE_ID_INVALID;
  view.last_discard = CJ4_TILE_ID_INVALID;
  view.kan_tile = CJ4_TILE_ID_INVALID;
  return view;
}

void test_evaluator(void) {
  cj4me_model_f32 *model = (cj4me_model_f32 *)calloc(1u, sizeof(*model));
  cj4me_model_i8 *model_i8 = (cj4me_model_i8 *)calloc(1u, sizeof(*model_i8));
  cj4me_inference_f32_scratch scratch;
  cj4me_inference_i8_scratch scratch_i8;
  cj4_player_view view = empty_view();
  cj4_rules rules = cj4_rules_default();
  cj4_action actions[3];
  cj4me_i8_output score_i8;
  uint8_t selected;
  float score;

  assert(model != NULL);
  assert(model_i8 != NULL);
  memset(actions, 0, sizeof(actions));
  actions[0].type = CJ4_ACTION_DISCARD;
  actions[0].tile = cj4_tile_make(0, 0);
  actions[1].type = CJ4_ACTION_DISCARD;
  actions[1].tile = cj4_tile_make(1, 3);
  actions[2].type = CJ4_ACTION_TSUMO;
  actions[2].tile = CJ4_TILE_ID_INVALID;

  assert(cj4me_select_action_f32(model, &view, &rules, actions, 3u, &scratch,
                                 &selected, &score));
  assert(selected == 2u);

  actions[2].type = CJ4_ACTION_PASS;
  actions[2].tile = CJ4_TILE_ID_INVALID;
  assert(cj4me_select_action_f32(model, &view, &rules, actions, 3u, &scratch,
                                 &selected, &score));
  assert(selected == 0u);

  model->weights1[0][CJ4ME_FEATURE_ACTION_TILE_OFFSET + 1u + 1u] = 1.0f;
  model->weights2[0][0] = 1.0f;
  model->weights3[0][0] = 1.0f;
  model->weights4[0][0] = 1.0f;
  assert(cj4me_select_action_f32(model, &view, &rules, actions, 2u, &scratch,
                                 &selected, &score));
  assert(selected == 1u);
  assert(score == 1.0f);

  actions[1].type = CJ4_ACTION_RON;
  assert(cj4me_select_action_f32(model, &view, &rules, actions, 2u, &scratch,
                                 &selected, &score));
  assert(selected == 1u);

  assert(!cj4me_select_action_f32(model, &view, &rules, actions, 0u, &scratch,
                                  &selected, &score));

  for (size_t i = 0; i < CJ4ME_MODEL_HIDDEN1_COUNT; ++i)
    model_i8->requant_multipliers1[i] = 1;
  for (size_t i = 0; i < CJ4ME_MODEL_HIDDEN2_COUNT; ++i)
    model_i8->requant_multipliers2[i] = 1;
  for (size_t i = 0; i < CJ4ME_MODEL_HIDDEN3_COUNT; ++i)
    model_i8->requant_multipliers3[i] = 1;
  model_i8->input_scale = 1.0f;
  model_i8->output_scale = 1.0f;
  model_i8->weights1[0][CJ4ME_FEATURE_ACTION_TILE_OFFSET + 1u + 1u] = 1;
  model_i8->weights2[0][0] = 1;
  model_i8->weights3[0][0] = 1;
  model_i8->weights4[0][0] = 1;
  actions[1].type = CJ4_ACTION_DISCARD;
  assert(cj4me_select_action_i8(model_i8, &view, &rules, actions, 2u,
                                &scratch_i8, &selected, &score_i8));
  assert(selected == 1u);
  assert(score_i8.quantized == 1);

  free(model);
  free(model_i8);
}
