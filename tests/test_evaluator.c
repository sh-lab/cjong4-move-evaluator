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

static void configure_f32(cj4me_model_f32 *model) {
  model->tile_weights1[0][CJ4ME_TILE_FEATURE_ACTION_PRIMARY] = 1.0f;
  model->tile_weights2[0][0] = 1.0f;
  model->score_weights1[0][0] = 1.0f;
  model->score_weights1[0][CJ4ME_TILE_EMBEDDING_COUNT] = 2.0f;
  model->score_weights2[0][0] = 1.0f;
  model->score_weights3[0][0] = 1.0f;
  model->score_weights4[0][0] = 1.0f;
}

static void configure_i8(cj4me_model_i8 *model) {
  for (size_t i = 0; i < CJ4ME_MODEL_TILE_HIDDEN_COUNT; ++i)
    model->tile_requant_multipliers1[i] = 1;
  for (size_t i = 0; i < CJ4ME_TILE_EMBEDDING_COUNT; ++i)
    model->tile_requant_multipliers2[i] = 1;
  for (size_t i = 0; i < CJ4ME_MODEL_HIDDEN1_COUNT; ++i)
    model->score_requant_multipliers1[i] = 1;
  for (size_t i = 0; i < CJ4ME_MODEL_HIDDEN2_COUNT; ++i)
    model->score_requant_multipliers2[i] = 1;
  for (size_t i = 0; i < CJ4ME_MODEL_HIDDEN3_COUNT; ++i)
    model->score_requant_multipliers3[i] = 1;
  model->tile_input_scale = 1.0f;
  model->score_input_scale = 1.0f;
  model->output_scale = 1.0f;
  model->tile_weights1[0][CJ4ME_TILE_FEATURE_ACTION_PRIMARY] = 1;
  model->tile_weights2[0][0] = 1;
  model->score_weights1[0][0] = 1;
  model->score_weights1[0][CJ4ME_TILE_EMBEDDING_COUNT] = 2;
  model->score_weights2[0][0] = 1;
  model->score_weights3[0][0] = 1;
  model->score_weights4[0][0] = 1;
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
  actions[0].player = view.player;
  actions[0].tile = cj4_tile_make(0, 0);
  actions[1].type = CJ4_ACTION_DISCARD;
  actions[1].player = view.player;
  actions[1].tile = cj4_tile_make(0, 1);
  actions[2].type = CJ4_ACTION_TSUMO;
  actions[2].player = view.player;
  actions[2].tile = CJ4_TILE_ID_INVALID;

  assert(cj4me_select_action_f32(model, &view, &rules, actions, 3u, &scratch,
                                 &selected, &score));
  assert(selected == 2u);

  actions[2].type = CJ4_ACTION_PASS;
  configure_f32(model);
  assert(cj4me_select_action_f32(model, &view, &rules, actions, 2u, &scratch,
                                 &selected, &score));
  assert(selected == 1u);
  assert(score == 2.0f);

  actions[1].type = CJ4_ACTION_RON;
  assert(cj4me_select_action_f32(model, &view, &rules, actions, 2u, &scratch,
                                 &selected, &score));
  assert(selected == 1u);
  assert(!cj4me_select_action_f32(model, &view, &rules, actions, 0u, &scratch,
                                  &selected, &score));

  actions[1].type = CJ4_ACTION_DISCARD;
  configure_i8(model_i8);
  assert(cj4me_select_action_i8(model_i8, &view, &rules, actions, 2u,
                                &scratch_i8, &selected, &score_i8));
  assert(selected == 1u);
  assert(score_i8.quantized == 2);

  free(model);
  free(model_i8);
}
