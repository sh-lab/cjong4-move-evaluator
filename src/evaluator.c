#include "cjong4_move_evaluator/evaluator.h"

#include <string.h>

#include "cjong4_move_evaluator/feature.h"

static bool validate_inputs(const void *model, const cj4_player_view *view,
                            const cj4_rules *rules, const cj4_action *actions,
                            uint8_t action_count, const void *scratch,
                            uint8_t *out_index) {
  return model && view && rules && cj4_rules_validate(rules) && actions &&
         scratch && out_index && action_count > 0u &&
         action_count <= CJ4M_MAX_ACTIONS;
}

static bool find_forced_action(const cj4_action *actions, uint8_t action_count,
                               uint8_t *out_index) {
  for (uint8_t i = 0; i < action_count; ++i) {
    if (actions[i].type == CJ4_ACTION_TSUMO) {
      *out_index = i;
      return true;
    }
  }
  for (uint8_t i = 0; i < action_count; ++i) {
    if (actions[i].type == CJ4_ACTION_RON) {
      *out_index = i;
      return true;
    }
  }
  return false;
}

bool cj4me_select_action_f32(const cj4me_model_f32 *model,
                             const cj4_player_view *view,
                             const cj4_rules *rules, const cj4_action *actions,
                             uint8_t action_count,
                             cj4me_inference_f32_scratch *scratch,
                             uint8_t *out_index, float *out_score) {
  float features[CJ4ME_FEATURE_COUNT];
  float best = 0.0f;

  if (!out_score || !validate_inputs(model, view, rules, actions, action_count,
                                     scratch, out_index)) {
    return false;
  }
  if (find_forced_action(actions, action_count, out_index)) {
    *out_score = 0.0f;
    return true;
  }

  for (uint8_t i = 0; i < action_count; ++i) {
    float score;
    if (!cj4me_encode_features(view, rules, &actions[i], features) ||
        !cj4me_infer_f32(model, features, scratch, &score)) {
      return false;
    }
    if (i == 0u || score > best) {
      best = score;
      *out_index = i;
    }
  }
  *out_score = best;
  return true;
}

bool cj4me_select_action_i8(const cj4me_model_i8 *model,
                            const cj4_player_view *view, const cj4_rules *rules,
                            const cj4_action *actions, uint8_t action_count,
                            cj4me_inference_i8_scratch *scratch,
                            uint8_t *out_index, cj4me_i8_output *out_score) {
  float features[CJ4ME_FEATURE_COUNT];
  cj4me_i8_output best = {0};

  if (!out_score || !validate_inputs(model, view, rules, actions, action_count,
                                     scratch, out_index)) {
    return false;
  }
  if (find_forced_action(actions, action_count, out_index)) {
    out_score->quantized = 0;
    out_score->scale = model->output_scale;
    out_score->value = 0.0f;
    return true;
  }

  for (uint8_t i = 0; i < action_count; ++i) {
    cj4me_i8_output score;
    if (!cj4me_encode_features(view, rules, &actions[i], features) ||
        !cj4me_quantize_input_i8(model, features, scratch->input) ||
        !cj4me_infer_i8(model, scratch->input, scratch, &score)) {
      return false;
    }
    if (i == 0u || score.quantized > best.quantized) {
      best = score;
      *out_index = i;
    }
  }
  *out_score = best;
  return true;
}

bool cj4me_evaluator_context_init(cj4me_evaluator_context *context,
                                  cj4me_model_kind kind, const void *model,
                                  const cj4_rules *rules) {
  if (!context || !model || !rules || !cj4_rules_validate(rules) ||
      (kind != CJ4ME_MODEL_KIND_F32 && kind != CJ4ME_MODEL_KIND_I8)) {
    return false;
  }
  memset(context, 0, sizeof(*context));
  context->kind = kind;
  context->model = model;
  context->rules = rules;
  return true;
}

cj4_action cj4me_evaluator_decide(void *opaque, const cj4_player_view *view,
                                  const cj4_action *actions,
                                  uint8_t action_count) {
  cj4me_evaluator_context *context = (cj4me_evaluator_context *)opaque;
  uint8_t selected = 0u;
  bool ok = false;

  if (!context || !actions || action_count == 0u)
    return (cj4_action){0};

  if (context->kind == CJ4ME_MODEL_KIND_F32) {
    float score;
    ok = cj4me_select_action_f32((const cj4me_model_f32 *)context->model, view,
                                 context->rules, actions, action_count,
                                 &context->scratch.f32, &selected, &score);
  } else if (context->kind == CJ4ME_MODEL_KIND_I8) {
    cj4me_i8_output score;
    ok = cj4me_select_action_i8((const cj4me_model_i8 *)context->model, view,
                                context->rules, actions, action_count,
                                &context->scratch.i8, &selected, &score);
  }
  if (!ok) {
    context->failed = true;
    return actions[0];
  }
  return actions[selected];
}
