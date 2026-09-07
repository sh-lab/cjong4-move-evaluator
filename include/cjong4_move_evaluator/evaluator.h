#ifndef CJ4ME_EVALUATOR_H
#define CJ4ME_EVALUATOR_H

#include <stdbool.h>
#include <stdint.h>

#include "cjong4/core/rules.h"
#include "cjong4/manager/delegate.h"
#include "cjong4/manager/manager.h"
#include "cjong4_move_evaluator/model.h"
#include "cjong4_move_evaluator/policy_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  cj4me_model_kind kind;
  const void *model;
  const cj4_rules *rules;
  cj4me_policy_filter_config policy_filter;
  /**
   * Set when the adapter falls back to the first action.
   * The owner must inspect and clear this field after `cj4m_step`.
   */
  bool failed;
  union {
    cj4me_inference_f32_scratch f32;
    cj4me_inference_i8_scratch i8;
  } scratch;
} cj4me_evaluator_context;

/**
 * Selects an action with a float32 model.
 * Forced tsumo/ron selections return a score of zero.
 */
bool cj4me_select_action_f32(const cj4me_model_f32 *model,
                             const cj4_player_view *view,
                             const cj4_rules *rules, const cj4_action *actions,
                             uint8_t action_count,
                             cj4me_inference_f32_scratch *scratch,
                             uint8_t *out_index, float *out_score);

/** Selects with an optional hard/soft policy filter before argmax. */
bool cj4me_select_action_f32_filtered(
    const cj4me_model_f32 *model, const cj4_player_view *view,
    const cj4_rules *rules, const cj4_action *actions, uint8_t action_count,
    const cj4me_policy_filter_config *filter,
    cj4me_inference_f32_scratch *scratch, uint8_t *out_index, float *out_score);

/**
 * Selects an action with an INT8 model using integer argmax.
 * Forced tsumo/ron selections return a quantized score of zero.
 */
bool cj4me_select_action_i8(const cj4me_model_i8 *model,
                            const cj4_player_view *view, const cj4_rules *rules,
                            const cj4_action *actions, uint8_t action_count,
                            cj4me_inference_i8_scratch *scratch,
                            uint8_t *out_index, cj4me_i8_output *out_score);

/** Selects with an optional filter, quantizing soft bias before integer argmax.
 */
bool cj4me_select_action_i8_filtered(
    const cj4me_model_i8 *model, const cj4_player_view *view,
    const cj4_rules *rules, const cj4_action *actions, uint8_t action_count,
    const cj4me_policy_filter_config *filter,
    cj4me_inference_i8_scratch *scratch, uint8_t *out_index,
    cj4me_i8_output *out_score);

/** Initializes a reusable cjong4 delegate adapter context. */
bool cj4me_evaluator_context_init(cj4me_evaluator_context *context,
                                  cj4me_model_kind kind, const void *model,
                                  const cj4_rules *rules);

/** Copies a validated policy filter into a reusable evaluator context. */
bool cj4me_evaluator_context_set_policy_filter(
    cj4me_evaluator_context *context, const cj4me_policy_filter_config *filter);

/** Adapter function suitable for `cj4m_player_delegate.decide`. */
cj4_action cj4me_evaluator_decide(void *context, const cj4_player_view *view,
                                  const cj4_action *actions,
                                  uint8_t action_count);

#ifdef __cplusplus
}
#endif

#endif /* CJ4ME_EVALUATOR_H */
