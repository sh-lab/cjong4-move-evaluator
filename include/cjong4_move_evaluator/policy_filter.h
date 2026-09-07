#ifndef CJ4ME_POLICY_FILTER_H
#define CJ4ME_POLICY_FILTER_H

#include <stdbool.h>
#include <stdint.h>

#include "cjong4/manager/manager.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CJ4ME_POLICY_STANDARD = 0,
  CJ4ME_POLICY_SAFE,
  CJ4ME_POLICY_MENZEN,
  CJ4ME_POLICY_CALL,
  CJ4ME_POLICY_SPEED,
  CJ4ME_POLICY_KOKUSHI,
  CJ4ME_POLICY_RIICHI,
  CJ4ME_POLICY_DAMA
} cj4me_policy_profile;

typedef enum {
  CJ4ME_POLICY_FILTER_HARD = 0,
  CJ4ME_POLICY_FILTER_SOFT = 1
} cj4me_policy_filter_mode;

typedef struct {
  bool exclude_open_calls;
  bool require_open_call;
  bool exclude_riichi;
  bool require_riichi;
  bool keep_best_shanten_discards;
  bool prefer_genbutsu_when_threatened;
  uint8_t safe_live_wall_threshold;
  uint8_t safe_tsumogiri_streak;
  bool prefer_kokushi_when_promising;
  uint8_t kokushi_min_unique_yaochu;
  uint8_t kokushi_max_shanten_gap;
  float open_call_bias;
  float pass_with_call_bias;
  float riichi_bias;
  float best_shanten_discard_bias;
  float safe_discard_bias;
  float kokushi_discard_bias;
} cj4me_policy_filter_config;

typedef struct {
  bool allowed[CJ4M_MAX_ACTIONS];
  float score_bias[CJ4M_MAX_ACTIONS];
  uint8_t allowed_count;
  uint8_t threat_mask;
  bool kokushi_active;
  bool used_fallback;
} cj4me_policy_filter_result;

/** Returns a no-op filter configuration. */
cj4me_policy_filter_config cj4me_policy_filter_default(void);

/** Validates combinations and finite soft-score biases. */
bool cj4me_policy_filter_config_validate(
    const cj4me_policy_filter_config *config);

/**
 * Builds one of the eight personality presets.
 *
 * `soft_strength` is expressed in model output units and must be finite and
 * nonnegative. Kokushi activates with at least nine unique terminal/honor
 * types and when its shanten is at most one behind the best ordinary shape.
 */
bool cj4me_policy_filter_preset(cj4me_policy_profile profile,
                                cj4me_policy_filter_mode mode,
                                float soft_strength,
                                cj4me_policy_filter_config *out_config);

/**
 * Filters legal actions and computes optional score biases.
 *
 * Tsumo and ron take precedence over every filter. If hard constraints would
 * remove all candidates, all original actions are restored and
 * `used_fallback` is set.
 */
bool cj4me_policy_filter_apply(const cj4me_policy_filter_config *config,
                               const cj4_player_view *view,
                               const cj4_action *actions, uint8_t action_count,
                               cj4me_policy_filter_result *out_result);

#ifdef __cplusplus
}
#endif

#endif /* CJ4ME_POLICY_FILTER_H */
