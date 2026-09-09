#ifndef CJ4ME_BATTLE_POLICY_H
#define CJ4ME_BATTLE_POLICY_H

#include "cjong4/manager/manager.h"
#include "cjong4/player/hand_analysis.h"
#include "rng.h"

/* Closed-hand baseline: no calls, kans, riichi or abortive draws.
 * All choices depend only on the supplied masked PlayerView. */
static bool battle_baseline(cj4me_rng *rng, bool random,
                            const cj4_player_view *view,
                            const cj4_action *actions, uint8_t count,
                            cj4_action *out) {
  uint8_t candidates[CJ4M_MAX_ACTIONS];
  uint32_t n = 0;
  int best = CJ4_SHANTEN_NOT_APPLICABLE;
  if (!count || count > CJ4M_MAX_ACTIONS)
    return false;
  for (uint8_t i = 0; i < count; ++i)
    if (actions[i].type == CJ4_ACTION_TSUMO ||
        actions[i].type == CJ4_ACTION_RON) {
      *out = actions[i];
      return true;
    }
  if (random) {
    *out = actions[cj4me_rng_bounded(rng, count)];
    return true;
  }
  for (uint8_t i = 0; i < count; ++i) {
    cj4_shanten_result s;
    int value;
    if (actions[i].type != CJ4_ACTION_DISCARD)
      continue;
    if (!cj4p_calculate_shanten_after_discard(view, actions[i].tile, &s))
      return false;
    value = s.standard;
    if (s.chiitoitsu < value)
      value = s.chiitoitsu;
    if (s.kokushi < value)
      value = s.kokushi;
    if (value == CJ4_SHANTEN_NOT_APPLICABLE)
      return false;
    if (value < best) {
      best = value;
      n = 0;
    }
    if (value == best)
      candidates[n++] = i;
  }
  if (n) {
    *out = actions[candidates[cj4me_rng_bounded(rng, n)]];
    return true;
  }
  for (uint8_t i = 0; i < count; ++i)
    if (actions[i].type == CJ4_ACTION_PASS) {
      *out = actions[i];
      return true;
    }
  return false;
}
#endif
