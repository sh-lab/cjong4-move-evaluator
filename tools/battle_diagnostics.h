#ifndef CJ4ME_BATTLE_DIAGNOSTICS_H
#define CJ4ME_BATTLE_DIAGNOSTICS_H

#include "cjong4/manager/manager.h"
#include "cjong4/player/hand_analysis.h"
#include <inttypes.h>
#include <stdio.h>

typedef struct {
  uint64_t decisions;
  uint64_t actions[CJ4_ACTION_PASS + 1];
  uint64_t call_opportunities, call_selected, call_pass;
  uint64_t riichi_opportunities, riichi_selected;
  uint64_t discard_decisions, discard_choices, worse_choices;
  uint64_t tenpai_available, tenpai_kept;
  int64_t before_sum, after_sum, best_sum, gap_sum;
} battle_diagnostics;

static inline int battle_shanten_min(cj4_shanten_result s) {
  int n = s.standard;
  if (s.chiitoitsu < n)
    n = s.chiitoitsu;
  if (s.kokushi < n)
    n = s.kokushi;
  return n;
}

static inline bool battle_is_discard(cj4_action_type t) {
  return t == CJ4_ACTION_DISCARD || t == CJ4_ACTION_RIICHI;
}

static inline bool battle_is_call(cj4_action_type t) {
  return t == CJ4_ACTION_CHI || t == CJ4_ACTION_PON || t == CJ4_ACTION_MINKAN;
}

/* Observe the chosen action; a competing claim may prevent its application.
 * No RNG use, model calls, or mutation of view/actions is allowed here. */
static inline bool battle_diagnostics_record(battle_diagnostics *d,
                                             const cj4_player_view *view,
                                             const cj4_action *actions,
                                             uint8_t count,
                                             const cj4_action *selected) {
  bool call = false, riichi = false, win = false;
  bool tiles[CJ4_TILE_ID_COUNT] = {false};
  unsigned unique = 0;
  int before = 0, after = 0, best = CJ4_SHANTEN_NOT_APPLICABLE;
  cj4_shanten_result s;
  if (!count || count > CJ4M_MAX_ACTIONS ||
      (unsigned)selected->type > CJ4_ACTION_PASS)
    return false;
  for (uint8_t i = 0; i < count; ++i) {
    call |= battle_is_call(actions[i].type);
    riichi |= actions[i].type == CJ4_ACTION_RIICHI;
    win |= actions[i].type == CJ4_ACTION_TSUMO ||
           actions[i].type == CJ4_ACTION_RON;
  }
  if (battle_is_discard(selected->type)) {
    if (!cj4p_calculate_shanten(view, &s))
      return false;
    before = battle_shanten_min(s);
    if (!cj4p_calculate_shanten_after_discard(view, selected->tile, &s))
      return false;
    after = battle_shanten_min(s);
    for (uint8_t i = 0; i < count; ++i) {
      if (!battle_is_discard(actions[i].type))
        continue;
      unsigned tile = actions[i].tile;
      if (tile >= CJ4_TILE_ID_COUNT)
        return false;
      if (tiles[tile])
        continue;
      tiles[tile] = true;
      ++unique;
      if (!cj4p_calculate_shanten_after_discard(view, actions[i].tile, &s))
        return false;
      int value = battle_shanten_min(s);
      if (value < best)
        best = value;
    }
    if (!unique || before == CJ4_SHANTEN_NOT_APPLICABLE ||
        after == CJ4_SHANTEN_NOT_APPLICABLE ||
        best == CJ4_SHANTEN_NOT_APPLICABLE ||
        selected->tile >= CJ4_TILE_ID_COUNT || !tiles[selected->tile] ||
        after < best)
      return false;
  }
  ++d->decisions;
  ++d->actions[selected->type];
  /* Forced wins are not counted as missed call/riichi opportunities. */
  if (!win && call) {
    ++d->call_opportunities;
    d->call_selected += battle_is_call(selected->type);
    d->call_pass += selected->type == CJ4_ACTION_PASS;
  }
  if (!win && riichi) {
    ++d->riichi_opportunities;
    d->riichi_selected += selected->type == CJ4_ACTION_RIICHI;
  }
  if (battle_is_discard(selected->type)) {
    ++d->discard_decisions;
    d->before_sum += before;
    d->after_sum += after;
    d->best_sum += best;
    d->gap_sum += after - best;
    if (unique > 1) {
      ++d->discard_choices;
      d->worse_choices += after > best;
    }
    if (best == 0) {
      ++d->tenpai_available;
      d->tenpai_kept += after == 0;
    }
  }
  return true;
}

static inline void battle_diagnostics_print(const battle_diagnostics *d) {
  static const char *names[] = {"discard", "chi",           "pon",    "ankan",
                                "minkan",  "kakan",         "riichi", "tsumo",
                                "ron",     "abortive_draw", "pass"};
  fprintf(stderr, "diagnostics target_only=1 decisions=%" PRIu64, d->decisions);
  for (unsigned i = 0; i <= CJ4_ACTION_PASS; ++i)
    fprintf(stderr, " %s=%" PRIu64, names[i], d->actions[i]);
  fprintf(stderr,
          "\ncall_opportunities=%" PRIu64 " call_selected=%" PRIu64
          " call_pass=%" PRIu64 " riichi_opportunities=%" PRIu64
          " riichi_selected=%" PRIu64 "\n",
          d->call_opportunities, d->call_selected, d->call_pass,
          d->riichi_opportunities, d->riichi_selected);
  fprintf(stderr,
          "discard_decisions=%" PRIu64 " discard_choices=%" PRIu64
          " worse_choices=%" PRIu64 " tenpai_available=%" PRIu64
          " tenpai_kept=%" PRIu64 "\n",
          d->discard_decisions, d->discard_choices, d->worse_choices,
          d->tenpai_available, d->tenpai_kept);
  if (d->discard_choices)
    fprintf(stderr, "worse_choice_rate=%.2f%%\n",
            100.0 * d->worse_choices / d->discard_choices);
  else
    fprintf(stderr, "worse_choice_rate=N/A\n");
  if (d->discard_decisions)
    fprintf(stderr,
            "mean_before_shanten=%.4f mean_after_shanten=%.4f "
            "mean_best_legal_shanten=%.4f mean_shanten_gap=%.4f\n",
            (double)d->before_sum / d->discard_decisions,
            (double)d->after_sum / d->discard_decisions,
            (double)d->best_sum / d->discard_decisions,
            (double)d->gap_sum / d->discard_decisions);
  else
    fprintf(stderr, "mean_before_shanten=N/A mean_after_shanten=N/A "
                    "mean_best_legal_shanten=N/A mean_shanten_gap=N/A\n");
}
#endif
