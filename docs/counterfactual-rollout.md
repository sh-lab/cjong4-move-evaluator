# Counterfactual Monte Carlo rollout

`cj4me_generate` can evaluate every legal action at selected decision points by
branching the complete cjong4 state and playing each branch to the end of the
round.

```sh
./build/cj4me_generate \
  --games 10 \
  --seed 1 \
  --epsilon 1.0 \
  --reward-scale 8000 \
  --rollouts-per-action 4 \
  --max-records-per-round 32768 \
  --output counterfactual.cj4medata
```

`--rollouts-per-action 0` is the default and retains the on-policy generator.
When the value is positive, the generator writes one dataset record for every
`legal action x rollout` pair instead of writing the selected on-policy action.
It does not average the rollouts before writing. Duplicate feature/action rows
therefore carry independent outcomes, allowing training to learn their expected
return while preserving exact deal-in, call, tenpai, and kokushi teacher facts.

For Monte Carlo value estimation, train with `--zero-keep-ratio 1` and
`--nonzero-sample-weight 1` unless the resulting sampling bias is deliberately
corrected. Outcome-based downsampling or weighting changes the empirical mean
of repeated rollouts and therefore changes the value being learned.

Tsumo and ron are still selected before NN evaluation and are not recorded.
Decisions with only one legal action are also omitted.

The existing `--epsilon` setting is also used by future decisions inside each
rollout. The configured model and policy filter are reused as well. With a
deterministic model, `epsilon=0`, and a fixed hidden state, repeating a rollout
produces the same result and adds no information. Use a nonzero epsilon for
stochastic rollout policies.

`--max-rollout-decisions N` limits counterfactual generation to the first `N`
eligible decision points in each game. Zero, the default, means unlimited. It
is intended mainly for smoke tests. Every legal action is still evaluated at
each included decision point.

`--skip-games N` simulates the first `N` games without writing records or
running rollouts, while preserving the main-game PRNG sequence and rollout
decision serials. It is intended to isolate and reproduce a later game from a
multi-game run. For example, `--skip-games 4 --games 1` reproduces zero-based
game index 4 from the same seed.

`--max-records-per-round` limits the total rollout records written by a single
round. Increase it when using several rollouts per action. Counterfactual mode
writes completed records directly and does not allocate a buffer proportional
to this limit.

## Parallel generation

Generate independent shards with different seeds when using multiple CPU
processes. Each process must write to a different output path. Merge completed
shards in a deterministic order before training:

```sh
python -m cj4me.merge \
  --output datasets/train.cj4medata \
  datasets/train-00.cj4medata \
  datasets/train-01.cj4medata
```

The merge command validates every shard header and exact file size, preserves
the supplied input order, streams records without loading the dataset into
memory, and writes the result atomically. It refuses to replace an existing
output unless `--overwrite` is specified.

When counterfactual generation fails, the error reports the zero-based game
index, game step, decision serial, candidate action index, rollout index,
records already written in the round, and configured per-round limit. Combine
the reported game index with `--skip-games` to reproduce only that game.

## Determinism and information boundary

Rollouts use a PRNG stream derived from the generator seed, decision serial,
and rollout index. All candidate actions at the same decision use the same
derived seed. The rollout PRNG is separate from the actual game's PRNG, so
enabling counterfactual generation does not change the played trajectory.

The NN features are always encoded from the acting player's `PlayerView` and
the candidate legal action. The complete state is used only by the offline
teacher to advance a copied branch and is never added to the NN input.

This first implementation keeps the copied branch's actual unknown tiles and
wall order. It varies future player decisions but does not re-determine hidden
tiles between rollouts. PlayerView-consistent hidden-state re-determination is
a separate roadmap item and should be added only if evaluation shows that the
fixed hidden state introduces material bias.
