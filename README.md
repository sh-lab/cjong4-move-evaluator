# cjong4-move-evaluator

`cjong4-move-evaluator` は、4人打ちリーチ麻雀のプレイヤービューと合法手1手を入力し、その行動の価値を評価するニューラルネットワークです。

評価対象の局面で全合法手を個別に採点し、最も高い評価値を持つ行動を選択します。対局シミュレーションには [cjong4](https://github.com/sh-lab/cjong4) を利用し、最終的な推論処理は Pure C で実装して WebAssembly（WASM）へ変換することを想定しています。

> [!NOTE]
> 現在のdataset/model形式は v1、特徴量スキーマは v2 です。
> 各形式はバージョンで互換性を検証します。

## 目的

- プレイヤーから見える情報だけを使って合法手を評価する
- モンテカルロ自己対局から行動価値を学習する
- 複数の合法手を高速に比較できる小型NNを構築する
- 面前型、鳴き型、安全型など、異なる打ち筋のモデルを生成する
- 学習済みモデルを Pure C で推論し、ネイティブ環境とWASMの両方で利用する

## リポジトリの責務

各リポジトリの役割は次のように分離します。

| リポジトリ | 責務 |
| --- | --- |
| `cjong4` | 麻雀のルール、状態遷移、合法手生成、プレイヤービュー、対局進行 |
| `cjong4-move-evaluator` | モンテカルロ自己対局、特徴量生成、NN学習、行動評価 |
| `cjong4-opponent` | 学習済みモデルを利用して実際に対局するプレイヤー |

## 入出力

入力は、プレイヤービューと合法手1手から生成した固定長特徴量です。

```text
(PlayerView, LegalAction) -> ActionValue
```

NNへ渡す局面情報はプレイヤービューに含まれる情報だけに限定し、他家の手牌や山などの非公開情報は入力しません。赤牌の判定には対局開始時から公開されている `cj4_rules` も使用します。

意思決定時には、同じプレイヤービューに対する全合法手を評価します。

```text
PlayerView + Action 0 -> score 0
PlayerView + Action 1 -> score 1
PlayerView + Action 2 -> score 2
                         ...

selected_action = argmax(scores)
```

ツモおよびロンが可能な場合は、NNの評価にかかわらず選択する制約を設けます。

## ニューラルネットワーク

初期モデルには、WASM上でも高速に推論できる小型の全結合ネットワークを使用します。

```text
Input
  -> Dense 128 + ReLU
  -> Dense 64  + ReLU
  -> Dense 16  + ReLU
  -> Dense 1   + Linear
  -> ActionValue
```

- 学習時の演算精度: `float32`
- 推論時の重みと活性値: 符号付き `int8`
- 推論時のバイアスと積和演算: `int32`
- 隠れ層の活性化関数: ReLU
- 出力層: 活性化関数なしのスカラー値
- 推論時の動的メモリ確保: なし
- 行動選択: 全合法手の評価値の `argmax`

### 量子化

最初に `float32` モデルを学習し、学習後量子化（PTQ）でC/WASM向けの整数モデルへ変換します。精度低下が大きい場合は、量子化対応学習（QAT）を使用します。

```text
int8 Input
  -> INT8 Dense 128 + ReLU
  -> INT8 Dense 64  + ReLU
  -> INT8 Dense 16  + ReLU
  -> INT8 Dense 1
  -> int32 ActionValue
```

全合法手に同じ出力スケールを適用し、最終出力を浮動小数点へ戻さず整数のまま比較できるようにします。ネイティブCとWASMで共通の量子化仕様を使用します。

## 学習方針

ルールベース教師の行動を直接模倣するのではなく、モンテカルロ自己対局で得られた結果を教師値にします。

基本的な学習サイクルは次のとおりです。

1. 現在のNNを使って自己対局する
2. 学習対象の局面と合法手を記録する
3. 実際に選択した行動へ、局終了時の点数変化を行動価値として割り当てる
4. 記録した `(PlayerView, LegalAction, ActionValue)` でNNを更新する
5. 更新したNNで次世代の自己対局を生成する

初期報酬には、局終了時の持ち点変化を正規化した値を使用します。

```text
base_reward = (score_after_round - score_before_round) / reward_scale
```

探索中は一定確率でランダムな合法手を選び、特定の行動だけに学習データが偏ることを防ぎます。過去世代のチェックポイントを使う対戦相手プールは将来の拡張候補で、初回実装には含みません。

## 打ち筋の特徴づけ

まずスタイル補正のない標準モデルを学習し、その重みを初期値として各スタイルへファインチューニングします。

```text
standard
  |- menzen
  |- call
  `- safe
```

各モデルは同じネットワーク構造を使用し、重みと報酬設定だけを変更します。

```text
total_reward = base_reward + style_weight * style_reward
```

- `menzen`: 面前を維持する行動を好む
- `call`: 鳴きを活用する行動を好む
- `safe`: 放銃回避を重視する

スタイル報酬は点数収支より十分小さく設定し、特徴づけによって基本的な強さが大きく損なわれないようにします。

## 実装方針

### C

- cjong4を利用した対局シミュレーション
- モンテカルロ自己対局
- プレイヤービューと合法手の特徴量変換
- 学習データ生成
- 学習済みNNの推論
- 学習済み重みのINT8量子化形式の読み込み
- ネイティブおよびWASM向けビルド

### Python / PyTorch

- Linux上でのNN学習
- GPUを利用したバッチ学習
- チェックポイント管理
- 学習済み重みのC向け変換

特徴量変換の基準実装はC側に置きます。Python側で同じ変換を重複実装せず、Cが生成した固定長テンソルを学習データとして使用します。

## モデルの配布

C/WASM側では同じ推論エンジンを使用し、モデルの重みだけを差し替えます。

```text
models/
  standard.cj4memodel
  menzen.cj4memodel
  call.cj4memodel
  safe.cj4memodel
```

重み形式と特徴量形式にはバージョンを持たせます。Cの構造体をそのままファイルへ保存せず、プラットフォームに依存しない固定形式へ変換します。

## 検証

最低限、次の項目を継続的に検証します。

- 同じ入力に対するPyTorch、ネイティブC、WASMの出力一致
- float32モデルとINT8モデルの評価誤差および対局成績
- 非公開情報が特徴量へ混入していないこと
- 同一シードによる自己対局の再現性
- モデル世代間およびスタイル間の対局成績
- 推論速度、モデルサイズ、WASMバイナリサイズ

```text
PyTorch output ~= native C output ~= WASM output
```

## 依存関係

- CMake 3.16 以上
- ISO C11 コンパイラ
- cjong4 3.2.0 以上（3.x）
- Python 3.10 以上
- NumPy、PyTorch、pytest

## ネイティブビルドとテスト

開発時は sibling repository の cjong4 を指定できます。

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCJONG4_SOURCE_DIR=../cjong4 \
  -DCJ4ME_BUILD_TESTS=ON \
  -DCJ4ME_BUILD_TOOLS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

`CJONG4_SOURCE_DIR` を省略した場合は
`find_package(cjong4 3.2.0 CONFIG REQUIRED)` を使用します。
親プロジェクトがすでに `cjong4::cj4` を定義している場合は、そのtargetを
再利用するため追加指定は不要です。

## データ生成

モデルを指定しない場合、ツモ・ロン以外を一様ランダムに選びます。

```sh
./build/cj4me_generate \
  --games 100 \
  --seed 1 \
  --epsilon 1.0 \
  --reward-scale 8000 \
  --output samples.cj4medata

./build/cj4me_generate \
  --games 20 \
  --seed 2 \
  --epsilon 1.0 \
  --reward-scale 8000 \
  --output validation.cj4medata
```

量子化済みまたは float32 モデルを使用する場合は `--model` を追加し、
`--epsilon` を `1.0` 未満にします。たとえば `--epsilon 0.1` では10%を
ランダム探索し、残りをモデルのargmaxで選びます。`--epsilon 1.0` は
モデル指定時も完全ランダム探索です。同じ seed、設定、モデルからは同じ
dataset が生成されます。
学習用と検証用は異なるseedで別々に生成してください。合法手が1つしかない
局面と、必ず選択するツモ・ロンはdatasetへ記録しません。鳴きと競合する
`PASS` は学習対象として記録します。

## Cでのモデル読み込みと行動選択

モデル本体は大きいため、setup時に一度だけ確保し、推論中は同じmodelと
scratchを再利用します。

```c
#include <stdlib.h>

#include <cjong4_move_evaluator/evaluator.h>
#include <cjong4_move_evaluator/model.h>

cj4me_model_i8 *model = malloc(sizeof(*model));
cj4me_evaluator_context evaluator;
cj4m_player_delegate delegate;
cj4_rules rules = cj4_rules_default();

if (!model ||
    !cj4me_model_i8_load_file(model, "model-i8.cj4memodel") ||
    !cj4me_evaluator_context_init(
        &evaluator, CJ4ME_MODEL_KIND_I8, model, &rules))
{
    /* Handle setup failure. */
}

delegate.ctx = &evaluator;
delegate.decide = cj4me_evaluator_decide;
```

WASMなどファイルシステムを前提にできない環境では
`cj4me_model_i8_load_memory()` を使用します。adapter contextがmodelと
scratchを保持するため、各推論では動的メモリ確保を行いません。
`cj4m_step()` 後に `evaluator.failed` を確認し、trueの場合は対局処理を
停止してください。

## Python環境、学習、export

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e trainer
python -m pytest trainer/tests

python -m cj4me.train \
  --dataset samples.cj4medata \
  --validation-dataset validation.cj4medata \
  --epochs 10 \
  --batch-size 1024 \
  --seed 1 \
  --output model.pt

python -m cj4me.export \
  --checkpoint model.pt \
  --float-output model-f32.cj4memodel \
  --int8-output model-i8.cj4memodel \
  --calibration-dataset samples.cj4medata
```

## Emscripten

CライブラリはOS固有API、スレッド、SIMDを要求しません。cjong4を同じ
toolchainで構成してください。

```sh
emcmake cmake -S . -B build-wasm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCJONG4_SOURCE_DIR=../cjong4 \
  -DCJ4ME_BUILD_TESTS=OFF \
  -DCJ4ME_BUILD_TOOLS=OFF
cmake --build build-wasm --parallel
```

モデルはファイルシステムを前提にせず、メモリ上のbyte列から読み込めます。
ネイティブとWASMで同じ特徴量生成・INT8推論コードを使用します。

## 形式仕様

- [特徴量スキーマ v2](docs/feature-schema-v2.md)
- [dataset形式 v1](docs/dataset-format-v1.md)
- [model形式 v1](docs/model-format-v1.md)

## 初回実装の制限

- 記録するのは実際に選択した行動の on-policy return だけです。
- 未選択の合法手へ分岐する counterfactual rollout は未実装です。
- 未知牌の再決定化は未実装です。
- SIMD最適化は未実装で、C11のscalar参照実装を使用します。
- 面前型・鳴き型・安全型のstyle rewardは差し替え境界だけを用意し、
  初期報酬は局の点数差だけです。
