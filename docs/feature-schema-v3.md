# Feature schema v3（採用設計・未実装）

この文書は、物理牌単位の情報を保持する特徴量スキーマv3の採用設計を定義する。
現在の実装は集約型のv2であり、v3の実装完了まではdatasetとmodelの現行形式を
変更しない。

## 原則

- 入力は `cj4_player_view`、公開済みの赤牌設定、合法手1手だけから作る。
- `cj4_mahjong`、他家の手牌、未公開の山などの非公開情報は使用しない。
- `view.locations[136]` を牌種別の枚数へ集約せず、物理牌IDごとに保持する。
- 同じ物理牌に属する山、捨て牌、捨て牌履歴、手牌・面子の対応を失わない。
- すべてのプレイヤー番号を観測者基準の相対番号へ変換する。
- ルール設定はNN入力へ含めず、モデルは学習時と同じルールプロファイル専用とする。
- 局番号と残り山牌数は入力しない。

相対プレイヤー番号は次の式で求める。

```text
relative = (absolute + 4 - view.player) % 4
```

`relative == 0` は常に観測者自身である。`view.player` は相対化後に常に0となるため、
独立した特徴量にはしない。

## ネットワーク構造

各物理牌に同じ重みのTile Encoderを適用する。136個の別モデルではなく、1個の
Tile Encoderを136回共有して使用し、スコアNNを含む全体をend-to-endで学習する。

```text
tile[0]   13 -> Dense 16 + ReLU -> Dense 8 + ReLU --+
tile[1]   13 ->        shared Tile Encoder          |
...                                                   +-> concatenate
tile[135] 13 ->        shared Tile Encoder          |      1088
                                                       + state 33
                                                       + action 2
                                                           |
                                                           v
                                            1123 -> 128 -> 64 -> 16 -> 1
```

スコアNNの隠れ層にはReLUを使用し、最後の1出力は線形値とする。

| 領域 | 形状 | 要素数 |
| --- | ---: | ---: |
| 物理牌入力 | `136 x 13` | 1768 |
| 局面入力 | `33` | 33 |
| 合法手入力 | `2` | 2 |
| エンコーダー外部入力合計 |  | 1803 |
| Tile Encoder出力 | `136 x 8` | 1088 |
| スコアNN入力 | `1088 + 33 + 2` | 1123 |

## 物理牌入力

物理牌ID `0..135` の順に、各牌を13要素で表す。物理牌IDは配列位置から決まり、
別の特徴量としては入力しない。

| Index | 名前 | 内容 |
| ---: | --- | --- |
| 0 | `wall` | 山位置 `0..135`、非公開はnone |
| 1 | `discard` | 相対プレイヤーとそのプレイヤー内の捨て牌番号 |
| 2 | `is_tsumogiri` | ツモ切りなら1 |
| 3 | `discard_history` | 対局全体の捨て牌履歴番号 `0..85` |
| 4 | `is_riichi_discard` | リーチ宣言牌なら1 |
| 5 | `hand_owner` | 手牌なら所有者の相対プレイヤー |
| 6 | `meld` | 面子の所有者、グループ、種別 |
| 7 | `is_aka` | 公開ルール上の赤牌なら1 |
| 8 | `is_draw_tile` | `view.draw_tile` がこの物理牌なら1 |
| 9 | `is_last_discard` | `view.last_discard` がこの物理牌なら1 |
| 10 | `is_kan_tile` | `view.kan_tile` がこの物理牌なら1 |
| 11 | `is_action_primary` | `action.tile` がこの物理牌なら1 |
| 12 | `is_action_member` | `action.tiles[0..tile_count)` に含まれるなら1 |

### カテゴリ化

元のpacked byteから重要なフラグを分離し、残りを重複しないカテゴリ番号へ変換する。
noneは有効値の直後のカテゴリに割り当て、`0xff`を有効値と混同しない。

| 名前 | 有効カテゴリ | noneカテゴリ |
| --- | --- | ---: |
| `wall` | 元の山位置 `0..135` | 136 |
| `discard` | `relative_player * 31 + discard_index` (`0..123`) | 124 |
| `discard_history` | 履歴番号 `0..85` | 86 |
| `hand_owner` | 相対プレイヤー `0..3` | 4 |
| `meld` | `((relative_player * 4 + group) * 5 + type)` (`0..79`) | 80 |

`discard` のツモ切りビットは `is_tsumogiri` へ分ける。
`discard_history` のリーチビットは `is_riichi_discard` へ分ける。
`placement` は `hand_owner` と `meld` へ分ける。`meld` にはチー、ポン、明槓、
暗槓、加槓を含む。該当しないbooleanは0とする。

この分解後も、相対化済みの元の `wall`、`discard`、`placement`、
`discard_history` を物理牌ごとに復元できる。例えば鳴かれた捨て牌では、同じ牌に
捨て牌情報、全体履歴、現在の面子情報が同時に残る。

### 正規化

最大カテゴリが `max_category` のカテゴリ値は次の式で `[-1, 1]` へ変換する。

```text
normalized = 2 * category / max_category - 1
```

これを `wall`、`discard`、`discard_history`、`hand_owner`、`meld` に適用する。
有効カテゴリとnoneカテゴリの総数はいずれもINT8の256値以内であり、量子化後も
カテゴリを区別できるスケールを使用する。booleanと参照フラグは `0` または `1`
とする。

## 局面入力

牌参照は物理牌入力側のフラグへ移し、`cj4_player_view` の非牌フィールドを33要素で
表す。

| 領域 | 長さ | Encoding |
| --- | ---: | --- |
| scores | 4 | 相対席順、`score / 100000`、clampしない |
| phase | 8 | one-hot |
| current player | 4 | 相対プレイヤーone-hot |
| dealer | 4 | 相対プレイヤーone-hot |
| round wind | 4 | one-hot |
| honba | 1 | `honba / 16`、clampしない |
| riichi sticks | 1 | `riichi_sticks / 16`、clampしない |
| riichi state | 4 | 相対席順のboolean |
| self flags | 3 | 一時フリテン、リーチフリテン、第一巡継続 |
| 合計 | 33 | |

局番号と残り山牌数は入力しない。`draw_tile`、`last_discard`、`kan_tile` はこの領域へ
牌IDとして置かず、該当する物理牌の参照フラグで表す。

## 合法手入力

合法手固有の大域入力は2要素とする。牌IDは物理牌側の参照フラグで表現する。

| Index | 名前 | Encoding |
| ---: | --- | --- |
| 0 | action type | `2 * action.type / 10 - 1` |
| 1 | tile count | `2 * action.tile_count / 4 - 1` |

`action.player` はcjong4のManagerが観測者本人の合法手だけをDelegateへ渡すため、
相対化後は常に0となる。エンコーダーは `action.player == view.player` を検証し、
異なる場合は入力エラーとする。

行動例は次のとおり。

- 打牌・リーチ: 対象牌の `is_action_primary` と `is_action_member` が1、countは1。
- チー・ポン: 鳴く捨て牌のprimaryが1、手牌から使う2牌のmemberが1、countは2。
- 明槓: 鳴く捨て牌のprimaryが1、手牌から使う3牌のmemberが1、countは3。
- 暗槓: primaryが1、構成する4牌のmemberが1、countは4。
- 加槓: 加える牌のprimaryとmemberが1、countは1。
- ツモ・ロン: 対象牌のprimaryが1、memberはなく、countは0。
- パス: すべての行動参照フラグが0、countは0。

## ルールの扱い

ルール設定はNN入力に含めない。赤牌だけは各物理牌の `is_aka` として、公開済みの
`cj4_rules.aka_tiles` から入力する。学習、自己対局、実運用では同じルール
プロファイルを使用する。

異なるルールでモデルを誤用しないよう、将来のmodel形式にはルールプロファイルID
またはcanonical rules hashをメタデータとして記録し、ロード時に検証する。これは
互換性検証用メタデータであり、NN入力には数えない。

## v2との互換性

v3は牌種別枚数へ集約するv2と互換性がない。実装時には少なくとも次を更新する。

- feature schema version
- PyTorchモデル構造
- Pure C float32/INT8推論
- model形式
- dataset形式またはdatasetからv3入力を生成する境界
- C/Python相互fixture

既存のv2 dataset、checkpoint、float32 model、INT8 modelはv3モデルへ読み込まない。
