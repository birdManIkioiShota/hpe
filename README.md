# Head-pose evaluation

6DRepNet360 のベース重みとファインチューニング後の重みを、同一条件で評価・比較するためのツールです。実装は `src/` に集約します。FT実験のスクリプトは `src/training/`、FTの実行結果だけを `ft_runs/` に保存します。前処理済みデータは `datasets/prepared/`、すべての評価結果は `eval/` に保存します。

## 環境

```bash
uv sync
```

PyTorch 2.9.0 / torchvision 0.24.0 の CUDA 12.8 wheel を使用します。評価の既定デバイスは `cuda:0` で、CUDAが利用できない場合にCPUへ自動切り替えは行いません。

## 固定モデルの監査と基準評価

以下は人間が実行するコマンドです。基準評価は学習や次段階の起動を行いません。元の `6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` をSHA-256で照合し、ネットワーク構造を変更せずstrict loadします。

```bash
uv run python -m training.evaluate_baseline \
  --run-id baseline_fp32 \
  --datasets agora_hpe aflw2000 300w_lp
```

監査の後、明示したデータセットだけをGPUで評価します。既定はFP32、batch size 256、8 workers、seed 0です。CUDAが使えない場合は開始前に停止します。メモリが不足する場合は `--batch-size 64` などを明示し、以後の比較も同じ条件で行ってください。

推論せず監査だけを行う場合は、別のrun名で以下を実行します。GPUは不要です。

```bash
uv run python -m training.evaluate_baseline \
  --run-id baseline_audit \
  --datasets agora_hpe aflw2000 300w_lp \
  --audit-only
```

基準評価の監査内容は次のとおりです。

- 指定重みのSHA-256・strict load、ソースコードとuv.lockのハッシュ保存。
- 固定した評価manifestのSHA-256、全件数、ID重複、角度・bbox・クロップの検査。
- 参照される全画像のSHA-256、ヘッダ検証、画像寸法の一致確認。現在の画像を基準として記録します。旧baselineには画像ハッシュがないため、旧実行時との画像同一性を証明するものではありません。
- 独立したSciPy計算と既知の回転による、軸順・符号・後方・特異姿勢・左右反転・SO(3)距離の検査。
- 既存のクロップ規則、RGB、Resize256、CenterCrop224、ImageNet正規化の確認。

注釈と評価対象は変更しません。監査の合格は入力整合性と計算規則の確認であり、注釈の真値性や学習データとの非重複を保証するものではありません。300W-LPは元モデルの既学習データで、退行確認として記録します。

監査は `--max-samples` の指定にかかわらず選択データセットの全入力を確認します。例えば `--max-samples 32 --run-id baseline_smoke` を加えると、推論だけ先頭32件に制限し、結果に `smoke` と明記します。`--audit-only` と `--max-samples` は併用できません。`--precision fp16` は別runで指定できますが、FP32とのスコア差をFT改善として比較しないでください。

監査・画像ハッシュ・推論はtqdmで件数、処理速度、経過時間、残り時間を表示します。推論中は途中の平均回転誤差も表示します。各データセット終了時に件数と平均回転誤差を表示し、途中経過は約10秒ごとに `events.jsonl` に保存します。

```text
eval/<run-id>/
  config.json                 # 実行条件・seed・ソースハッシュ
  status.json                 # running / completed / failed / interrupted
  events.jsonl                # 時刻付き進捗・完了ログ
  audit.json                  # 入力・計算規則の監査結果
  input_lock/*_images.jsonl   # 画像ごとのハッシュ
  evaluations/baseline/      # 監査のみの場合は作りません
    run.json
    summary.json / summary.csv
    predictions/*.csv.gz     # v2では予測・正解の回転行列も保存
    datasets/*/metrics/
    datasets/*/plots/
```

同名runは上書きしません。失敗・Ctrl+C時には `status.json` と `error.txt` を残し、途中の評価は `.baseline.partial/` に保持します。基準評価は再開非対応ですので、再実行には新しい `--run-id` を指定してください。SIGKILL・電源断では終了処理ができず、状態が `running` のまま残ることがあります。

## DAD-3DHeads validationの追加

以下を順に実行します。前処理は`val.tar`だけを読み、validation 4,312件を`datasets/prepared/dad3dheads/`に展開し、`manifest.jsonl`・`metadata.json`・`status.json`を保存します。train/testは読みません。元のアーカイブ・既存の結果は変更・削除しません。途中失敗時は出力を残し、再実行には別の`--data-id`を指定します。

```bash
uv run python -m training.prepare_dad3dheads --data-id dad3dheads

uv run python -m training.evaluate_baseline \
  --run-id baseline_dad_fp32 \
  --datasets dad3dheads \
  --dad-data-id dad3dheads
```

既存の3データセットを再実行する必要はありません。まとめて新規評価する場合のみ、`--datasets agora_hpe aflw2000 300w_lp dad3dheads`と別run名を指定してください。DADの結果は`eval/baseline_dad_fp32/evaluations/baseline/`に保存されます。

DADは回転行列を正解として評価し、前方だけを選別する処理はありません。座標変換は`R_hpe = model_view_matrix[:3,:3].T`です。[公式の座標変換](https://github.com/PinataFarms/DAD-3DHeads/blob/main/dad_3dheads_benchmark/benchmark.py)と[HPEの角度変換](https://github.com/hnuzhy/SemiUHPE/blob/main/src/datasets/dataset_DAD3DHeads.py)を組み合わせると、`R_dad = D @ M`、`R_hpe = R_dad.T @ D = M.T`（`D=diag(1,-1,-1)`）になります。

クロップは提供bbox（xywh）をそのまま使い、既存と同じResize256 → CenterCrop224です。本プロジェクトの固定条件であり、DAD公式の3Dメッシュ評価スコアとは別です。occlusionの真偽属性を面積割合に変換せず、不明として扱います。

DADは元のEuler角を提供しないため、yaw帯の集計には回転行列から導出したRzRyRxのうち`|pitch|<=90°`の等価表現を使用します。後方帯はその`|yaw|>=120°`です。倒立・特異姿勢では帯分類に曖昧さがあり、主指標のSO(3)誤差はこのEuler表現を経由せず計算します。既存3データセットのラベルと帯分類は変更しません。

DAD validationは評価専用です。学習ループ・dev・best重みの選択には接続しません。

## VGGHeadsによる基本FT

モデル構造・初期重みは基準評価と同じです。学習はVGGHeadsのみを使用します。以下のコマンドは人間が実行します。前処理・学習・ベンチマーク評価は独立しており、自動では次段階に進みません。

### 1. 学習用manifestの準備

```bash
uv run python -m training.prepare_data --data-id vgg_data
```

`datasets/VGGHeads/small` と `large` の画像・NPZを対応付け、`datasets/prepared/vgg_data/` にtrain/dev/holdoutのmanifestを作成します。画像は複製・変更しません。注釈のない画像、不正な注釈、画像外が50%を超えるcropなどは理由付きで除外します。

VGGHeadsの`extended_bbox`はxywhとして読み、413次元パラメータの403:409から回転を復元します。座標変換は `R_hpe = R_flame.T @ diag(1,-1,-1)` です。Euler角を経由した後方姿勢の切り捨てや、小規模版の保存済み3D頂点の使用は行いません。[注釈形式・回転変換の参照元](https://github.com/ThomasAston/VGGHeads/tree/10190e90d7d047f7848472d8c913f430f7027cef/yolo_head_training)に合わせています。

学習用のtrain/dev/holdoutは、同一画像の全頭部と同一collection・元ファイル名を同じグループにまとめ、seed固定で概ね80/10/10に分割します。ベンチマークは読み込まず、画像間・データセット間の完全一致照合や類似画像照合は行いません。

SHA-256はファイルが前処理後に変更されていないことを確認するために残しています。画像同士の照合には使いません。学習開始時には分割所属・manifestの整合性に加え、全ユニーク画像を読み直してSHA-256を確認します。学習中も実際にデコードするバイト列を照合し、一時的な不一致はファイルを開き直して最大3回確認します。再試行の内容は`integrity_events.jsonl`に保存し、3回とも一致しなければ停止します。holdoutは推論・モデル選択には使いません。

旧形式で作成済みの前処理結果も、そのまま `--data-id <既存のディレクトリ名>` で使用できます。実行中の旧プロセスは読み込み済みの処理を続行するため、今回の変更で途中から照合処理が省略されるわけではありません。旧結果の除外・分割・メタデータは書き換えません。

前処理は全画像の読み込みとハッシュ計算を伴います。各段階の件数・処理速度・残り時間をtqdmで表示します。既存の同名データ出力は上書きしません。失敗時には別の`--data-id`で再実行してください。

### 2. GPUで学習

```bash
uv run python -m training.train --run-id vgg_ft --data-id vgg_data
```

既定値は以下です。ベンチマークを使ったパラメータ探索は行っていません。

| 項目 | 既定値 |
|---|---|
| Optimizer | AdamW、betas=(0.9, 0.999)、eps=1e-8 |
| 学習率 | backbone 1e-5、出力層 1e-4 |
| Weight decay | 1e-4 |
| バッチ | 64 × 勾配累積2 = 実効128 |
| エポック | 20 |
| Warmup | 最初の1エポックは出力層のみ更新、LRも1エポック線形warmup |
| 以後の学習 | 全層更新、cosine LR decay |
| BatchNorm | running mean/varianceは固定、全層更新時はaffineを学習 |
| Loss | SO(3)測地距離（ラジアン、atan2形式） |
| 勾配クリップ | norm 1.0 |
| 精度 | BF16 AMP、回転変換とlossはFP32、dev推論はFP32・指標はFP64 |
| データ変換 | extended bbox、Resize256 → CenterCrop224、ImageNet正規化 |
| 学習時拡張 | 左右反転50%（正解はSRS変換）、明度・コントラスト・彩度±20% |
| GPU / workers / seed | cuda:0 / 8 / 42 |

CUDAが使用できなければ停止します。CPUへの自動切り替えはありません。OOM時は例えば`--batch-size 32 --accumulation 4`で実効128を維持できます。BF16非対応環境では`--precision fp32`を明示してください。各値はCLI引数で変更できます。

更新前のVGGHeads dev誤差を保存し、各エポック終了時のdev平均測地距離で`best.pth`を選びます。一度も改善しなければ`best.pth`は初期重みのままです。後方帯や小頭部への特別な重み付け、蒸留、他データの追加は基本FTには含めません。VGGHeadsの姿勢は推定された注釈であり、devスコアもその注釈に対する一致度です。

tqdmにはepoch、処理済みバッチ、残り時間、平均loss、出力層LR、頭部/秒を表示します。dev中は平均回転誤差を表示します。

```text
datasets/prepared/vgg_data/
  metadata.json / status.json
  train.jsonl / dev.jsonl / holdout.jsonl
  candidates.jsonl / exclusions.jsonl / image_groups.jsonl
ft_runs/vgg_ft/
  config.json / status.json / events.jsonl
  dev_baseline.json / epoch_001.json ...
  best.pth                 # 既存評価ローダーでstrict load可能
  last.pt                  # 最新の途中checkpoint。モデル・AdamW・scheduler・乱数状態
  integrity_events.jsonl   # 読み込み時のハッシュ再試行（問題がなければ空）
```

中断後の再開は次のコマンドです。

```bash
uv run python -m training.train --run-id vgg_ft --resume
```

`last.pt`は既定で500 optimizer stepごと、および各エポック完了時に原子的に更新します。再開位置は最後に保存されたoptimizer更新の直後です。サンプル順とaugmentationはepoch・sample IDから再構成するため、保存済みの途中位置から継続できます。最初のdev評価中に停止して`last.pt`がない場合は、新しいrun名で開始してください。再開時は元の設定を読み、設定変更の引数は受け付けません。学習に関係するソース、データ、設定の変更時も再開を拒否します。同じrunを複数プロセスで同時に再開しないでください。

中断したFT結果は`ft_runs/vgg_ft_interrupted/`に保存されています。完了済みのFT結果は`ft_runs/vgg_ft/`です。

```bash
uv run python -m training.train --run-id vgg_ft --data-id vgg_data
```

### 3. 学習完了後、基準評価と同条件でベンチマーク評価

```bash
uv run python -m training.evaluate --run-id vgg_ft --baseline-run baseline_fp32
```

これは明示的に実行する最終評価です。学習完了を確認し、`best.pth`を指定した基準評価と同じv2/FP32・同じデータセット選択・同じmanifestで評価し、paired比較を保存します。DADのみの基準評価ならDADのみ、4データセットの基準評価なら4つを評価します。既存の3データセット基準評価もディレクトリ名を指定して使用できます。学習コードへ結果を戻して重みを選び直す処理はありません。

結果は`eval/<run-id>/`、比較は`eval/comparisons/baseline_vs_<run-id>/`です。AGORAの後方帯の測地距離、中央値・P90、90°超率と、AFLW2000・300W-LPの退行を確認できます。従来の`hpe evaluate`は既定v1のため、この基準評価との比較には使いません。

評価時の画像読み込み・デコードは、失敗時にファイルを開き直して最大3回試します。`training.evaluate`では、デコードするバイト列のSHA-256も基準評価の画像ロックと照合します。3回失敗した画像はパス・実測ハッシュ付きで報告して停止します。対象画像のスキップや破損画像の許容は行いません。再試行の記録は評価出力の`image_read_errors/<dataset>/worker_<pid>.jsonl`に保存します。途中停止した場合は`eval/.<run-id>.partial/`に残ります。

途中停止した評価の部分出力を削除した後は、同じ評価名で先頭から実行できます（学習済み`best.pth`を使います）。

```bash
uv run python -m training.evaluate \
  --run-id vgg_ft \
  --baseline-run baseline_fp32
```

基本FTの軽量テストは以下です。注釈・分割所属、実モデルの合成データ1更新と重み互換、optimizer/scheduler/乱数の再開を確認します。実データ学習・ベンチマーク推論は行いません。

```bash
uv run python -m unittest discover -s src/tests -p test_vgg_training.py -v
```

## 基準評価で固定する評価プロトコル v2

旧 `eval/baseline/` はv1です。基準評価は次の差を持つv2として保存し、異なる版の比較を拒否します。旧結果は上書きしません。

| 項目 | v1（従来） | v2（基準評価） |
|---|---|---|
| Vec1/2/3・VMAE | 回転行列の列ベクトル | 配布元と同じ行ベクトル |
| 6Dから回転への正規化 | 推論側の精度設定 | FP32以上に固定 |
| 指標計算 | FP32・acos測地距離 | FP64・0°/180°付近に安定なatan2測地距離 |
| 基準評価の推論設定 | 旧baselineはFP16 | 既定FP32、TF32無効、決定的計算 |

主指標はSO(3)回転誤差で、中央値・P90・P95、90°超の件数・割合も保存します。後方帯の定義は元ラベルの `|yaw| >= 120°` のままです。軸別MAEは、正解・予測とも正規Euler表現にしてから計算する独自の全周評価規則を維持します。**正規Eulerのyawは360°の方位角ではありません。** 配布元の論文数値と直接同条件になることを保証するものではありません。

参照：[配布元の評価・ネットワーク](https://github.com/thohemp/6DRepNet360/blob/master/sixdrepnet360/test.py)、[回転の定義](https://github.com/thohemp/6DRepNet360/blob/master/sixdrepnet360/utils.py)、[クロップ](https://github.com/thohemp/6DRepNet360/blob/master/sixdrepnet360/datasets.py)。

テストは `uv run python -m unittest discover -s src/tests -v` で実行できます。合成画像の結合テストでは実ベンチマーク推論は行いません。従来のテストには、手元の重みのstrict loadとAFLW2000の1画像読み込みが含まれます。

## ベースライン評価

```bash
uv run hpe evaluate \
  --checkpoint checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth \
  --run-name baseline
```

既定では AGORA-HPE、AFLW2000、300W-LP を、batch size 256、8 workers、FP16 autocastで評価します。短い動作確認には `--max-samples` を使用できます。CPUを明示して確認する場合だけ `--device cpu --no-amp` を指定します。

評価中はデータセットごとにtqdmのプログレスバーを表示し、処理済み頭部数、経過時間、処理速度、残り時間を確認できます。比較処理でもデータセット単位の進捗を表示します。

## FT後の評価と比較

```bash
uv run hpe evaluate \
  --checkpoint checkpoints/finetuned.pth \
  --run-name finetuned

uv run hpe compare \
  --baseline eval/baseline \
  --candidate eval/finetuned
```

比較はインスタンス単位で対応付け、候補値－ベースライン値の差とpaired bootstrap 95%信頼区間を出力します。差が負なら改善です。評価プロトコル、マニフェスト、対象サンプル数が一致しないrun同士は比較を拒否します。

## 出力

各runには次を保存します。

- `run.json`: 重みとマニフェストのSHA-256、前処理、実行環境、速度
- `summary.csv` / `summary.json`: データセット別の総合集計
- `predictions/*.csv.gz`: インスタンス単位の正解、予測、各誤差
- `datasets/*/metrics/`: overall、yaw帯、30度yaw bin、頭部サイズ、occlusion別の集計
- `datasets/*/plots/`: yaw bin別の誤差図

主要指標はSO(3)測地距離です。補助指標としてpitch/yaw/rollの循環MAE、3軸の平均MAE、Vec1/Vec2/Vec3/VMAEを保存します。後方帯は元アノテーションの `|yaw| >= 120°` で区分します。後方姿勢ではEuler角に等価表現が存在するため、軸別MAEは正解・予測の回転行列を同じ正規Euler表現へ変換してから計算します。
