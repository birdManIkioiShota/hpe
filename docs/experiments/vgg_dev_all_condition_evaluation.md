# VGGHeads dev 全実験統一評価

## 目的

この評価は、過去のHPE実験で生成した38個の論理条件を、同じVGGHeads development split 51,914件で再評価するためのものです。

38条件には同一checkpoint SHA-256を持つ組が2組あります。評価では論理条件を38件のまま保持し、同一SHA-256のcheckpointは入力identity、評価設定、評価コードhashが一致する場合だけ推論結果を共有します。固定レジストリ上の一意checkpoint数は36件です。

本評価は外部benchmarkの結果をcheckpoint選択へ利用する処理ではありません。既存実験で既に選択済み、または固定epochとして定義済みのcheckpointを、VGGHeads dev上で共通条件に揃えて比較します。

## 入力

評価入力は datasets/prepared/vgg_data/dev.jsonl へ固定します。件数は51,914、datasetはvggheads、splitはdevです。

実行時には datasets/prepared/vgg_data/metadata.json と照合し、manifest SHA-256、件数、dataset、split、instance IDの重複、rotation matrix、画像SHA-256フィールドを検証します。instance IDはmanifest順にSHA-256へ集約し、推論後のpredictionも同じID順であることを確認します。

画像はmanifestに記録されたcropを使用し、既存のevaluation transformを適用します。前処理はResize 256、CenterCrop 224、ImageNet mean/std normalizationで、augmentationは使用しません。画像読込ではmanifestのimage_sha256を使用し、読込またはhash照合に失敗した画像を黙って除外しません。

## 評価対象

評価対象は src/experiments/common/vgg_dev_targets.py へ固定しています。

| 対象 | 条件数 |
|---|---:|
| audited base checkpoint | 1 |
| VGGHeads通常fine-tuning best epoch 19 | 1 |
| weight interpolation alpha 0.25 | 1 |
| rear-balanced distillation best epoch 7 | 1 |
| rear flip-consistency best epoch 8 | 1 |
| rear sampling × flip weight final epoch 10 | 9 |
| YawPose reliability adoption final epoch 10 | 6 |
| YawPose 15度層別single-draw final epoch 10 | 12 |
| YawPose loss weight final epoch 10 | 6 |
| 合計 | 38 |

同一SHA-256の共有組は、experiment_7_yawpose_stratified:Y_base_vgg_dad と experiment_8_yawpose_loss_weight:Y_base_vgg_dad、および experiment_7_yawpose_stratified:Y_top60_vgg_dad と experiment_8_yawpose_loss_weight:Y_w020_vgg_dad です。

各対象は、checkpointファイルのSHA-256だけでなく、既存の外部評価run.jsonに保存されたcheckpoint SHA-256とも照合します。

## 推論条件

モデルは全条件でSixDRepNet360-ResNet50に固定します。

| 項目 | 設定 |
|---|---|
| model mode | evaluation |
| inference | torch.inference_mode |
| model output | FP32 rotation matrix |
| AMP | 無効 |
| default batch size | 256 |
| default workers | 8 |
| sample order | manifest順、shuffleなし |
| deterministic algorithms | 有効 |
| CUDA matmul FP32 precision | IEEE |
| cuDNN convolution FP32 precision | IEEE |
| metric dtype | float64 |

batch size、workers、device、入力identity、checkpoint SHA-256、評価コードhashはevaluation fingerprintへ含めます。保存済み結果はfingerprintと成果物hashが一致した場合だけ再利用します。

## 指標

画像単位では、instance ID、image path、GT rotation matrix、predicted rotation matrix、GT canonical Euler pitch/yaw/roll、predicted canonical Euler pitch/yaw/roll、GT head-forward full-range yaw、pitch/yaw/rollのabsolute circular error、3軸平均誤差、SO(3) geodesic errorを保存します。

集計する5指標は次のとおりです。

| 指標 | 定義 |
|---|---|
| SO(3) mean | predictionとGT rotation matrix間のstable atan2 geodesic error平均 |
| pitch MAAE | canonical Euler pitchのabsolute circular error平均 |
| yaw MAAE | canonical Euler yawのabsolute circular error平均 |
| roll MAAE | canonical Euler rollのabsolute circular error平均 |
| 3-axis mean | pitch/yaw/roll MAAEの算術平均 |

単位はすべて度です。

## 15度yaw区間

15度区間の分類にはEuler yawを使用しません。GT rotation matrixでhead-local +Z 軸を回転し、水平面への射影から atan2(R[0,2], R[2,2]) で全周yawを求めます。

区間は [-180,-165) から開始する24区間で、最後の区間だけ [165,180] として180度を含みます。head-forward方向の水平成分が1e-8未満の場合はyaw未定義として扱います。yaw未定義画像は全体指標に含め、24区間には入れず、別groupとして件数と指標を保存します。空区間の指標は0ではなくnullとして保存します。

24区間とyaw未定義groupの件数合計は、必ず全体件数51,914件と一致させます。また、各group平均を件数で重み付けすると全体平均へ戻ることを実行時に検証します。

## 実行

全38条件を評価するコマンドは次のとおりです。

    uv run python -m experiments.scripts.evaluate_vgg_dev --all

一つの実験だけを実行する場合はexperiment IDを指定します。

    uv run python -m experiments.scripts.evaluate_vgg_dev --experiment experiment_7_yawpose_stratified

一つの論理条件だけを実行する場合は完全なlogical IDを指定します。

    uv run python -m experiments.scripts.evaluate_vgg_dev --condition experiment_8_yawpose_loss_weight:Y_w060_vgg_dad

保存済みpredictionだけから再集計する場合は --aggregate-only を使用します。このモードではcheckpointをロードせず、predictionの件数とinstance ID順を現在のVGGHeads dev manifestと照合してから集計します。

    uv run python -m experiments.scripts.evaluate_vgg_dev --all --aggregate-only

## 成果物

標準出力先は eval/vgg_dev_all_conditions/ です。

    eval/vgg_dev_all_conditions/
      models/
        <checkpoint-sha256>/
          run.json
          status.json
          validation.json
          predictions.csv.gz
          metrics/
            aggregate.json
            overall.csv
            yaw_bins_15.csv
            undefined_yaw.csv
      comparisons/
        all/
          overall.json
          overall.csv
          yaw_bins_15.json
          yaw_bins_15.csv
          targets.json
        <experiment-or-condition-selection>/
          ...
      reports/
        all/
          <experiment>_overall.md
          <experiment>_yaw_bins_15.md
        <experiment-or-condition-selection>/
          ...

predictionは一意checkpoint単位で保存します。比較表では同じpredictionを共有する条件も38個の論理条件として別行へ展開します。

partial resultは models/.<checkpoint-sha256>.partial/ に保存します。途中状態はcompleted resultと区別し、次回実行では未完了checkpointだけを先頭から再実行します。

completed resultはevaluation fingerprint一致、predictionファイル存在、aggregateファイル存在、prediction SHA-256一致、aggregate SHA-256一致をすべて満たす場合だけ再利用します。

## 回帰確認

audited base checkpointとVGGHeads通常fine-tuningについては、既存のVGGHeads dev評価値を回帰確認値として固定しています。

| 条件 | 既存SO(3) mean |
|---|---:|
| audited base | 21.426986640470798° |
| VGGHeads FT best epoch 19 | 2.919495880048786° |

新評価との差が1e-4°を超えた場合は評価を失敗させます。

## テスト

評価契約の単体テストは src/tests/test_vgg_dev_evaluation.py です。38 logical targets / 36 unique checkpoints、同一checkpoint共有2組、179°と-179°の円周誤差2°、full-range head-forward yaw、vertical head-forwardのyaw未定義、15度境界と±180°、空区間の欠測値、24区間＋undefinedからの全体再構成、prediction保存後の再集計、instance ID順hash、evaluation fingerprintを対象とします。
