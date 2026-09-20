# VGGHeads姿勢分布監査とweight-space interpolation実験レポート

## 目的

6DRepNet360のネットワーク構造と6D回転出力を変更せず、後方姿勢（`|yaw| >= 120°`）の精度を改善しながら、既存データセットに対する性能を維持するための実験です。

既存のVGGHeads fine-tuningでは、VGGHeads devの平均SO(3)誤差が大幅に低下する一方、AGORA-HPEの後方帯とAFLW2000で退行が確認されています。本レポートでは、次の2点を実測しました。

1. VGGHeadsの学習データが全周姿勢をどの程度含むかを、Euler yawに依存しない全周方位で監査しました。
2. 配布元のbase checkpointと既存VGGHeads fine-tuned checkpointをweight spaceで補間し、既存性能の保持とfine-tuningで得た改善を両立できるかを評価しました。

本レポートに記載する結果は、リポジトリに保存された実験成果物から再現可能です。

## 使用したモデルとcheckpoint

学習対象のモデル構造はSixDRepNet360-ResNet50で固定されています。

| 種別 | checkpoint | SHA-256 |
|---|---|---|
| Base | `checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` | `3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6` |
| VGGHeads FT | `ft_runs/vgg_ft/best.pth` | `f36abf2e366ce8cf13049202c2c970d8ea6bf723ee10e7ce9551e5d40f792ea4` |
| Interpolation α=0.25 | `experiments/runs/weight_interp_a025/checkpoints/best.pth` | `c7b60cecdca9dcc9fcbcb8c05d856f7a9164521aeab1824ddb42a0bdb0fcc650` |

VGGHeads FT checkpointは、VGGHeadsのみを使用した既存のfine-tuning結果です。モデル構造はbase checkpointと同一で、320個のstate-dict tensorを持ちます。

## 実験1: VGGHeads姿勢分布監査

### 方位角の定義

通常のcanonical Euler yawは全周の方位角を表さないため、VGGHeadsの後方姿勢を数える用途には使用していません。

各rotation matrix `R`について、head-localの`+Z`軸をhead-forward方向として回転させ、そのXZ平面への射影から方位角を計算しました。

```text
forward = R @ [0, 0, 1]
azimuth = atan2(forward.x, forward.z)
```

方位帯は絶対値で分類しました。

| 帯 | 定義 |
|---|---|
| front | `|azimuth| < 60°` |
| side | `60° <= |azimuth| < 120°` |
| rear-near | `120° <= |azimuth| < 150°` |
| rear-deep | `150° <= |azimuth| <= 180°` |

全520,646サンプルについて方位角を計算でき、undefined azimuthは0件でした。

### 実行コマンド

```bash
uv run python -m experiments.scripts.analyze_pose_distribution \
  --run-id vgg_pose_distribution \
  --data-id vgg_data
```

入力manifestは次の3ファイルです。

| split | samples | manifest SHA-256 |
|---|---:|---|
| train | 417,055 | `087e644fa095374c730b0d10e38af8e8371556ffbb9c2d617aa1f5d3a644b394` |
| dev | 51,914 | `ba83be1e197275697e8919959519e4a5da314d460a34d5457ac9788f1a18d13b` |
| holdout | 51,677 | `0232f40eacde52115d66106ebfe3509635031048a27b4969fd0ec41a939ed95b` |

### 姿勢分布

train splitの姿勢分布は次の結果でした。

| 方位帯 | samples | train比率 |
|---|---:|---:|
| front | 387,509 | 92.916% |
| side | 21,785 | 5.224% |
| rear-near | 3,040 | 0.729% |
| rear-deep | 4,721 | 1.132% |
| rear合計 | 7,761 | 1.861% |

devとholdoutでも同様の分布です。

| split | front | side | rear-near | rear-deep | rear合計 |
|---|---:|---:|---:|---:|---:|
| dev | 48,260 (92.961%) | 2,631 (5.068%) | 383 (0.738%) | 640 (1.233%) | 1,023 (1.971%) |
| holdout | 47,952 (92.792%) | 2,794 (5.407%) | 345 (0.668%) | 586 (1.134%) | 931 (1.802%) |

train rearの左右件数は、negative azimuthが3,648件、positive azimuthが4,113件でした。

### 観測

VGGHeads trainの約93%がfrontで、`|azimuth| >= 120°`のrearは約1.86%です。したがって、通常の一様sample fine-tuningでは、最適化更新の大部分がfront姿勢に由来します。

この分布だけではAGORA-HPE後方帯の退行原因を確定できませんが、VGGHeads全体のdev平均をcheckpoint選択指標にすると、rearがモデル選択へ与える寄与が非常に小さくなることは確認できます。

保存結果は `experiments/runs/vgg_pose_distribution/` にあります。

## 実験2: BaseとVGGHeads FTのweight-space interpolation

### 方法

同一モデル構造のbase checkpointを `W_base`、VGGHeads FT checkpointを `W_ft` とし、各floating-point tensorを次の式で補間しました。

```text
W(alpha) = (1 - alpha) * W_base + alpha * W_ft
```

本実験では `alpha = 0.25` としました。整数型state tensorは両checkpointで完全一致することを要求し、補間後checkpointは未変更のSixDRepNet360へstrict loadして互換性を確認しています。

### 実行コマンド

checkpoint生成は次のコマンドで行いました。

```bash
uv run python -m experiments.scripts.interpolate_checkpoints \
  --run-id weight_interp_a025 \
  --alpha 0.25 \
  --candidate-checkpoint ft_runs/vgg_ft/best.pth
```

正式評価は次のコマンドで行いました。

```bash
uv run python -m experiments.scripts.evaluate_candidate \
  --run-id weight_interp_a025 \
  --baseline-run baseline_fp32
```

### 評価条件

評価は保存済みのFP32 baselineと同じmanifest・画像lock・batch sizeを使用しました。

主要条件は次のとおりです。

| 項目 | 設定 |
|---|---|
| 推論精度 | FP32 |
| AMP | 無効 |
| batch size | 256 |
| workers | 8 |
| device | `cuda:0` |
| metric dtype | FP64 |
| SO(3)誤差 | atan2形式のgeodesic distance |
| vector metric | rotation matrixのrow vectors |
| deterministic | 有効 |
| CUDA matmul / cuDNN conv FP32 precision | IEEE |

評価対象はAGORA-HPE、AFLW2000、300W-LPです。DAD-3DHeadsはこのinterpolation runでは評価していません。

## 評価結果

### データセット全体

平均SO(3) geodesic errorを比較すると、次の結果です。

| dataset | Base | VGGHeads FT | α=0.25 interpolation | interpolation - Base |
|---|---:|---:|---:|---:|
| AGORA-HPE | 45.475° | 39.554° | **37.678°** | **-7.797°** |
| AFLW2000 | 6.562° | 9.423° | **6.046°** | **-0.516°** |
| 300W-LP | 5.845° | **4.423°** | 4.561° | **-1.284°** |

VGGHeads FTではAFLW2000がbaseから約2.86°退行しましたが、α=0.25 interpolationではbaseより約0.52°改善しました。

300W-LPでもbaseより約1.28°改善し、AGORA-HPE全体ではbaseより約7.80°改善しました。

### AGORA-HPE yaw帯

AGORA-HPEをsource yawで分割した結果は次のとおりです。

| yaw帯 | Base mean | VGGHeads FT mean | α=0.25 mean |
|---|---:|---:|---:|
| front `|yaw| < 60°` | 51.091° | **33.789°** | 35.805° |
| side `60° <= |yaw| < 120°` | 47.754° | 40.257° | **39.035°** |
| rear `|yaw| >= 120°` | **37.900°** | 44.820° | 38.458° |

interpolationは、full fine-tuningで悪化したrear meanを44.820°から38.458°まで回復しました。ただしbaseの37.900°は下回っていません。

paired comparisonでrear meanの差は `+0.558°` で、95% bootstrap CIは `[-0.346°, +1.460°]` でした。2,644件中1,539件でbaseより改善し、1,105件で悪化しました。

### AGORA-HPE rearの誤差分布

rearでは平均値だけでなくtailが変化しています。

| metric | Base | VGGHeads FT | α=0.25 |
|---|---:|---:|---:|
| mean | **37.900°** | 44.820° | 38.458° |
| median | 30.502° | 31.186° | **27.731°** |
| P90 | **71.800°** | 113.851° | 80.911° |
| >90° count | **153** | 345 | 221 |
| >90° rate | **5.787%** | 13.048% | 8.359% |

α=0.25ではmedianがbaseより2.77°改善しています。一方でP90と90°超率はbaseより悪化しており、大誤差サンプルがrear平均を押し上げています。

rearの軸別平均では、yaw誤差はbaseの26.610°から24.304°へ改善しましたが、pitchは22.026°から26.438°、rollは17.807°から22.292°へ悪化しました。

### AGORA-HPE rearの左右・角度別結果

30° yaw binでrearを分解すると、左右で異なる挙動が確認されました。

| source yaw bin | Base mean | α=0.25 mean | 差 |
|---|---:|---:|---:|
| -180° to -150° | 29.415° | 34.035° | +4.619° |
| -150° to -120° | 33.738° | 37.268° | +3.530° |
| +120° to +150° | 51.119° | **45.174°** | **-5.945°** |
| +150° to +180° | 36.262° | 36.846° | +0.584° |

特に`+120° to +150°`では明確な改善が得られています。一方、negative yaw側の2 binはどちらも悪化しています。

`-180° to -150°`では、P90が49.212°から76.716°、90°超率が3.869%から7.887%へ増加しました。rear全体のtail悪化には、このような特定領域の大誤差増加が含まれています。

## 考察

### VGGHeads fine-tuningはfront中心のデータ分布で行われている

VGGHeads trainではrearが1.861%しか存在しません。学習時にrearを明示的に重み付けしない場合、rear教師信号はfront教師信号に対して約1対50の規模です。

既存fine-tuningがVGGHeads全体devの平均SO(3)誤差を21.43°から約2.92°まで下げながらAGORA-HPE rearを悪化させたことと、この分布は整合します。ただし、分布不均衡だけから因果関係を断定することはできません。

### BaseとVGGHeads FTの能力は単純な二者択一ではない

α=0.25 interpolationは、AFLW2000でVGGHeads FTが起こした退行を解消し、AGORA-HPE全体・front・sideと300W-LPではbaseより良い値を維持しました。

したがって、VGGHeads FTで得た改善を保ちながらbase側の汎化能力を回復できるweight-space領域が存在することが実測できました。

### Rearは典型誤差とtail誤差を分けて扱う必要がある

α=0.25ではrear medianがbaseより改善していますが、P90と90°超率は悪化しています。さらにbaseより改善したrearサンプル数の方が多いにもかかわらずmeanは悪化しています。

このため、rear meanだけを最適化するより、P90・90°超率・大誤差サンプルの発生領域を独立して追跡する必要があります。

### Rearの失敗は左右対称ではない

α=0.25ではpositive yawの`+120° to +150°`が約5.95°改善する一方、negative yawのrear binは3.53〜4.62°悪化しています。

VGGHeads train rear自体はnegative 3,648件、positive 4,113件であり、件数差だけでこの評価差を説明するには不十分です。ラベル変換、画像条件、augmentation、元モデルとFTモデルの誤差分布をサンプル単位で追加確認する必要があります。

## 制約と注意事項

本実験には次の制約があります。

- weight interpolationは`alpha=0.25`の1条件だけを評価しています。
- AGORA-HPE、AFLW2000、300W-LPの評価結果は既に観測済みです。この結果を見ながら同じbenchmark上でalphaや学習パラメータを探索すると、benchmarkがモデル選択用データになります。
- DAD-3DHeads validationはinterpolation candidateに対して評価していません。
- VGGHeadsのpose annotationは生成された推定値であり、motion-capture ground truthではありません。
- 姿勢分布監査はrotation matrixの幾何学的整合性と分布を確認していますが、rear画像とannotationの目視一致までは検証していません。
- AGORA-HPE rearの左右差について、原因は本実験だけでは確定していません。

したがって、α=0.25のbenchmark結果は保持と忘却の関係を確認する診断結果として扱い、同じheld-out benchmarkを反復的なハイパーパラメータ選択には使用しません。

## 結論

VGGHeadsはtrainの約93%がfrontで、rearは約1.86%です。VGGHeadsのみのfull fine-tuningではAGORA-HPE rearとAFLW2000に退行が生じました。

一方、base checkpointを75%、VGGHeads FT checkpointを25%としたweight-space interpolationでは、AGORA-HPE全体、AFLW2000、300W-LPの平均SO(3)誤差がすべてbaseより改善しました。AGORA-HPE rearもfull fine-tuningの44.820°から38.458°まで回復しました。

ただし、rearのbase 37.900°を下回るには至っていません。rear medianはbaseより改善している一方、P90と90°超率は悪化しており、特にnegative yaw側のrear binでtail errorが増加しています。

この結果から、既存データセットの能力保持とfine-tuningによる改善は両立可能である一方、後方帯の改善にはrear教師信号の強化と大誤差領域を対象とした追加対策が必要であることが示されています。

## 保存された成果物

再確認に使用する主要ファイルは次のとおりです。

- `experiments/runs/vgg_pose_distribution/config.json`
- `experiments/runs/vgg_pose_distribution/provenance.json`
- `experiments/runs/vgg_pose_distribution/metrics/pose_distribution.json`
- `experiments/runs/weight_interp_a025/config.json`
- `experiments/runs/weight_interp_a025/provenance.json`
- `experiments/runs/weight_interp_a025/status.json`
- `eval/weight_interp_a025/summary.json`
- `eval/weight_interp_a025/datasets/agora_hpe/metrics/yaw_bands.json`
- `eval/weight_interp_a025/datasets/agora_hpe/metrics/yaw_bins.json`
- `eval/comparisons/baseline_vs_weight_interp_a025/comparison.csv`
