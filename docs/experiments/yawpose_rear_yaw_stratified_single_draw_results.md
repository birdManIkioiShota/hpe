# YawPose後方yaw・15度層別・1エポック1回投入実験結果

## 概要

本実験では、Head Pose Estimation（HPE）の全周推定モデルSixDRepNet360-ResNet50に対して、YawPoseの後方合成画像をyawだけの追加教師として使用したときの効果を再評価しました。

比較対象となる先行実験は`docs/experiments/yawpose_rear_yaw_reliability_results.md`です。先行実験では、後方候補14,049件をまとめて信頼度順に並べ、採用率にかかわらずYawPoseを1エポックあたり約451,000件使用していました。そのため、採用率を変えると、採用する画像の信頼度だけでなく、yaw帯の構成と同一画像の反復回数も同時に変化していました。

今回の実験では、後方候補を15度ごとの8区分に分け、各区分の中で信頼度順に並べます。各条件では、それぞれの区分から上位20%、40%、60%、80%、100%を採用します。また、採用した画像は1エポックにつき1回だけ学習に使用し、同じエポック内では繰り返し使用しません。

既存のHPE学習データにはVGGHeadsを必ず使用し、DAD-3DHeadsを加えない系列と加える系列を比較しました。

## 目的

本実験では、次の3点を確認します。

1. yaw帯の偏りと同一画像の反復を抑えた条件で、YawPoseの信頼度による選別が後方姿勢推定へ与える影響を確認します。
2. AGORA-HPEを15度ごとのyaw帯に分け、後方全体の平均だけでは見えない角度依存の変化を確認します。
3. VGGHeadsだけを使用した場合と、VGGHeadsにDAD-3DHeadsを加えた場合を比較します。

DAD-3DHeadsは非商用条件を含むため、最終的な学習構成から除外することを前提としています。

## 使用データ

### YawPose

YawPoseは`PINTO0309/YawNet`の`resources` releaseで公開されている合成データセットで、yawを中心に構成され、一部にpitch情報を持ちますがroll情報は持ちません。本実験ではYawPoseのpitchとrollを教師信号として使用せず、yawだけを使用します。使用した公開データは次のとおりです。

| 項目 | 値 |
|---|---|
| upstream repository | `PINTO0309/YawNet` |
| dataset documentation revision | `4af7fa9d73e94790688c518376d3be7c41c74cc3` |
| release | `resources` |
| archive | `yawpose.tar.gz` |
| archive SHA-256 | `08df8f2e5df8d2c5c0509475688cf1366c69b96b393f1cbbbf87aeb9ced9118a` |
| cleaned labels | `labels_fixed.jsonl`、42,135件 |
| image crops | 43,258件 |
| image size | 320×320 |

YawPoseの公開yawは`[0°, 360°)`です。本実験では`[-180°, 180°)`へ変換し、`|yaw| >= 120°`の14,049件だけを学習候補とします。

### 既存HPE学習データ

既存のHPE教師にはVGGHeadsとDAD-3DHeadsを使用します。両データセットではrotation matrixを教師としてSO(3) lossを計算します。

| データ | 学習用 | checkpoint選択用 |
|---|---:|---:|
| VGGHeads | 417,055 | 51,914 |
| DAD-3DHeads | 34,035 | 使用しない |

VGGHeads-only系列ではVGGHeadsだけを使用します。VGGHeads+DAD-3DHeads系列では、VGGHeadsとDAD-3DHeadsを同じ学習データ集合へ入れます。

両系列とも、既存HPEデータは417,024件/epochだけ使用します。そのため、DAD-3DHeadsを加えても1エポックあたりの更新回数は増えず、VGGHeadsの一部がDAD-3DHeadsへ置き換わる構成になります。

checkpoint選択には、両系列ともVGGHeads dev 51,914件を使用します。

## 評価指標

モデルの出力は3×3 rotation matrixです。姿勢全体の誤差にはSO(3) geodesic errorを使用し、値が小さいほどGT rotationに近いことを表します。

後方yawの評価では、予測rotation `R`から次式でhead-forward yawを求めます。

`yaw_head_forward = atan2(R[0,2], R[2,2])`

AGORA-HPEでは、manifestに保存されたsource yawをyaw評価の正解として使用し、予測したhead-forward yawとの円周角度差を求めます。

姿勢帯は次のように区分します。

| 姿勢帯 | 定義 |
|---|---|
| front | `|yaw| < 60°` |
| side | `60° <= |yaw| < 120°` |
| rear | `|yaw| >= 120°` |
| rear-near | `120° <= |yaw| < 150°` |
| rear-deep | `150° <= |yaw| <= 180°` |

VGGHeads devではrotation matrixから求めたhead-forward azimuthを姿勢帯の判定に使用します。AGORA-HPEなどの外部評価では、各manifestに保存されたsource yawを姿勢帯の判定に使用します。

## 学習条件

対象モデルはSixDRepNet360-ResNet50です。モデル構造は変更せず、ResNet50の`layer4`とrotation regression headを更新します。

| 項目 | 設定 |
|---|---|
| 初期checkpoint | `checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` |
| SHA-256 | `3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6` |
| epochs | 10 |
| batch size | 64 |
| gradient accumulation | 2 |
| precision | BF16 |
| backbone learning rate | `1e-6` |
| head learning rate | `3e-5` |
| weight decay | `1e-4` |
| warmup updates | 100 |
| gradient clip norm | 1.0 |
| seed | 42 |

総損失は次式で構成します。

`L_total = L_supervised + 1.0 L_distillation + 0.2 L_flip + 0.2 L_yawpose`

各損失の役割は次のとおりです。

| 損失 | 適用対象 | 内容 |
|---|---|---|
| `L_supervised` | 既存HPEデータ全体 | GT rotation matrixに対するSO(3) loss |
| `L_distillation` | 既存HPEデータの非rear | 初期checkpointの予測を保持するためのSO(3) distillation loss |
| `L_flip` | 既存HPEデータのrear | 水平反転前後の予測を元の座標系で一致させるSO(3) consistency loss |
| `L_yawpose` | 採用したYawPose後方画像 | 基準yawに対する周期的yaw absolute error |

`Y_base`ではYawPoseを使用しないため、`L_yawpose`を0とします。

## YawPoseの選別方法

### 基準yawと後方候補

YawPoseの公開yawは、次式で`[-180°, 180°)`へ変換します。

`yaw_signed = ((yaw_deg + 180) mod 360) - 180`

本レポートでは、この変換後に学習と選別の基準として使用するyawを基準yawと呼びます。`manual_corrections.jsonl`に修正値が存在する場合は、その値で上書きします。

今回使用した14,049件の後方候補にはmanual correction対象が含まれていないため、実際の基準yawはすべて`labels_fixed.jsonl`に由来します。

基準yawが`|yaw| >= 120°`の画像だけを後方候補とし、次の8区分へ分けます。

| yaw帯 |
|---|
| `[-180°, -165°)` |
| `[-165°, -150°)` |
| `[-150°, -135°)` |
| `[-135°, -120°]` |
| `[120°, 135°)` |
| `[135°, 150°)` |
| `[150°, 165°)` |
| `[165°, 180°)` |

### 教師モデル

信頼度計算には次の3モデルを使用します。

| ID | モデル |
|---|---|
| `sixdrepnet360_base` | 学習開始時のSixDRepNet360-ResNet50 |
| `semiuhpe_effnetv2s` | SemiUHPE EfficientNetV2-S |
| `whenet` | WHENet |

教師モデルの予測で基準yawを書き換えることはしません。各モデルのyaw予測を同じ`[-180°, 180°)`の規約へ揃え、基準yawとの一致度と教師モデル同士の一致度を信頼度計算に使用します。

### 信頼度スコア

角度`a`と`b`の円周距離は次式で計算します。

`d_circ(a,b) = |((a - b + 180) mod 360) - 180|`

各画像について、3教師モデルと基準yawの誤差から`gt_median_error_deg`と`gt_max_error_deg`を求め、教師モデル間の全組み合わせの円周距離から`teacher_dispersion_deg`を求めます。

各15度yaw帯の中で、この3指標をそれぞれ0から1のパーセンタイル順位へ変換し、その平均を信頼度スコアとします。

`reliability_score = (r_median + r_dispersion + r_max) / 3`

スコアが小さいほど、同じ15度yaw帯の中で基準yawと教師モデル群の整合度が相対的に高い画像です。この値は、ラベルが正しい確率を表すものではありません。

同一スコアの場合は、`gt_median_error_deg`、`teacher_dispersion_deg`、`gt_max_error_deg`、`instance_id`の順で順位を確定します。

### 採用集合

各15度yaw帯の中で信頼度スコア順に並べ、それぞれ同じ割合だけ採用します。

| 条件 | 各15度帯からの採用率 | 採用数 |
|---|---:|---:|
| top20 | 20% | 2,812 |
| top40 | 40% | 5,623 |
| top60 | 60% | 8,433 |
| top80 | 80% | 11,244 |
| top100 | 100% | 14,049 |

各yaw帯の中では`top20 ⊂ top40 ⊂ top60 ⊂ top80 ⊂ top100`の関係を維持します。これにより、採用率を変更してもrear-nearとrear-deepの比率が大きく変化しないようにしています。

生成元の構成は採用率によって変化するため、信頼度と生成元の影響は完全には分離されていません。

## YawPoseの学習への投入方法

採用したYawPoseの各画像は、1エポックにつき1回だけ学習に使用します。各エポックの開始時に使用順を固定乱数で入れ替え、同じエポック内で同一画像を繰り返し使用しません。

1エポックあたりのYawPose使用数は、top20で2,812件、top40で5,623件、top60で8,433件、top80で11,244件、top100で14,049件です。採用率を増やすと、1エポックあたりのYawPose教師信号の総量も増加します。

## 実験条件

YawPoseを使用しない基準条件と5段階の採用率を、VGGHeads-onlyとVGGHeads+DAD-3DHeadsの2系列で実行しました。

| 系列 | 条件 |
|---|---|
| VGGHeads-only | `Y_base_vgg`、`Y_top20_vgg`、`Y_top40_vgg`、`Y_top60_vgg`、`Y_top80_vgg`、`Y_top100_vgg` |
| VGGHeads+DAD-3DHeads | `Y_base_vgg_dad`、`Y_top20_vgg_dad`、`Y_top40_vgg_dad`、`Y_top60_vgg_dad`、`Y_top80_vgg_dad`、`Y_top100_vgg_dad` |

VGGHeads devによるcheckpoint選択では、VGGHeads-only系列の全条件がepoch 8、VGGHeads+DAD-3DHeads系列の全条件がepoch 10でbestとなりました。条件間の比較ではcheckpoint選択時期を揃えるため、すべてepoch 10を評価します。

## 評価条件

評価にはAGORA-HPE、AFLW2000、300W-LP、DAD-3DHeads official validationを使用します。これらは学習やcheckpoint選択には使用しません。

| 項目 | 設定 |
|---|---|
| inference precision | FP32 |
| AMP | 無効 |
| batch size | 256 |
| metric dtype | FP64 |
| deterministic algorithms | 有効 |
| crop | evaluation manifestのGT crop |

AGORA-HPE、AFLW2000、300W-LP、DAD-3DHeads validationの全体評価では、各データセットのGT rotation matrixを正解としてSO(3) geodesic errorを計算します。

AGORA-HPEの後方yaw評価では、manifestのsource yawが`|yaw| >= 120°`の2,644件だけを対象とし、source yawを正解としてhead-forward yaw errorを計算します。

保存済みのbootstrap CIは、各条件と初期checkpointとの差に対して計算したものです。同系列の`Y_base`と`Y_top20`〜`Y_top100`の差には直接CIを計算していません。

## 結果

### データセット全体のSO(3)誤差

各条件のepoch 10 checkpointを、AGORA-HPE、AFLW2000、300W-LP、DAD-3DHeads validationのGT rotation matrixに対して評価しました。表の値はSO(3) geodesic errorの平均であり、小さいほど誤差が小さいことを表します。

| 条件 | AGORA-HPE | AFLW2000 | 300W-LP | DAD validation |
|---|---:|---:|---:|---:|
| 初期checkpoint | 45.475° | 6.562° | 5.845° | 31.482° |
| Y_base_vgg | 44.498° | 6.541° | 5.809° | 29.497° |
| Y_top20_vgg | 44.494° | 6.543° | 5.807° | 29.487° |
| Y_top40_vgg | 44.482° | 6.547° | 5.804° | 29.481° |
| Y_top60_vgg | 44.460° | 6.548° | 5.802° | 29.498° |
| Y_top80_vgg | 44.448° | 6.547° | 5.806° | 29.491° |
| Y_top100_vgg | 44.471° | 6.549° | 5.800° | 29.506° |
| Y_base_vgg_dad | 44.415° | 6.535° | 5.803° | 29.460° |
| Y_top20_vgg_dad | 44.413° | 6.536° | 5.804° | 29.454° |
| Y_top40_vgg_dad | 44.404° | 6.539° | 5.799° | 29.452° |
| Y_top60_vgg_dad | 44.388° | 6.541° | 5.794° | 29.454° |
| Y_top80_vgg_dad | 44.377° | 6.541° | 5.795° | 29.453° |
| Y_top100_vgg_dad | 44.392° | 6.541° | 5.798° | 29.450° |

同系列の`Y_base`と比べたYawPose追加条件の差は、AGORA-HPEで最大約-0.050°、AFLW2000で約+0.001〜+0.008°、300W-LPで約-0.002〜-0.009°です。

### AGORA-HPEの後方yaw誤差

各条件のepoch 10 checkpointを、AGORA-HPEのうちsource yawが`|yaw| >= 120°`の2,644件に対して評価しました。正解にはsource yawを使用し、表の値はhead-forward yaw errorの平均です。

| 条件 | VGGHeads-only | VGGHeads+DAD-3DHeads |
|---|---:|---:|
| Y_base | 27.333° | 27.775° |
| Y_top20 | 27.337° | 27.771° |
| Y_top40 | 27.325° | 27.772° |
| Y_top60 | 27.307° | 27.758° |
| Y_top80 | 27.303° | 27.758° |
| Y_top100 | 27.301° | 27.765° |

初期checkpointの後方yaw誤差は30.800°です。同系列の`Y_base`から`Y_top80`への変化は、VGGHeads-onlyで-0.030°、VGGHeads+DAD-3DHeadsで-0.018°です。

### AGORA-HPEの15度yaw帯

AGORA-HPEをsource yawで15度ごとに分け、それぞれの区間でGT rotation matrixに対するSO(3) geodesic errorを計算しました。次の表は、各系列の`Y_top80`から`Y_base`を引いた差です。

| yaw帯 | VGGHeads-only | VGGHeads+DAD-3DHeads |
|---|---:|---:|
| -180〜-165° | +0.123° | +0.105° |
| -165〜-150° | -0.001° | +0.004° |
| -150〜-135° | -0.105° | -0.079° |
| -135〜-120° | -0.223° | -0.175° |
| +120〜+135° | -0.189° | -0.155° |
| +135〜+150° | -0.081° | -0.075° |
| +150〜+165° | -0.000° | +0.010° |
| +165〜+180° | +0.114° | +0.099° |

負値は`Y_base`より誤差が小さく、正値は誤差が大きいことを表します。両系列とも120〜150°付近では誤差が低下し、150〜165°付近ではほとんど変化せず、165〜180°付近では誤差が増加しています。

## 考察

### 先行実験との差

先行実験ではYawPoseを1エポックあたり約451,000件使用していたため、top20では同じ画像を1エポックあたり平均約160.5回、top100でも1エポックあたり平均約32.1回使用していました。また、後方候補全体をまとめて信頼度順に並べていたため、top20の97.6%が120〜150°へ集中していました。

今回の実験では、各15度yaw帯から同じ割合を採用し、各画像を1エポックにつき1回だけ使用しています。

AGORA-HPEの後方yaw誤差における`Y_base → Y_top80`の変化は、先行実験では約-0.334°でしたが、今回はVGGHeads-onlyで約-0.030°、VGGHeads+DAD-3DHeadsで約-0.018°でした。先行実験で観測されたAFLW2000の0.3〜0.5°規模の悪化も、今回は再現していません。

今回の再設計では、同一画像の反復回数とyaw帯ごとの選別方法を同時に変更しています。そのため、効果の縮小をどちらか一方だけへ帰属することはできません。

### 信頼度による選別

top20が他の採用率より一貫して低い誤差を示す結果にはなっていません。AGORA-HPEのSO(3)誤差では、両系列ともtop60〜80付近までわずかに低下し、top100で一部戻っています。

ただし、採用率を増やすと1エポックあたりのYawPose使用数も増えるため、採用率間の差には信頼度と教師信号量の両方が含まれます。また、15度yaw帯の比率は揃えていますが、生成元の構成は採用率によって変化します。

### DAD-3DHeadsの影響

YawPoseを使用しない`Y_base`同士を比較すると、DAD-3DHeadsを含む系列はAGORA-HPE、AFLW2000、300W-LP、DAD-3DHeads validationのSO(3)誤差がわずかに低い一方、AGORA-HPEの後方yaw誤差はVGGHeads-onlyより0.442°高くなっています。

AGORA-HPEの15度集計でも、DAD-3DHeads追加の影響はyaw帯によって異なります。`Y_base_vgg_dad - Y_base_vgg`は、-105〜-90°で約-1.078°、-90〜-75°で約-0.993°ですが、+135〜+150°で約+0.609°、+150〜+165°で約+0.895°、+165〜+180°で約+0.960°です。

したがって、DAD-3DHeadsを加えた場合の変化は、全角度帯で同じ方向に現れているわけではありません。

### 教師モデル

信頼度スコアにはSixDRepNet360 base、SemiUHPE、WHENetの3モデルを使用しています。yaw符号較正後の方向検証集合に対する平均誤差は、WHENetが約29.677°、SemiUHPEが約122.134°です。

SemiUHPEはYawPoseの後方画像に対する絶対誤差が大きく、`gt_max_error_deg`を含む信頼度スコアへ影響します。本実験では教師モデルの構成を変更していないため、各教師モデルが順位へ与えた影響は分離していません。

## 制約

本実験の解釈には次の制約があります。

- 学習はseed 42の1回だけであり、seed間のばらつきを測定していません。
- 同系列の`Y_base`とYawPose追加条件の差には直接paired bootstrap CIを計算していないため、0.01〜0.05°規模の差について統計的な方向は確定していません。
- YawPoseの採用率と1エポックあたりのYawPose教師信号量が連動しています。
- 15度yaw帯内の採用率は固定していますが、生成元の構成は採用率によって変化します。
- 信頼度スコアはラベル正解確率ではなく、同一15度yaw帯内の相対順位です。
- 教師モデルにはYawPoseの後方画像に対する誤差が大きいSemiUHPEが含まれます。
- DAD-3DHeadsの有無を比較するときも既存HPEデータの使用数は固定しているため、DAD-3DHeadsを加えた系列ではVGGHeadsの一部がDAD-3DHeadsへ置き換わります。
- DAD-3DHeads validationの後方画像は47件であり、細かな後方yaw帯を安定して評価できる件数ではありません。

## 結論

各15度yaw帯で信頼度順に選別し、採用したYawPose画像を1エポックにつき1回だけ使用した今回の実験では、YawPose追加による全体平均と後方平均の変化は先行実験より小さくなりました。AGORA-HPEの後方yaw誤差における`Y_base → Y_top80`の変化は、VGGHeads-onlyで約-0.030°、VGGHeads+DAD-3DHeadsで約-0.018°です。

一方、15度単位では両系列に共通して120〜150°付近の誤差低下と165〜180°付近の誤差増加が観測されました。YawPoseの影響は後方全体へ一様に現れるのではなく、yaw帯によって方向が異なっています。

信頼度上位20%だけが他の採用率より明確に低い誤差を示す関係は確認されませんでした。また、DAD-3DHeadsを使用しない系列でもYawPoseによる角度依存の変化は観測されています。

## 保存成果物

本レポートの根拠となる主要成果物は次のとおりです。

- 実験計画は`docs/experiments/yawpose_rear_yaw_stratified_single_draw_plan.md`です。
- 先行YawPose実験結果は`docs/experiments/yawpose_rear_yaw_reliability_results.md`です。
- 信頼度計算の設定は`experiments/runs/yawpose_reliability_stratified15/config.json`です。
- 信頼度の集計結果は`experiments/runs/yawpose_reliability_stratified15/metrics/summary.json`です。
- 画像ごとの信頼度は`experiments/runs/yawpose_reliability_stratified15/predictions/reliability.jsonl`です。
- 学習条件ごとの結果は`experiments/runs/yawpose_rear_stratified_search/`です。
- 外部評価結果は`eval/yawpose_rear_stratified_search/conditions/`です。
- head-forward yawの比較結果は`eval/yawpose_rear_stratified_search/yaw_comparisons/`です。
- AGORA-HPE、AFLW2000、300W-LPのyaw集計は`eval/yawpose_rear_stratified_search/yaw_summary_baseline_fp32_final.json`です。
- DAD-3DHeads validationのyaw集計は`eval/yawpose_rear_stratified_search/yaw_summary_baseline_dad_fp32_final__dad.json`です。
