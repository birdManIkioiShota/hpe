# YawPose後方yaw・15度層別・single-draw実験結果

## 概要

本実験では、Head Pose Estimation（HPE）の全周推定モデルSixDRepNet360-ResNet50に対して、YawPoseの後方合成画像をyaw-only教師信号として追加したときの効果を再評価しました。YawPoseはPINTO0309/YawNetで公開されている合成データセットで、yawを中心に構成され、一部にpitch情報を持ちますがroll情報は持ちません。本実験ではYawPoseのpitchとrollを使用せず、yawだけを教師信号として使用します。

先行実験は`docs/experiments/yawpose_rear_yaw_reliability_results.md`で報告しています。先行実験では、14,049件の後方候補全体で信頼度順位を計算し、YawPoseの使用回数を約451,000件/エポックへ固定していました。そのため、採用率を変えると信頼度だけでなく、yaw帯の構成と1画像あたりの反復回数も同時に変化していました。

今回の実験では、後方候補をsigned yawの15度単位で8区分へ分け、各区分内で独立して信頼度順位を計算しました。選択したYawPoseの各サンプルは1エポックにつき1回だけ学習に使用し、採用集合が小さい条件でも同じサンプルを繰り返して使用回数を揃えません。本レポートでは、この方式を`single-draw`と呼びます。

既存HPE学習データにはVGGHeadsを全条件で使用し、DAD-3DHeads trainを使用しない系列と使用する系列を比較しました。主結果では、12条件を同一の外部評価表にまとめています。

## 目的

本実験では、次の3点を確認します。

1. YawPoseの信頼度採用率による性能変化を、後方yaw帯の構成を揃え、同一サンプルのエポック内反復を除いた条件で再評価します。
2. AGORA-HPEを全周15度単位で評価し、後方平均だけでは確認できない角度依存の変化を測定します。
3. VGGHeadsを固定して使用し、DAD-3DHeads trainの有無による性能差を比較します。

DAD-3DHeadsは非商用条件を含むため、最終的な学習構成から除外することを前提としています。

## 使用データ

### YawPose

本実験では、`PINTO0309/YawNet`の`resources` releaseで公開されているYawPoseを使用します。使用データのprovenanceは次のとおりです。

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

既存HPE教師にはVGGHeadsとDAD-3DHeads trainを使用します。両データセットではrotation matrixを教師としてSO(3) supervised lossを計算します。

| データ | train | development |
|---|---:|---:|
| VGGHeads | 417,055 | 51,914 |
| DAD-3DHeads train由来split | 34,035 | checkpoint選択には不使用 |

`vgg_only`系列ではVGGHeads trainだけを使用し、`vgg_plus_dad`系列ではVGGHeads trainとDAD-3DHeads trainを同じ学習poolへ入れます。両系列とも既存HPEデータの使用数を417,024件/エポックへ固定するため、DAD-3DHeadsを追加してもoptimizer update数は増えません。

checkpoint選択には両系列ともVGGHeads dev 51,914件を使用します。

## 評価指標

モデルの主要出力は3×3 rotation matrixです。予測rotationとGT rotationの差はSO(3) geodesic errorで評価し、値が小さいほど誤差が小さいことを表します。

後方yawの評価では、予測rotation `R`から次式でhead-forward yawを求めます。

`yaw_head_forward = atan2(R[0,2], R[2,2])`

AGORA-HPEではmanifestに保存されたsource yawをyaw評価のGTとして使用し、予測yawとの円周距離をhead-forward yaw errorとして計算します。

姿勢帯は次の境界で区分します。

| 姿勢帯 | 定義 |
|---|---|
| front | `|yaw| < 60°` |
| side | `60° <= |yaw| < 120°` |
| rear | `|yaw| >= 120°` |
| rear-near | `120° <= |yaw| < 150°` |
| rear-deep | `150° <= |yaw| <= 180°` |

内部VGGHeads devではrotation matrixから求めたhead-forward azimuthを姿勢区分に使用し、外部評価ではevaluation manifestのsource yawを姿勢区分に使用します。

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

各損失の適用対象は次のとおりです。

| 損失 | 適用対象 | 内容 |
|---|---|---|
| `L_supervised` | 既存HPEデータ全体 | GT rotation matrixに対するSO(3) geodesic loss |
| `L_distillation` | 既存HPEデータの非rear | 初期checkpointのpredictionを保持するSO(3) distillation loss |
| `L_flip` | 既存HPEデータのrear | horizontal flip前後のpredictionを元座標系で一致させるSO(3) consistency loss |
| `L_yawpose` | 選択したYawPose rear | canonical yawに対する周期的yaw absolute error |

`Y_base`ではYawPoseを使用しないため、`L_yawpose`を0とします。外部評価には全条件のepoch 10 checkpointを使用します。

## YawPoseの選別方法

### canonical yawと後方候補

YawPoseのcanonical yawは、`labels_fixed.jsonl`のyawを次式でsigned yawへ変換した値です。

`yaw_signed = ((yaw_deg + 180) mod 360) - 180`

`manual_corrections.jsonl`に修正値が存在する場合は`corrected_yaw`で上書きします。今回の14,049件の後方候補にはmanual correction対象が含まれていないため、実際に使用したcanonical yawは`labels_fixed.jsonl`に由来します。

canonical yawが`|yaw| >= 120°`のサンプルだけを後方候補とし、次の8区分へ分けます。

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

### teacher model

信頼度計算には次の3モデルを使用します。

| ID | モデル |
|---|---|
| `sixdrepnet360_base` | 学習開始時のSixDRepNet360-ResNet50 |
| `semiuhpe_effnetv2s` | SemiUHPE EfficientNetV2-S |
| `whenet` | WHENet |

teacher modelはcanonical yawの置き換えには使用しません。各モデルのyaw予測を共通の`[-180°, 180°)`規約へ揃え、canonical yawとの一致度とteacher model同士の一致度を信頼度計算に使用します。

### 信頼度スコア

角度`a`と`b`の円周距離は次式で計算します。

`d_circ(a,b) = |((a - b + 180) mod 360) - 180|`

各サンプルについて、3 teacherとcanonical yawの誤差から`gt_median_error_deg`と`gt_max_error_deg`を求め、teacher間の全pairの円周距離から`teacher_dispersion_deg`を求めます。

各15度yaw帯の内部で、この3指標をそれぞれ0から1のpercentile rankへ変換し、その平均を信頼度スコアとします。

`reliability_score = (r_median + r_dispersion + r_max) / 3`

スコアが小さいほど、同じ15度yaw帯の中でcanonical yawとteacher群の整合度が相対的に高いサンプルです。この値はラベルが正しい確率ではありません。

同一スコアの場合は、`gt_median_error_deg`、`teacher_dispersion_deg`、`gt_max_error_deg`、`instance_id`の順で順位を確定します。

### 採用集合

各15度yaw帯の内部で信頼度スコア順に並べ、各帯から同じ割合を採用します。

| 条件 | 各15度帯からの採用率 | 採用数 |
|---|---:|---:|
| top20 | 20% | 2,812 |
| top40 | 40% | 5,623 |
| top60 | 60% | 8,433 |
| top80 | 80% | 11,244 |
| top100 | 100% | 14,049 |

各帯の内部では`top20 ⊂ top40 ⊂ top60 ⊂ top80 ⊂ top100`の関係を維持します。これにより、採用率を変更してもrear-nearとrear-deepの構成比が大きく変わらないようにしています。

生成元の構成は採用率によって変化するため、信頼度採用率と生成元の影響は完全には分離されていません。

## YawPoseの学習への投入方法

選択したYawPoseの各サンプルは、1エポックにつき1回だけ学習に使用します。各エポックの開始時に使用順を固定乱数で入れ替え、同じエポック内で同一サンプルを繰り返し使用しません。

1エポックあたりのYawPose使用数は、top20で2,812件、top40で5,623件、top60で8,433件、top80で11,244件、top100で14,049件です。採用率を増やすと、1エポックあたりのYawPose教師信号の総量も増加します。

## 実験条件

YawPoseを使用しないcontrolと5段階の採用率を、VGGHeads-onlyとVGGHeads+DAD-3DHeadsの2系列で実行しました。

| 系列 | 条件 |
|---|---|
| VGGHeads-only | `Y_base_vgg`、`Y_top20_vgg`、`Y_top40_vgg`、`Y_top60_vgg`、`Y_top80_vgg`、`Y_top100_vgg` |
| VGGHeads+DAD-3DHeads | `Y_base_vgg_dad`、`Y_top20_vgg_dad`、`Y_top40_vgg_dad`、`Y_top60_vgg_dad`、`Y_top80_vgg_dad`、`Y_top100_vgg_dad` |

VGGHeads devによるcheckpoint選択では、VGGHeads-only系列の全条件がepoch 8、VGGHeads+DAD-3DHeads系列の全条件がepoch 10でbestとなりました。主結果では条件間のcheckpoint選択時期を揃えるため、すべてepoch 10を評価しています。

## 外部評価条件

主評価にはAGORA-HPE、AFLW2000、300W-LP、DAD-3DHeads official validationを使用します。外部評価データは学習やcheckpoint選択には使用しません。

| 項目 | 設定 |
|---|---|
| inference precision | FP32 |
| AMP | 無効 |
| batch size | 256 |
| metric dtype | FP64 |
| deterministic algorithms | 有効 |
| crop | evaluation manifestのGT crop |

AGORA-HPE、AFLW2000、300W-LPのSO(3)評価では各データセットのGT rotation matrixを正解として使用します。DAD-3DHeads validationも同様にGT rotation matrixで評価します。

AGORA-HPE rear yawでは、manifestのsource yawが`|yaw| >= 120°`の2,644件を対象とし、source yawを正解としてhead-forward yaw errorを計算します。

保存済みbootstrap CIは各candidateと初期checkpointの差に対する区間です。同系列の`Y_base`と`Y_top20`〜`Y_top100`の差には直接CIを計算していません。

## 結果

### 主結果

12条件の外部評価結果を次の表にまとめます。AGORA、AFLW2000、300W-LP、DAD valはSO(3) geodesic meanで、AGORA rear yawだけはhead-forward yaw meanです。すべて値が小さいほど誤差が小さいことを表します。

| 条件 | AGORA SO(3) | AGORA rear yaw | AFLW2000 SO(3) | 300W-LP SO(3) | DAD val SO(3) |
|---|---:|---:|---:|---:|---:|
| 初期checkpoint | 45.475° | 30.800° | 6.562° | 5.845° | 31.482° |
| Y_base_vgg | 44.498° | 27.333° | 6.541° | 5.809° | 29.497° |
| Y_top20_vgg | 44.494° | 27.337° | 6.543° | 5.807° | 29.487° |
| Y_top40_vgg | 44.482° | 27.325° | 6.547° | 5.804° | 29.481° |
| Y_top60_vgg | 44.460° | 27.307° | 6.548° | 5.802° | 29.498° |
| Y_top80_vgg | 44.448° | 27.303° | 6.547° | 5.806° | 29.491° |
| Y_top100_vgg | 44.471° | 27.301° | 6.549° | 5.800° | 29.506° |
| Y_base_vgg_dad | 44.415° | 27.775° | 6.535° | 5.803° | 29.460° |
| Y_top20_vgg_dad | 44.413° | 27.771° | 6.536° | 5.804° | 29.454° |
| Y_top40_vgg_dad | 44.404° | 27.772° | 6.539° | 5.799° | 29.452° |
| Y_top60_vgg_dad | 44.388° | 27.758° | 6.541° | 5.794° | 29.454° |
| Y_top80_vgg_dad | 44.377° | 27.758° | 6.541° | 5.795° | 29.453° |
| Y_top100_vgg_dad | 44.392° | 27.765° | 6.541° | 5.798° | 29.450° |

YawPose追加による同系列の`Y_base`からの変化は小さく、VGGHeads-onlyのAGORA rear yawではtop80が-0.030°、top100が-0.032°です。VGGHeads+DAD-3DHeadsではtop80が-0.018°、top100が-0.010°です。

### AGORA-HPEの15度yaw帯

全rear平均では小さい変化しか見えないため、AGORA-HPEをsource yawで15度ごとに分割し、各区間のGT rotation matrixに対するSO(3) geodesic meanを確認しました。代表として、両系列で変化が比較的大きかったtop80と各系列の`Y_base`との差を示します。

| yaw帯 | VGGHeads-only | VGGHeads+DAD |
|---|---:|---:|
| -180〜-165° | +0.123° | +0.105° |
| -165〜-150° | -0.001° | +0.004° |
| -150〜-135° | -0.105° | -0.079° |
| -135〜-120° | -0.223° | -0.175° |
| +120〜+135° | -0.189° | -0.155° |
| +135〜+150° | -0.081° | -0.075° |
| +150〜+165° | -0.000° | +0.010° |
| +165〜+180° | +0.114° | +0.099° |

負値は`Y_base`より誤差が小さく、正値は誤差が大きいことを表します。両系列とも120–150°付近では誤差が低下し、150–165°付近ではほぼ変化せず、165–180°付近では誤差が増加しています。

## 考察

### 先行実験との差

先行実験ではYawPoseの使用回数を約451,000件/エポックへ固定していたため、top20では1画像あたり平均約160.5回/エポック、top100でも約32.1回/エポック提示していました。また、rear全体で信頼度順位を計算していたため、top20の97.6%が120–150°へ集中していました。

AGORA-HPE rear yawの`Y_base → Y_top80`は、先行実験では約-0.334°でしたが、今回のVGGHeads-onlyでは約-0.030°、VGGHeads+DAD-3DHeadsでは約-0.018°でした。先行実験で観測されたAFLW2000の0.3〜0.5°規模の悪化も、今回は再現していません。

今回の実験では反復回数とyaw帯ごとの選別方法を同時に変更しているため、差の縮小をどちらか一方だけへ帰属することはできません。

### 信頼度採用率

top20が他の採用率より一貫して低い誤差を示す結果にはなっていません。AGORA SO(3)では両系列ともtop60〜80付近まで誤差が小さくなり、top100で一部戻っています。

ただし、採用率を増やすと1エポックあたりのYawPose使用数も増えるため、採用率間の差には信頼度閾値と教師信号量の両方が含まれます。また、15度yaw帯の比率は揃えていますが、生成元の構成は採用率によって変化します。

### DAD-3DHeadsの影響

YawPoseを使用しない`Y_base`同士を主結果表で比較すると、DAD-3DHeadsを含む系列はAGORA、AFLW2000、300W-LP、DAD valのSO(3) meanがわずかに低い一方、AGORA rear yawはVGGHeads-onlyより0.442°高くなっています。

AGORA-HPEの15度集計では、DAD追加の影響もyaw帯によって異なります。例えば`Y_base_vgg_dad - Y_base_vgg`は-105〜-90°で約-1.078°、-90〜-75°で約-0.993°ですが、+135〜+150°で約+0.609°、+150〜+165°で約+0.895°、+165〜+180°で約+0.960°です。

したがって、DAD-3DHeadsによるoverall SO(3)の変化は全角度帯の一様な変化ではありません。

### teacher ensemble

信頼度スコアにはSixDRepNet360 base、SemiUHPE、WHENetの3 teacher modelを使用しています。yaw符号較正後の方向検証集合に対する平均誤差は、WHENetが約29.677°、SemiUHPEが約122.134°です。

SemiUHPEはYawPose後方domainに対する絶対誤差が大きく、`gt_max_error_deg`を含む信頼度スコアへ影響します。本実験ではteacher構成を変更していないため、個々のteacher modelが順位へ与えた影響は分離していません。

## 制約

本実験の解釈には次の制約があります。

- 学習はseed 42の1回だけであり、seed間分散を測定していません。
- 同系列の`Y_base`とYawPose条件の差には直接paired bootstrap CIを計算していないため、0.01〜0.05°規模の差について統計的な方向は確定していません。
- YawPoseの採用率と1エポックあたりのYawPose教師信号量が連動しています。
- 15度yaw帯内の採用率は固定していますが、生成元の構成は採用率によって変化します。
- 信頼度スコアはラベル正解確率ではなく、同一15度yaw帯内の相対順位です。
- teacher ensembleにはYawPose後方domainとの整合度が低いSemiUHPEが含まれます。
- DAD有無の比較では既存HPEデータの使用数を固定しているため、DADあり系列ではVGGHeadsの一部がDAD-3DHeadsへ置き換わります。
- DAD-3DHeads validationのrearは47件であり、細かなrear角度帯を安定して評価できる件数ではありません。

## 結論

今回のsingle-draw条件では、YawPose追加による全体平均とrear平均の変化は先行実験より小さくなりました。AGORA rear yawの`Y_base → Y_top80`は、VGGHeads-onlyで約-0.030°、VGGHeads+DAD-3DHeadsで約-0.018°です。

一方、15度単位では両系列に共通して120–150°付近の誤差低下と165–180°付近の誤差増加が観測されました。YawPoseの影響はrear全体へ一様に現れるのではなく、yaw帯によって方向が異なっています。

信頼度上位20%だけが他の採用率より明確に低い誤差を示す関係は確認されませんでした。また、DAD-3DHeadsを使用しない系列でもYawPoseによる角度依存の変化は観測されています。

## 保存成果物

本レポートの根拠となる主要成果物は次のとおりです。

- 実験計画は`docs/experiments/yawpose_rear_yaw_stratified_single_draw_plan.md`です。
- 先行YawPose実験結果は`docs/experiments/yawpose_rear_yaw_reliability_results.md`です。
- reliability設定は`experiments/runs/yawpose_reliability_stratified15/config.json`です。
- reliability集計は`experiments/runs/yawpose_reliability_stratified15/metrics/summary.json`です。
- reliability sample-level結果は`experiments/runs/yawpose_reliability_stratified15/predictions/reliability.jsonl`です。
- 12条件の内部評価は`experiments/runs/yawpose_rear_stratified_search/`です。
- 条件別外部評価は`eval/yawpose_rear_stratified_search/conditions/`です。
- head-forward yaw比較は`eval/yawpose_rear_stratified_search/yaw_comparisons/`です。
- AGORA-HPE、AFLW2000、300W-LPのyaw集計は`eval/yawpose_rear_stratified_search/yaw_summary_baseline_fp32_final.json`です。
- DAD-3DHeads validationのyaw集計は`eval/yawpose_rear_stratified_search/yaw_summary_baseline_dad_fp32_final__dad.json`です。
