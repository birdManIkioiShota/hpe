# YawPose後方yaw・15度層別single-draw実験結果

## 概要

本実験は、Head Pose Estimation（HPE）の全周推定モデルSixDRepNet360-ResNet50に対して、YawPoseの後方合成画像をyaw-only教師信号として追加したときの効果を再評価したものです。YawPoseはPINTO0309/YawNetで公開されている合成データセットで、yawを中心に構成され、一部にpitch情報を持ちますがroll情報は持ちません。本実験ではYawPose由来のpitchとrollを教師信号として使用せず、canonical yawだけを使用します。

先行実験は`docs/experiments/yawpose_rear_yaw_reliability_results.md`で報告しています。先行実験では、14,049件のrear candidate全体でreliability rankingを計算し、YawPose総draw数を約451,000件/epochへ固定していました。そのため、採用率を変えるとreliabilityだけでなく、yaw角度分布と1画像あたりの反復回数も同時に変化していました。

今回の実験では、rear candidateをsigned yawの15度単位で8区分へ分け、各区分内で独立してreliability rankingを計算しました。また、選択したYawPose sampleは各epochで1回だけdrawし、subsetが小さい条件でも反復して総draw数を揃えません。本レポートでは、このsampling方式を`single-draw`と呼びます。

既存HPE学習データについてはVGGHeadsを全条件で使用し、DAD-3DHeads trainを使用しない系列と使用する系列を分けました。固定epoch 10 checkpointによる比較では、YawPose追加による平均性能の変化は先行実験より小さくなりました。一方、AGORA-HPEの15度yaw帯では、120–150°付近で誤差が低下し、165–180°付近で誤差が増加する角度依存の変化が、両系列で共通して観測されました。

## 目的

本実験の検証対象は次の3点です。

1. YawPoseのreliability採用率による性能変化を、rear内のyaw帯構成を揃え、同一sampleのepoch内反復を除いた条件で再評価します。
2. AGORA-HPEを全周15度単位で評価し、rear平均だけでは確認できない角度依存の変化を測定します。
3. VGGHeadsを固定して使用し、DAD-3DHeads trainの有無による性能差を比較します。

DAD-3DHeadsは非商用条件を含むため、最終的な学習構成から除外することを前提としています。本実験では、DAD-3DHeadsを学習に使用しない系列でも後方性能を維持できるかを確認します。

## データ

### YawPose

本実験で使用するYawPoseは、`PINTO0309/YawNet`の`resources` releaseで公開されている合成頭部姿勢データです。使用した公開データのprovenanceは次のとおりです。

| 項目 | 値 |
|---|---|
| upstream repository | `PINTO0309/YawNet` |
| dataset documentation revision | `4af7fa9d73e94790688c518376d3be7c41c74cc3` |
| release | `resources` |
| archive | `yawpose.tar.gz` |
| archive SHA-256 | `08df8f2e5df8d2c5c0509475688cf1366c69b96b393f1cbbbf87aeb9ced9118a` |
| published cleaned labels | `labels_fixed.jsonl`、42,135件 |
| image crops | 43,258件 |
| image size | 320×320 |

YawPoseの公開yawは`[0°, 360°)`です。本実験ではこれを`[-180°, 180°)`へ変換し、`|yaw| >= 120°`のsampleだけを学習候補とします。YawPoseから使用する教師信号はyawだけであり、pitchとrollは使用しません。

### Existing-HPE data

既存HPE教師にはVGGHeadsとDAD-3DHeads trainを使用します。両データセットでは完全なrotation matrixを教師としてSO(3) supervised lossを計算します。

| データ | train | development |
|---|---:|---:|
| VGGHeads | 417,055 | 51,914 |
| DAD-3DHeads train由来split | 34,035 | 本実験のcheckpoint選択には不使用 |

`vgg_only`系列ではVGGHeads trainだけを使用します。`vgg_plus_dad`系列ではVGGHeads trainとDAD-3DHeads trainを学習poolとして使用します。

両系列のexisting-HPE draw数は417,024件/epochへ固定します。この値はVGGHeads train件数をeffective batch size 128の境界へ切り下げた値です。そのため、DAD-3DHeadsを追加した系列ではtraining poolは増えますが、optimizer update数は増えません。

checkpoint selectionに使用するdevelopment splitは両系列ともVGGHeads dev 51,914件へ固定します。したがって、DAD有無の比較は、固定training budgetと固定development集合のもとで学習poolだけを変更した比較です。

## 用語と評価指標

### yaw規約

YawPoseの公開yawは次式でsigned yawへ変換します。

`yaw_signed = ((yaw_deg + 180) mod 360) - 180`

値域は`[-180°, 180°)`です。0°は正面、+90°と-90°は左右の側面、±180°付近は完全後方に相当します。

### 姿勢区分

内部development評価では、rotation matrixからhead-localの`+Z`軸をhead-forward方向として取り出し、そのXZ平面上のazimuthを姿勢区分に使用します。外部評価では、各evaluation manifestに保存されたsource yawを姿勢区分に使用します。

姿勢帯の境界は次のとおりです。

| 姿勢帯 | 定義 |
|---|---|
| front | `|yaw| < 60°` |
| side | `60° <= |yaw| < 120°` |
| rear | `|yaw| >= 120°` |
| rear-near | `120° <= |yaw| < 150°` |
| rear-deep | `150° <= |yaw| <= 180°` |

内部developmentのazimuthと外部評価のsource yawは同一のデータ源ではないため、個々のsampleについて同じ数値量として直接比較しません。姿勢帯の意味を揃えた集計単位として使用します。

### SO(3) geodesic error

モデルの主要出力は3×3 rotation matrixです。予測rotationと正解rotationの回転差をSO(3) geodesic errorとして度数で評価します。値が小さいほど正解rotationへ近いことを表します。

本レポートでは主にmean、median、P90、90°超過率を使用します。YawPoseはyaw-only教師信号であるため、SO(3)に加えてyaw誤差も確認します。

### head-forward yaw error

rotation matrixからhead-forward yawを求める場合は、予測rotation `R`に対して次式を使用します。

`yaw_head_forward = atan2(R[0,2], R[2,2])`

予測yawとGT source yawの差は360°周期を考慮した円周距離で評価します。本レポートでは、この指標をhead-forward yaw errorと呼びます。

### Euler yaw error

通常のexternal evaluationでは、targetとpredictionのrotation matrixをcanonical `RzRyRx` Euler表現へ変換したaxis errorも保存します。このEuler yaw errorとhead-forward yaw errorは定義が異なるため、同一指標として扱いません。

## 対象モデルと学習方法

対象モデルはSixDRepNet360-ResNet50です。モデル構造は変更せず、ResNet50の`layer4`とrotation regression headを更新します。

| 項目 | 設定 |
|---|---|
| 初期checkpoint | `checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` |
| SHA-256 | `3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6` |
| update scope | ResNet50 `layer4` + regression head |
| epochs | 10 |
| existing-HPE batch size | 64 |
| YawPose batch size | 64 |
| gradient accumulation | 2 |
| effective existing-HPE batch | 128 |
| precision | BF16 |
| backbone learning rate | `1e-6` |
| head learning rate | `3e-5` |
| weight decay | `1e-4` |
| warmup updates | 100 |
| gradient clip norm | 1.0 |
| seed | 42 |

総損失は次の4項から構成します。

`L_total = L_supervised + 1.0 L_distillation + 0.2 L_flip + 0.2 L_yawpose`

各損失項の適用対象は次のとおりです。

| 損失 | 適用対象 | 内容 |
|---|---|---|
| `L_supervised` | existing-HPE data全体 | GT rotation matrixに対するSO(3) geodesic loss |
| `L_distillation` | existing-HPE dataの非rear | 初期checkpointのpredictionを保持するSO(3) distillation loss |
| `L_flip` | existing-HPE dataのrear | horizontal flip前後の予測を元座標系で一致させるSO(3) consistency loss |
| `L_yawpose` | 選択されたYawPose rear | canonical yawに対する周期的yaw absolute error |

`Y_base`ではYawPose streamを使用しないため、`L_yawpose`のweightを0とします。その他の主要学習条件は同系列内で固定します。

主比較には全条件の固定epoch 10 checkpointを使用します。内部best checkpointも保存しますが、条件によって異なるcheckpoint選択時期を主比較へ混ぜないため、外部比較には固定epochを使用します。

## YawPoseデータ選別

### canonical yawとrear candidate

YawPoseのcanonical yawは、公開`labels_fixed.jsonl`のyawをsigned yawへ変換した値です。`manual_corrections.jsonl`に同一画像の修正が存在する場合だけ、`corrected_yaw`で上書きします。

今回のrear candidate 14,049件では、保存されたsubset summary上の`human_corrected`はすべて0件です。したがって、今回使用したcandidateのcanonical yawは公開`labels_fixed.jsonl`に由来します。

teacher予測はcanonical yawの置換には使用しません。canonical yawが`|yaw| >= 120°`のsampleだけをrear candidateとします。

### 15度rear区分

rear candidateをsigned yawで次の8区分へ分けます。

| 区分 | yaw範囲 |
|---|---|
| negative -180〜-165 | `[-180°, -165°)` |
| negative -165〜-150 | `[-165°, -150°)` |
| negative -150〜-135 | `[-150°, -135°)` |
| negative -135〜-120 | `[-135°, -120°]` |
| positive 120〜135 | `[120°, 135°)` |
| positive 135〜150 | `[135°, 150°)` |
| positive 150〜165 | `[150°, 165°)` |
| positive 165〜180 | `[165°, 180°)` |

signed yawでは180°が-180°へwrapされるため、180°相当はnegative側の最深部区分へ入ります。

### teacher ensemble

reliability事前計算には次の3 teacherを使用します。

| teacher ID | モデル |
|---|---|
| `sixdrepnet360_base` | 学習開始時のSixDRepNet360-ResNet50 |
| `semiuhpe_effnetv2s` | SemiUHPE EfficientNetV2-S |
| `whenet` | WHENet |

SemiUHPEとWHENetのpredictionには、teacher ID、固定implementation revision、checkpoint SHA-256、candidate manifest SHA-256、yaw符号較正結果を記録したprovenance metadataを付属させています。reliability事前計算では、このmetadataとprediction内容を検証してから使用します。

yaw符号較正には、YawPose生成時の方向情報が検証済みである`intent_s004`と`intent_s005`のsample群を使用しています。これはSemiUHPEとWHENetの出力方向を共通yaw規約へ合わせるための検証集合であり、reliability rankingの教師ラベルとしては使用しません。

teacherごとのyawは共通の`[-180°, 180°)`規約へ変換してから比較します。

### 円周距離

yaw差は通常の絶対差ではなく円周距離で計算します。角度`a`と`b`の差は次式で`[-180°, 180°)`へwrapし、その絶対値を使用します。

`d_circ(a,b) = |((a - b + 180) mod 360) - 180|`

例えば179°と-179°の距離は358°ではなく2°です。

### reliability指標

teacher `i`の予測yawを`t_i`、canonical yawを`y`とすると、各teacherとcanonical yawの誤差を次式で求めます。

`e_i = d_circ(t_i, y)`

各sampleについて使用するreliability指標は次のとおりです。

| 指標 | 定義 |
|---|---|
| `gt_median_error_deg` | 3 teacherとcanonical yawの誤差`e_i`のmedian |
| `gt_max_error_deg` | 3 teacherとcanonical yawの誤差`e_i`の最大値 |
| `teacher_dispersion_deg` | 3 teacher間の全pairに対する円周距離のmedian |

指標名の`gt`は実装上の名称であり、本実験ではcanonical yawを参照しています。teacher predictionをpseudo-GTへ置き換える処理は行いません。

`agree_10`、`agree_20`、`agree_30`としてcanonical yawから10°、20°、30°以内に入るteacher比率も保存しますが、reliability scoreには使用しません。

### 15度帯内percentileとscore

先行実験ではrear candidate 14,049件全体でpercentile rankを計算していました。今回は8個の15度yaw区分ごとに独立してpercentile rankを計算します。

各区分の内部で`gt_median_error_deg`、`teacher_dispersion_deg`、`gt_max_error_deg`をそれぞれ0から1のpercentile rankへ変換し、`r_median`、`r_dispersion`、`r_max`とします。値が小さいほどrankも小さくし、同値にはaverage rankを使用します。

reliability scoreは次式です。

`reliability_score = (r_median + r_dispersion + r_max) / 3`

scoreが小さいほど、同じ15度yaw帯の中でteacher群とcanonical yawの整合度が相対的に高いsampleです。このscoreはラベルが正しい確率ではなく、同一15度帯内の相対順位です。

同一scoreが存在する場合は、`gt_median_error_deg`、`teacher_dispersion_deg`、`gt_max_error_deg`、`instance_id`の順でtie-breakします。

### subset生成

各15度区分の内部でreliability scoreを昇順に並べ、採用率ごとに上位sampleを選択します。100%以外では各区分について`ceil(N_bin × ratio)`件を採用します。

| subset | 各15度区分からの採用率 | 全体件数 |
|---|---:|---:|
| `top020` | 20% | 2,812 |
| `top040` | 40% | 5,623 |
| `top060` | 60% | 8,433 |
| `top080` | 80% | 11,244 |
| `top100` | 100% | 14,049 |

各15度区分の内部では`top20 ⊂ top40 ⊂ top60 ⊂ top80 ⊂ top100`のnested関係を維持します。これにより、採用率を変えたときにrear-nearとrear-deepの構成比が大きく変化する問題を抑えています。

subsetの生成source構成は次のとおりです。

| subset | synthetic_001 | synthetic_004 | synthetic_005 |
|---|---:|---:|---:|
| top20 | 845 | 1,232 | 735 |
| top40 | 1,482 | 2,592 | 1,549 |
| top60 | 1,699 | 4,286 | 2,448 |
| top80 | 1,846 | 6,110 | 3,288 |
| top100 | 1,909 | 7,934 | 4,206 |

yaw帯ごとの採用率は固定されていますが、source構成は採用率によって変化しています。

## YawPose sampling

選択されたYawPose sampleは、各epochでそれぞれ1回だけdrawします。各epochの開始時にsubset全体をseed付きでshuffleし、同一epoch内で同じsampleを反復しません。

YawPose batchはexisting-HPE epoch全体へ決定論的に分散して配置します。最後のYawPose batchが64件未満の場合は、実サンプル数に比例するようyaw lossの寄与を補正します。

保存されたsampling結果では、YawPose使用条件の各epochで次の関係が成立しています。

`selected_unique_samples = unique_samples_drawn = total_draws`

`repeated_draws = 0`

`repeat_ratio = 0`

1 epochあたりのYawPose draw数はtop20で2,812件、top40で5,623件、top60で8,433件、top80で11,244件、top100で14,049件です。採用率が増えると1 epochあたりのYawPose教師信号総量も増加します。

## 条件行列

YawPose採用率6条件とexisting-HPE data 2系列を組み合わせ、合計12条件を比較しました。

| 系列 | 条件 |
|---|---|
| VGGHeads-only | `Y_base_vgg`、`Y_top20_vgg`、`Y_top40_vgg`、`Y_top60_vgg`、`Y_top80_vgg`、`Y_top100_vgg` |
| VGGHeads+DAD | `Y_base_vgg_dad`、`Y_top20_vgg_dad`、`Y_top40_vgg_dad`、`Y_top60_vgg_dad`、`Y_top80_vgg_dad`、`Y_top100_vgg_dad` |

各系列の`Y_base`はYawPoseを使用しないcontrol条件です。

## 評価条件

外部評価にはAGORA-HPE、AFLW2000、300W-LP、DAD-3DHeads official validationを使用します。外部benchmarkは学習やcheckpoint selectionには使用しません。

主要なinference設定は次のとおりです。

| 項目 | 設定 |
|---|---|
| inference precision | FP32 |
| AMP | 無効 |
| batch size | 256 |
| workers | 8 |
| metric dtype | FP64 |
| SO(3) geodesic formula | stable `atan2` |
| deterministic algorithms | 有効 |
| CUDA matmul / cuDNN conv FP32 precision | IEEE |
| crop source | evaluation manifestのGT crop |
| yaw group source | evaluation manifestのsource yaw |

candidateと保存済みbaselineは同一manifestとimage lockを使用して比較します。AGORA-HPE、AFLW2000、300W-LPでは`baseline_fp32`、DAD-3DHeads validationでは`baseline_dad_fp32`を基準にしています。

head-forward yawのbaseline-candidate比較では、同一sampleの誤差差に対して1,000回のpaired bootstrapを行い、乱数seedは0です。sample数が10,000件を超える場合は10,000件へ固定seedでsubsampleします。

保存済みbootstrap CIは各candidateと初期checkpointの差に対する区間です。同系列の`Y_base`と`Y_top20`〜`Y_top100`の差へ直接CIを付けたものではありません。

AGORA-HPEでは通常の30度yaw集計に加え、source yaw全周を15度単位の24区分に分けた集計も保存します。15度集計はAGORA-HPEだけに追加し、AFLW2000、300W-LP、DAD-3DHeads validationには追加しません。

## 結果

### 内部development

checkpoint selectionに使用したVGGHeads devの固定epoch 10結果は次のとおりです。

#### VGGHeads-only

| 条件 | rear SO(3) mean | rear P90 | front mean | side mean | rear yaw mean | rear yaw P90 | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Y_base | 25.712° | 51.564° | 19.695° | 20.935° | 22.277° | 49.312° | 8 |
| Y_top20 | 25.711° | 51.654° | 19.686° | 20.934° | 22.276° | 49.309° | 8 |
| Y_top40 | 25.696° | 51.545° | 19.689° | 20.924° | 22.258° | 49.288° | 8 |
| Y_top60 | 25.693° | 51.482° | 19.701° | 20.915° | 22.248° | 49.271° | 8 |
| Y_top80 | 25.688° | 51.520° | 19.694° | 20.911° | 22.241° | 49.256° | 8 |
| Y_top100 | 25.673° | 51.482° | 19.700° | 20.918° | 22.234° | 49.274° | 8 |

VGGHeads-onlyでは`Y_base → Y_top100`でrear SO(3) meanが約-0.039°、rear yaw meanが約-0.043°です。

#### VGGHeads+DAD-3DHeads

| 条件 | rear SO(3) mean | rear P90 | front mean | side mean | rear yaw mean | rear yaw P90 | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Y_base | 26.483° | 53.624° | 19.886° | 21.016° | 22.914° | 50.700° | 10 |
| Y_top20 | 26.477° | 53.613° | 19.883° | 21.016° | 22.906° | 50.342° | 10 |
| Y_top40 | 26.474° | 53.419° | 19.881° | 21.009° | 22.901° | 49.799° | 10 |
| Y_top60 | 26.482° | 53.541° | 19.888° | 21.001° | 22.906° | 50.364° | 10 |
| Y_top80 | 26.487° | 53.596° | 19.884° | 20.995° | 22.912° | 50.410° | 10 |
| Y_top100 | 26.461° | 53.357° | 19.881° | 21.003° | 22.884° | 49.810° | 10 |

VGGHeads+DAD系列では`Y_base → Y_top100`でrear SO(3) meanが約-0.022°、rear yaw meanが約-0.029°です。

内部best epochはVGGHeads-only系列の全条件が8、VGGHeads+DAD系列の全条件が10でした。主比較には全条件のepoch 10を使用しています。

### 外部benchmark全体

固定epoch 10のSO(3) geodesic meanは次のとおりです。

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

同系列の`Y_base`との差は、AGORA-HPEで最大約-0.050°、AFLW2000で約+0.001〜+0.008°、300W-LPで約-0.002〜-0.009°です。

### AGORA-HPE後方yaw

AGORA-HPEのrear `|yaw| >= 120°`は2,644件です。head-forward yaw meanは次のとおりです。

| 条件 | VGGHeads-only | VGGHeads+DAD |
|---|---:|---:|
| Y_base | 27.333° | 27.775° |
| Y_top20 | 27.337° | 27.771° |
| Y_top40 | 27.325° | 27.772° |
| Y_top60 | 27.307° | 27.758° |
| Y_top80 | 27.303° | 27.758° |
| Y_top100 | 27.301° | 27.765° |

VGGHeads-onlyの`Y_base → Y_top80`は約-0.030°、VGGHeads+DADの`Y_base → Y_top80`は約-0.018°です。初期checkpointのAGORA-HPE rear yaw meanは30.800°です。

### AGORA-HPE 15度yaw帯

`Y_top80 - Y_base`のSO(3) geodesic mean差をrear 8区分で比較した結果は次のとおりです。負値は同系列の`Y_base`より誤差が低いことを表します。

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

head-forward yaw errorでも同じ方向の変化が観測されます。VGGHeads-onlyの`Y_top80`では、学習対象外の-120〜-105°で約-0.239°、+105〜+120°で約-0.193°のhead-forward yaw mean低下も観測されています。

### DAD-3DHeads利用有無

YawPoseを使用しない`Y_base`同士の結果は次のとおりです。

| 評価 | VGGHeads-only | VGGHeads+DAD | DADあり − VGG-only |
|---|---:|---:|---:|
| AGORA-HPE overall | 44.498° | 44.415° | -0.083° |
| AFLW2000 overall | 6.541° | 6.535° | -0.006° |
| 300W-LP overall | 5.809° | 5.803° | -0.007° |
| DAD validation overall | 29.497° | 29.460° | -0.037° |
| AGORA rear yaw mean | 27.333° | 27.775° | +0.442° |
| VGGHeads dev rear SO(3) | 25.712° | 26.483° | +0.771° |

AGORA-HPEの15度SO(3)では、`Y_base_vgg_dad - Y_base_vgg`が-105〜-90°で約-1.078°、-90〜-75°で約-0.993°、+135〜+150°で約+0.609°、+150〜+165°で約+0.895°、+165〜+180°で約+0.960°です。

### DAD-3DHeads validation

DAD-3DHeads official validation全体では、初期checkpointのSO(3) mean 31.482°に対して、今回の全条件が約29.45〜29.51°です。

rear `|yaw| >= 120°`は47件です。rear yaw meanはVGGHeads-only系列で約50.20〜50.41°、VGGHeads+DAD系列で約51.25〜51.40°です。

## 考察

### 先行YawPose実験との比較

先行実験ではYawPose総draw数を約451,000件/epochへ固定していたため、top20では約2,810件のsubsetを平均約160.5回/画像/epoch、top100でも約32.1回/画像/epoch提示していました。また、rear全体でrankingしていたため、top20の97.6%が120–150°へ集中していました。

今回の実験では、各15度yaw帯から同じ採用率を取り、各sampleを1 epochにつき1回だけ提示しています。

AGORA-HPE rear yaw meanの`Y_base → Y_top80`変化は、先行実験の約-0.334°に対して、今回のVGGHeads-onlyが約-0.030°、VGGHeads+DADが約-0.018°です。また、先行実験で観測されたAFLW2000の0.3〜0.5°規模の退行は、今回のsingle-draw条件では再現していません。

今回の再設計では反復回数とyaw帯ranking方式を同時に変更しているため、effect sizeの縮小を反復回数だけに帰属することはできません。確認できるのは、先行実験で観測された大きな改善と大きな外部domain退行の双方が、各sampleを1 epochにつき1回だけ提示する今回の条件では再現しなかったことです。

### Reliability採用率

15度帯内でreliability rankingを行った今回の結果では、top20が他の採用率より一貫して低い誤差を示していません。AGORA-HPE overallでは両系列ともtop60〜80付近まで誤差がわずかに低下し、top100で一部戻る形があります。

ただし、top20からtop100へ採用率を広げると、1 epochあたりのYawPose教師sample数も2,812件から14,049件へ増加します。そのため、採用率間の差にはreliability thresholdとYawPose教師信号総量の両方が含まれます。

また、15度yaw帯の構成は揃えていますが、生成source構成は採用率によって変化します。したがって、今回の採用率比較もreliability scoreだけの効果を完全に分離したものではありません。

### 角度依存

AGORA-HPEの15度集計では、VGGHeads-onlyとVGGHeads+DADの両系列で、120–150°付近の誤差低下と165–180°付近の誤差増加が共通して観測されました。

この方向はhead-forward yaw errorだけでなくSO(3) geodesic errorでも確認されています。したがって、通常evaluationのEuler yaw表現やhead-forward yaw計算だけに起因する見かけの差ではありません。

全rear平均では各15度帯の正負の変化が相殺されるため、YawPose追加によるmeanの変化は数百分の一度まで小さくなっています。

### DAD-3DHeads

DAD-3DHeadsを追加した系列では、AGORA-HPE、AFLW2000、300W-LP、DAD validationのoverall SO(3) meanがVGGHeads-onlyよりわずかに低い値です。一方、AGORA-HPE rear yawとVGGHeads dev rear SO(3)ではVGGHeads-onlyの方が低い値です。

AGORA-HPEの15度集計でもDAD追加の影響は一様ではなく、negative側の一部side帯では誤差が低下し、positive rearでは誤差が増加しています。そのため、AGORA-HPE overallの約-0.083°という差は、全yaw帯が一様に改善した結果ではありません。

今回の比較ではexisting-HPE draw数を417,024件/epochへ固定しているため、DADあり系列ではVGGHeads sampleの一部がDAD sampleに置き換わります。したがって、DAD有無の差はDADを追加して総学習量を増やした効果ではありません。

### Teacher ensemble

reliability scoreにはSixDRepNet360 base、SemiUHPE、WHENetの3 teacherを等しい指標構成で使用しています。

yaw符号較正後の方向検証集合に対する平均誤差は、WHENetが約29.677°、SemiUHPEが約122.134°です。SemiUHPEは反対符号より選択符号の方が低いため符号較正自体は成立していますが、YawPose後方domainに対する絶対誤差は大きい状態です。

reliability scoreには`gt_max_error_deg`も含むため、domain mismatchの大きいteacherもrankingへ影響します。本実験ではteacher構成を変更していないため、個々のteacherがrankingへ与えた寄与は分離していません。

## 制約

本実験の解釈に関係する制約は次のとおりです。

- 学習は単一seed 42で実行しており、seed間分散は測定していません。
- 同系列の`Y_base`とYawPose条件の差に直接paired bootstrap CIを計算していないため、0.01〜0.05°規模の差について統計的な方向は確定していません。
- YawPose adoption ratioと1 epochあたりのYawPose教師sample総数が連動しています。
- 15度yaw帯内の採用率は固定していますが、生成source構成はadoption ratioによって変化します。
- reliability scoreはラベル正解確率ではなく、同一15度yaw帯内の相対順位です。
- teacher ensembleにはYawPose後方domainとの整合度が低いSemiUHPEが含まれます。
- YawPose rear candidateの判定はcanonical yawに依存するため、canonical yawが大きく誤っているsampleはcandidate集合の構成にも影響します。
- DAD有無の比較ではexisting-HPE training budgetを固定しているため、DADあり系列ではVGGHeads sampleの一部がDAD sampleに置き換わります。
- AGORA-HPEではEuler yaw errorとhead-forward yaw errorの両方を保存しており、両指標は同一ではありません。
- DAD-3DHeads official validationのrearは47件であり、細かなrear角度帯を安定して評価できる件数ではありません。

## 結論

15度yaw帯ごとにreliability rankingを行い、選択したYawPose sampleを各epochで1回だけ使用した今回の実験では、YawPose追加による全体平均とrear平均の変化は先行実験より小さくなりました。AGORA-HPE rear yaw meanの`Y_base → Y_top80`は、VGGHeads-onlyで約-0.030°、VGGHeads+DADで約-0.018°でした。

一方、15度単位では120–150°付近の誤差低下と165–180°付近の誤差増加が両系列で共通して観測されました。YawPoseの影響は全rearを一様に改善する形ではなく、yaw帯によって方向が異なっています。

reliability上位20%だけが明確に低い誤差を示す関係は確認されませんでした。ただし、採用率とYawPose教師sample総数が連動しているため、採用率間の差をreliability scoreだけへ帰属することはできません。

DAD-3DHeadsを使用しないVGGHeads-only系列でも、YawPoseによる角度依存の変化は再現しています。また、AGORA-HPE rear yawとVGGHeads dev rear SO(3)ではVGGHeads-only系列の方が低い値でした。本実験の範囲では、DAD-3DHeadsを学習データから外すことで後方yaw性能が成立しなくなる結果は観測されていません。

## 保存成果物

本レポートの根拠となる主要成果物は次のとおりです。

- 実験計画は`docs/experiments/yawpose_rear_yaw_stratified_single_draw_plan.md`です。
- 先行YawPose実験結果は`docs/experiments/yawpose_rear_yaw_reliability_results.md`です。
- reliability設定は`experiments/runs/yawpose_reliability_stratified15/config.json`です。
- reliability provenanceは`experiments/runs/yawpose_reliability_stratified15/provenance.json`です。
- reliability集計は`experiments/runs/yawpose_reliability_stratified15/metrics/summary.json`です。
- reliability sample-level結果は`experiments/runs/yawpose_reliability_stratified15/predictions/reliability.jsonl`です。
- reliability subsetは`experiments/runs/yawpose_reliability_stratified15/predictions/subsets/`です。
- 12条件の設定は`experiments/runs/yawpose_rear_stratified_search/config.json`です。
- 条件別内部評価は`experiments/runs/yawpose_rear_stratified_search/conditions/<condition>/metrics/`です。
- 内部固定epoch比較は`experiments/runs/yawpose_rear_stratified_search/metrics/comparison.json`です。
- 条件別外部評価は`eval/yawpose_rear_stratified_search/conditions/`です。
- baselineとのSO(3)比較は`eval/yawpose_rear_stratified_search/comparisons/`です。
- head-forward yaw比較は`eval/yawpose_rear_stratified_search/yaw_comparisons/`です。
- AGORA-HPE、AFLW2000、300W-LPのyaw集計は`eval/yawpose_rear_stratified_search/yaw_summary_baseline_fp32_final.json`です。
- DAD-3DHeads validationのyaw集計は`eval/yawpose_rear_stratified_search/yaw_summary_baseline_dad_fp32_final__dad.json`です。
