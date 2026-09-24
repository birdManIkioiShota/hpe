# YawPose後方yaw信頼度フィルタリング実験結果

## 概要

本実験は、SixDRepNet360-ResNet50の全周頭部姿勢推定に対して、YawPoseの後方合成画像をyaw-only教師信号として追加し、複数teacherとの一致度から作成したreliability rankingの採用範囲が学習結果へ与える影響を評価したものです。

比較ではYawPoseを使用しない`Y_base`をcontrol条件とし、reliability ranking上位20%、40%、60%、80%、100%を使用する5条件を同一学習設定で実行しました。主要比較には各条件の固定epoch 10 checkpointを使用し、内部development split、AGORA-HPE、AFLW2000、300W-LP、DAD-3DHeads validationで評価しました。

結果の解釈では、reliability scoreそのものだけでなく、採用subsetのyaw角度分布、生成source分布、固定YawPose draw数によって生じる反復回数の差を分離して扱います。

## 目的

YawPoseにはyawラベルの揺らぎと画像品質のばらつきがあるため、YawPoseの全後方候補を一律に使用する代わりに、固定teacher ensembleとcanonical yawの一致度を用いて相対的な信頼度順位を作成しました。

中心的な検証対象は、reliability rankingの上位範囲を段階的に広げたときに、後方yaw誤差、SO(3)回転誤差、front / side保持性能、外部データセットへのgeneralizationがどのように変化するかです。

teacher予測はYawPoseラベルの置換には使用しません。teacher ensembleはcanonical yawの相対的な信頼度を事前計算するためだけに使用し、学習時のYawPose教師信号には固定したcanonical yawを使用します。

## 対象モデル

対象モデルはSixDRepNet360-ResNet50です。モデル構造は変更せず、ResNet50の`layer4`とrotation regression headだけを更新します。

初期checkpointは次の固定checkpointです。

| 項目 | 値 |
|---|---|
| checkpoint | `checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` |
| SHA-256 | `3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6` |
| model | SixDRepNet360-ResNet50 |
| update scope | ResNet50 `layer4` + regression head |

本レポートでは、この学習開始時checkpointを「初期checkpoint」と表記します。`Y_base`は初期checkpointそのものではなく、YawPose streamを無効化した状態で同じfine-tuning手順を10 epoch実行したcontrol条件です。

## データ

### 既存HPEデータ

既存HPE学習データにはVGGHeadsとDAD-3DHeads trainを使用します。これらのサンプルには完全な回転行列教師があるため、SO(3) geodesic lossで学習します。

VGGHeadsとDAD-3DHeads trainを合わせた既存HPEデータの自然な後方比率は1.743%です。本実験ではYawPose自体が追加の後方教師信号になるため、既存HPE側のrear oversamplingは行わず、この自然比率を維持します。

既存HPEデータでは、非後方サンプルに初期checkpointへのretention distillationを適用し、後方サンプルにはhorizontal flip前後の予測整合性を要求するSO(3) flip-consistency lossを適用します。

### YawPose

YawPoseは`PINTO0309/YawNet`の`resources` releaseで公開されている合成頭部yawデータセットです。本実験で固定した公開データのprovenanceは次のとおりです。

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

YawPoseの公開yawは`yaw_deg ∈ [0, 360)`で、0°が正面、+90°がviewer-leftです。公開データには`synthetic_001`から`synthetic_006`までの生成sourceがあります。

公開`labels_fixed.jsonl`にはYawNet側で行われたラベルcleaningが既に反映されています。`synthetic_001`の`intent_rear`の一部には上流データ作成時のDINOv3 teacherによる再ラベルが含まれますが、この処理は公開データセットのprovenanceとして扱い、本実験のteacher ensembleによる処理とは区別します。

YawPoseにはpitch情報を持つサンプルがありますが、本実験ではYawPose由来のpitchとrollを教師信号として使用しません。YawPose streamから使用する教師信号はcanonical yawだけです。

### canonical yaw

canonical yawはYawPoseサンプルの学習教師として固定するyawです。未修正サンプルでは公開`labels_fixed.jsonl`の`yaw_deg`を共通yaw規約へ変換した値を使用し、固定した`manual_corrections.jsonl`に同一サンプルの修正がある場合だけ修正値で上書きします。

teacher ensembleの平均yaw、median yaw、その他の集約値をcanonical yawへ置き換える処理は行いません。したがって、reliability scoreはpseudo-GTを生成する仕組みではなく、固定されたcanonical yawとteacher群の整合度を順位化する仕組みです。

保存されたsubset summaryでは、`top020`から`top100`までの`human_corrected`件数はいずれも0件です。

## yaw規約と姿勢区分

### 共通yaw規約

YawPoseの公開`yaw_deg`は次式でsigned yawへ変換します。

`yaw_signed = ((yaw_deg + 180) mod 360) - 180`

値域は`[-180°, 180°)`です。0°は正面、+90°はviewer-left、-90°はviewer-rightです。

回転行列を出力するモデルでは、head-localの`+Z`軸をhead-forward vectorとして扱います。

`v = R * [0, 0, 1]^T`

`v = [v_x, v_y, v_z]^T`としたとき、head-forward yawは`atan2(v_x, v_z)`で求めます。

内部development評価では、このrotation matrix由来のhead-forward yawを使用します。外部評価では各evaluation manifestに保存されたsource yawをGT yawと姿勢区分に使用するため、内部developmentのyaw値と外部データセットのsource yaw値をサンプル横断で直接比較しません。

### 姿勢区分

YawPoseの学習対象はcanonical yawが`|yaw| >= 120°`となる後方候補だけです。前方と側方のYawPoseサンプルは学習へ使用しません。

後方内部は次の4区分で診断します。

| 区分 | yaw範囲 |
|---|---|
| negative rear-near | `-150° < yaw <= -120°` |
| negative rear-deep | `-180° <= yaw <= -150°` |
| positive rear-near | `120° <= yaw < 150°` |
| positive rear-deep | `150° <= yaw <= 180°` |

本レポートでは120–150°をrear-near、150–180°をrear-deepとして扱います。

### 評価指標

回転全体の精度にはSO(3) geodesic errorを使用します。YawPoseはyaw-only教師信号であるため、SO(3)とは別に周期境界を考慮したyaw absolute angular errorも記録します。

rear yawではmean、median、P90、`>30°`率、`>60°`率、`>90°`率を確認します。front / sideについてもmeanとP90を確認し、後方改善に伴う保持性能の変化を診断します。

## reliability score

### teacher ensemble

reliability事前計算には次の3モデルを固定teacherとして使用します。

| teacher ID | モデル | 固定実装またはcheckpoint |
|---|---|---|
| `sixdrepnet360_base` | SixDRepNet360-ResNet50 | 本リポジトリの初期checkpoint |
| `semiuhpe_effnetv2s` | SemiUHPE EfficientNetV2-S full-range | `hnuzhy/SemiUHPE@c8f67102bf5aba8869b3f23453ac67599f21aa1f` |
| `whenet` | WHENet wide-range yaw | `Ascend-Research/HeadPoseEstimation-WHENet@a0d7bdfb5e2ac97ae6b0ae3eef79fdcf4075ab82` |

teacherごとにyaw定義が異なるため、各出力は共通yaw規約へ変換してから比較します。SemiUHPEとWHENetのyaw符号は、生成時の方向QAを通過した`intent_s004`と`intent_s005`の後方サンプル9,138件を用いて実データ上で較正しています。

符号較正で選択された符号における平均yaw誤差はWHENetが29.677°、SemiUHPEが122.134°です。SemiUHPEでは反対符号の平均誤差152.551°より選択符号の方が小さいため符号自体は区別されていますが、YawPose後方集合に対する絶対誤差は大きい状態です。

### 事前計算指標

YawPose rear candidateを14,049件抽出し、各サンプルについてteacher群とcanonical yawの関係から次の指標を事前計算します。

| 指標 | 定義 |
|---|---|
| `gt_median_error` | 各teacherとcanonical yawの円周角度差のmedianです。 |
| `gt_max_error` | 各teacherとcanonical yawの円周角度差の最大値です。 |
| `teacher_dispersion` | teacher同士のpairwise yaw差のmedianです。 |
| `agree_10` | canonical yawから10°以内に入るteacherの割合です。 |
| `agree_20` | canonical yawから20°以内に入るteacherの割合です。 |
| `agree_30` | canonical yawから30°以内に入るteacherの割合です。 |

角度差には円周距離を使用します。

`d_circ(a, b) = abs(atan2(sin(a - b), cos(a - b)))`

`agree_10`、`agree_20`、`agree_30`は診断値として保存しますが、reliability scoreには含めません。

### scoreとranking

`gt_median_error`、`teacher_dispersion`、`gt_max_error`をそれぞれrear candidate集合内のpercentile rankへ変換します。小さい誤差ほど小さいrank値となるようにし、それぞれを`r_median`、`r_dispersion`、`r_max`とします。

reliability scoreは次式です。

`reliability_score = (r_median + r_dispersion + r_max) / 3`

値域は0から1で、値が小さいほどreliability ranking上位です。このscoreはラベルが正しい確率ではなく、14,049件の候補集合内における相対順位指標です。

同一scoreの場合は`gt_median_error`、`teacher_dispersion`、`gt_max_error`、sample IDの順でtie-breakします。reliabilityは学習前に一度だけ計算し、学習中のモデル予測やepochによって更新しません。

## 実験設計

### 比較条件

比較条件は次の6条件です。

| 条件 | YawPose採用範囲 |
|---|---|
| `Y_base` | YawPoseを使用しません。 |
| `Y_top20` | reliability ranking上位20%を使用します。 |
| `Y_top40` | reliability ranking上位40%を使用します。 |
| `Y_top60` | reliability ranking上位60%を使用します。 |
| `Y_top80` | reliability ranking上位80%を使用します。 |
| `Y_top100` | rear candidate 14,049件をすべて使用します。 |

subsetはnested subsetであり、`Y_top20 ⊂ Y_top40 ⊂ Y_top60 ⊂ Y_top80 ⊂ Y_top100`の関係です。すべての条件は同一の初期checkpointから独立に開始します。

### 学習損失

総損失は概念的に次式です。

`L_total = L_original_supervised + 1.0 * L_distillation + 0.2 * L_original_flip + 0.2 * L_yawpose`

各損失項の適用対象は次のとおりです。

| 損失 | 適用対象 | 内容 |
|---|---|---|
| `L_original_supervised` | existing HPE data全体 | GT rotation matrixに対するSO(3) geodesic lossです。 |
| `L_distillation` | existing HPE dataの非後方 | 初期checkpointの予測に対するretention distillationです。 |
| `L_original_flip` | existing HPE dataの後方 | horizontal flip前後の予測に対するSO(3) flip-consistency lossです。 |
| `L_yawpose` | 選択されたYawPose rear candidate | canonical yawだけを教師とするyaw-only lossです。 |

YawPoseの予測yawを`y_pred`、canonical yawを`y_gt`としたとき、yaw-only lossには周期境界を考慮した角度差を使用します。

`delta = atan2(sin(y_pred - y_gt), cos(y_pred - y_gt))`

`L_yawpose = mean(abs(delta))`

loss計算はradianで行います。

### 共通学習条件

YawPose採用範囲以外の主要条件は固定しています。

| 項目 | 設定 |
|---|---|
| existing train data | VGGHeads + DAD-3DHeads train |
| existing rear sampling | 自然比率1.743% |
| epochs | 10 |
| samples / epoch | 451,072 |
| existing HPE micro-batch | 64 |
| YawPose micro-batch | 64 |
| gradient accumulation | 2 |
| existing HPE effective batch | 128 |
| YawPose effective batch | 128 |
| precision | BF16 |
| optimizer | AdamW |
| backbone LR | `1e-6` |
| regression head LR | `3e-5` |
| weight decay | `1e-4` |
| warmup updates | 100 |
| gradient clip norm | 1.0 |
| retention distillation weight | 1.0 |
| existing rear flip-consistency weight | 0.2 |
| YawPose yaw loss weight | 0.2 |
| seed | 42 |

`Y_base`ではYawPose weightを0にし、それ以外の学習条件を揃えます。`Y_top20`から`Y_top100`ではYawPose batch sizeとYawPose loss weightを固定するため、採用subsetが小さくても1 updateあたりのYawPose streamの存在量は減りません。

### checkpoint比較

主比較には各条件の固定epoch 10 checkpointを使用します。条件ごとに異なるbest epochを主比較へ使用すると、YawPose採用範囲の効果とcheckpoint選択時期の効果が混在するためです。

内部best checkpointは別途保存されますが、本レポートのadoption ratio比較には使用しません。外部benchmarkもcheckpoint selectionには使用していません。

## subset診断

### 角度分布

reliability ranking上位集合のrear-near / rear-deep構成は次のとおりです。

| subset | 件数 | 120–150° | 150–180° | 150–180°比率 |
|---|---:|---:|---:|---:|
| top20 | 2,810 | 2,742 | 68 | 2.4% |
| top40 | 5,620 | 4,776 | 844 | 15.0% |
| top60 | 8,430 | 5,826 | 2,604 | 30.9% |
| top80 | 11,240 | 6,768 | 4,472 | 39.8% |
| top100 | 14,049 | 7,576 | 6,473 | 46.1% |

top20では97.6%が120–150°であり、150–180°は2.4%です。採用範囲を広げるにつれてrear-deepの構成比が増加し、top100では46.1%になります。

この分布は、reliability scoreとyaw角度帯が独立していないことを示します。reliability閾値を変更する操作は、同時に学習subsetの角度分布も変更しています。

### 生成source分布

rear candidateのsubset summaryでは、選択サンプルは`synthetic_001`、`synthetic_004`、`synthetic_005`の3 sourceから構成されています。

| subset | synthetic_001 | synthetic_004 | synthetic_005 |
|---|---:|---:|---:|
| top20 | 865 | 869 | 1,076 |
| top40 | 1,547 | 2,149 | 1,924 |
| top60 | 1,785 | 4,044 | 2,601 |
| top80 | 1,843 | 6,019 | 3,378 |
| top100 | 1,909 | 7,934 | 4,206 |

採用範囲を広げると角度分布だけでなく生成source構成も変化します。したがって、adoption ratio間の差にはreliability score、yaw帯、生成sourceの3要素が同時に含まれます。

### 反復量

YawPoseを使用する全条件では、subset sizeにかかわらず1 epochあたり451,072件をYawPose streamからdrawします。そのため、小さいsubsetほど同一画像の反復回数が増えます。

| 条件 | selected unique | total draws / epoch | 平均draw / image / epoch | repeat ratio |
|---|---:|---:|---:|---:|
| `Y_top20` | 2,810 | 451,072 | 160.5 | 99.38% |
| `Y_top40` | 5,620 | 451,072 | 80.3 | 98.75% |
| `Y_top60` | 8,430 | 451,072 | 53.5 | 98.13% |
| `Y_top80` | 11,240 | 451,072 | 40.1 | 97.51% |
| `Y_top100` | 14,049 | 451,072 | 32.1 | 96.89% |

`Y_top20`では1画像が平均160.5回/epoch提示され、10 epochでは単純平均で約1,605回です。`Y_top100`では平均32.1回/epochであるため、1画像あたりの平均提示回数には約5倍の差があります。

全条件でYawPose loss weightは0.2ですが、1サンプル当たりの累積的な提示回数は同一ではありません。この反復量の差はreliability閾値比較に含まれる交絡要因です。

## 内部development評価

内部development splitはVGGHeadsとDAD-3DHeadsのdevelopmentデータ55,719件です。YawPoseサンプルは内部developmentへ含めません。

### 全体と後方

初期checkpointと固定epoch 10の主要結果は次のとおりです。

| 条件 | SO(3) overall mean | SO(3) rear mean | SO(3) rear P90 | yaw overall mean | yaw rear mean | yaw rear P90 |
|---|---:|---:|---:|---:|---:|---:|
| 初期checkpoint | 22.640° | 35.462° | 71.487° | 19.776° | 28.819° | 67.205° |
| `Y_base` | 21.163° | 26.593° | 53.863° | 18.501° | 22.811° | 50.342° |
| `Y_top20` | 21.006° | 26.900° | 53.179° | 18.373° | 23.212° | 50.236° |
| `Y_top40` | 20.981° | 26.746° | 53.138° | 18.346° | 23.014° | 50.124° |
| `Y_top60` | 20.983° | 26.575° | 52.614° | 18.343° | 22.795° | 49.875° |
| `Y_top80` | 20.993° | 26.294° | 52.456° | 18.350° | 22.485° | 49.848° |

初期checkpointから`Y_base`への変化だけでrear SO(3) meanは8.869°、rear yaw meanは6.008°低下しています。YawPoseを追加した条件の効果は、この`Y_base`からの差として分離して解釈する必要があります。

`Y_base`比では`Y_top20`のrear yaw meanが+0.401°、`Y_top40`が+0.203°、`Y_top60`が-0.017°、`Y_top80`が-0.326°です。

`Y_top100`の`experiments/runs/yawpose_rear_search/conditions/Y_top100/metrics/epoch_010.json`はリポジトリ上で0 byteのため、固定epoch 10の内部development値を保存成果物から検証できません。一方、外部評価成果物には`Y_top100`のepoch 10 checkpointを用いた評価結果とcheckpoint SHA-256が保存されています。

### front / side保持

内部SO(3)のfront / side meanとP90は次のとおりです。

| 条件 | front mean | front P90 | side mean | side P90 |
|---|---:|---:|---:|---:|
| 初期checkpoint | 22.354° | 57.575° | 23.201° | 75.533° |
| `Y_base` | 21.023° | 48.615° | 21.755° | 67.625° |
| `Y_top20` | 20.852° | 48.115° | 21.681° | 67.203° |
| `Y_top40` | 20.832° | 47.933° | 21.606° | 67.217° |
| `Y_top60` | 20.840° | 47.880° | 21.578° | 67.085° |
| `Y_top80` | 20.854° | 47.920° | 21.597° | 67.313° |

内部developmentでは、`Y_top20`から`Y_top80`までのfront / side SO(3) meanとP90は`Y_base`と近い範囲にあり、YawPose条件で大きな内部退行は観測されていません。

### 後方4区分

内部developmentのyaw meanを後方4区分へ分解すると次の結果です。

| 条件 | negative 120–150° | negative 150–180° | positive 120–150° | positive 150–180° |
|---|---:|---:|---:|---:|
| 初期checkpoint | 21.742° | 23.591° | 35.779° | 33.372° |
| `Y_base` | 21.070° | 18.556° | 27.381° | 24.836° |
| `Y_top20` | 19.436° | 19.987° | 26.971° | 25.972° |
| `Y_top40` | 19.176° | 19.842° | 26.606° | 25.856° |
| `Y_top60` | 19.156° | 19.503° | 26.531° | 25.545° |
| `Y_top80` | 19.173° | 19.032° | 26.327° | 25.126° |

`Y_top20`を`Y_base`と比較すると、negative 120–150°は-1.633°、positive 120–150°は-0.410°です。一方、negative 150–180°は+1.431°、positive 150–180°は+1.137°です。

採用範囲を広げると150–180°の悪化量は縮小し、`Y_top80`では`Y_base`との差がnegative側+0.476°、positive側+0.290°まで小さくなっています。

### rear yaw tail error

内部developmentのrear yaw分布は次のとおりです。

| 条件 | mean | median | P90 | >30° | >60° | >90° |
|---|---:|---:|---:|---:|---:|---:|
| 初期checkpoint | 28.819° | 17.259° | 67.205° | 31.625% | 11.992% | 6.383% |
| `Y_base` | 22.811° | 13.051° | 50.342° | 22.534% | 7.930% | 4.836% |
| `Y_top20` | 23.212° | 13.463° | 50.236° | 22.147% | 7.834% | 5.029% |
| `Y_top40` | 23.014° | 13.051° | 50.124° | 21.954% | 7.737% | 5.029% |
| `Y_top60` | 22.795° | 12.744° | 49.875° | 21.954% | 7.737% | 4.932% |
| `Y_top80` | 22.485° | 12.748° | 49.848° | 21.857% | 7.640% | 4.932% |

YawPose条件間の差はrear meanより小さいtail metricも多く、`Y_top20`から`Y_top80`までrear P90は50.236°から49.848°の範囲にあります。

## 外部評価

### 評価条件

外部評価では各条件の固定epoch 10 checkpointをFP32で評価します。AMPは無効、batch sizeは256、metric dtypeはFP64で、deterministic algorithmsを有効にしています。

AGORA-HPEは全周yawを含むためrear yawの主要な外部評価に使用します。AFLW2000と300W-LPは主としてfront / side保持の確認に使用し、DAD-3DHeads validationは独立validationとして使用します。

外部yaw比較では、candidateと初期checkpointを同一サンプルで比較し、保存されたpaired bootstrap CIも確認します。

### データセット全体のSO(3)

外部データセット全体のSO(3) geodesic meanは次のとおりです。

| 条件 | AGORA-HPE | AFLW2000 | 300W-LP | DAD-3DHeads validation |
|---|---:|---:|---:|---:|
| 初期checkpoint | 45.475° | 6.562° | 5.845° | 31.482° |
| `Y_base` | 44.402° | 6.534° | 5.803° | 29.385° |
| `Y_top20` | 44.140° | 7.011° | 5.931° | 29.086° |
| `Y_top40` | 44.128° | 7.030° | 5.669° | 29.054° |
| `Y_top60` | 44.092° | 6.923° | 5.618° | 29.071° |
| `Y_top80` | 44.159° | 6.823° | 5.637° | 29.112° |
| `Y_top100` | 44.275° | 6.754° | 5.650° | 29.135° |

AGORA-HPE、300W-LP、DAD-3DHeads validationでは一部のYawPose条件が`Y_base`より低いSO(3) meanを示します。一方、AFLW2000ではYawPoseを使用する全条件が`Y_base`より高いSO(3) meanです。

### AFLW2000と300W-LPのyaw

AFLW2000のsource yawによるoverall、front、sideのyaw meanは次のとおりです。

| 条件 | overall | front | side |
|---|---:|---:|---:|
| `Y_base` | 4.879° | 4.529° | 6.829° |
| `Y_top20` | 5.414° | 4.656° | 9.650° |
| `Y_top40` | 5.499° | 4.857° | 9.083° |
| `Y_top60` | 5.374° | 4.840° | 8.354° |
| `Y_top80` | 5.238° | 4.748° | 7.971° |
| `Y_top100` | 5.150° | 4.694° | 7.697° |

AFLW2000ではYawPose追加後の悪化がsideで大きく、`Y_top20`では`Y_base`比で+2.820°です。採用範囲を広げるとsideの悪化幅は縮小しますが、`Y_top100`でも`Y_base`より+0.868°高い値です。

300W-LPのsource yawによるoverall、front、sideのyaw meanは次のとおりです。

| 条件 | overall | front | side |
|---|---:|---:|---:|
| `Y_base` | 4.374° | 4.354° | 4.408° |
| `Y_top20` | 4.497° | 4.304° | 4.819° |
| `Y_top40` | 4.188° | 4.152° | 4.250° |
| `Y_top60` | 4.140° | 4.144° | 4.135° |
| `Y_top80` | 4.167° | 4.204° | 4.106° |
| `Y_top100` | 4.181° | 4.235° | 4.092° |

300W-LPでは`Y_top20`のsideが`Y_base`より高い一方、`Y_top40`以降はoverallが`Y_base`より低くなっています。外部domainによってYawPose追加後の変化方向は一致していません。

### AGORA-HPE後方yaw

AGORA-HPEのrear `|yaw| >= 120°`は2,644件です。rear yaw分布は次のとおりです。

| 条件 | mean | median | P90 | >30° | >60° | >90° |
|---|---:|---:|---:|---:|---:|---:|
| 初期checkpoint | 30.800° | 22.480° | 65.862° | 36.876% | 12.216% | 5.182% |
| `Y_base` | 27.720° | 21.277° | 53.542° | 34.304% | 7.375% | 3.631% |
| `Y_top20` | 27.534° | 20.879° | 53.706° | 33.359% | 7.375% | 3.555% |
| `Y_top40` | 27.479° | 20.722° | 53.468° | 33.359% | 7.337% | 3.555% |
| `Y_top60` | 27.434° | 20.657° | 53.469° | 33.018% | 7.337% | 3.555% |
| `Y_top80` | 27.385° | 20.623° | 53.501° | 33.359% | 7.337% | 3.555% |
| `Y_top100` | 27.472° | 20.902° | 53.297° | 33.737% | 7.375% | 3.631% |

初期checkpointから`Y_base`までにrear yaw meanは3.080°低下しています。YawPose追加による`Y_base`からの追加変化は`Y_top20`で-0.186°、`Y_top40`で-0.241°、`Y_top60`で-0.286°、`Y_top80`で-0.334°、`Y_top100`で-0.248°です。

rear P90は`Y_base`の53.542°に対してYawPose条件が53.297–53.706°であり、meanほど明確な採用率依存の変化はありません。

### AGORA-HPE後方4区分

AGORA-HPEのrear yaw meanを符号と角度帯で分解すると次の結果です。

| 条件 | negative 120–150° | negative 150–180° | positive 120–150° | positive 150–180° |
|---|---:|---:|---:|---:|
| `Y_base` | 31.155° | 19.309° | 39.005° | 20.763° |
| `Y_top20` | 29.270° | 20.800° | 37.075° | 22.392° |
| `Y_top40` | 29.203° | 20.790° | 36.960° | 22.365° |
| `Y_top60` | 29.227° | 20.578° | 37.132° | 22.192° |
| `Y_top80` | 29.484° | 20.235° | 37.372° | 21.836° |
| `Y_top100` | 30.363° | 19.923° | 37.846° | 21.142° |

`Y_top20`を`Y_base`と比較すると、120–150°ではnegative側が-1.885°、positive側が-1.930°です。一方、150–180°ではnegative側が+1.491°、positive側が+1.629°です。

採用範囲を広げるにつれて150–180°の悪化量は縮小し、`Y_top100`では`Y_base`との差がnegative側+0.614°、positive側+0.380°です。

この方向性は内部developmentの後方4区分でも観測されています。`Y_top20`では内部120–150°が`Y_base`より低下し、内部150–180°が両符号で増加しています。

### DAD-3DHeads validation後方yaw

DAD-3DHeads validationのrear `|yaw| >= 120°`は47件です。rear yaw meanと初期checkpointとの差に対する95% bootstrap CIは次のとおりです。

| 条件 | rear yaw mean | 初期checkpointとの差の95% CI |
|---|---:|---:|
| `Y_base` | 51.564° | [-9.679°, +0.706°] |
| `Y_top20` | 49.438° | [-11.364°, -1.946°] |
| `Y_top40` | 49.623° | [-10.834°, -2.035°] |
| `Y_top60` | 49.652° | [-10.853°, -1.978°] |
| `Y_top80` | 49.793° | [-11.203°, -1.360°] |
| `Y_top100` | 49.519° | [-12.110°, -1.159°] |

DAD-3DHeads validationではrear件数が47件しかなく、4区分へ分けるとnegative 150–180°が1件、positive 150–180°が7件になります。そのため、DAD-3DHeadsのrear細分化は補助的な確認に限定します。

## 学習挙動の解釈

### reliabilityと角度分布

reliability上位20%の97.6%が120–150°であり、150–180°は2.4%しか含まれません。したがって、`Y_top20`から`Y_top100`までの比較は、ラベル信頼度の閾値だけを変えた比較ではありません。

観測事実として、reliability ranking上位ほど120–150°へ偏り、採用範囲を広げるほど150–180°が増えています。また、内部developmentとAGORA-HPEの双方で、`Y_top20`は`Y_base`に対して120–150°のyaw meanが低下し、150–180°のyaw meanが増加しています。

この対応関係は、reliability rankingがyaw帯と強く関連していることと整合します。一方、120–150°が本質的に容易な姿勢であることを独立に測定した実験ではないため、reliability scoreを姿勢難易度そのものの指標とは扱いません。

### 反復集中

epoch 10のYawPose training lossは`Y_top20`が0.1099 rad、`Y_top40`が0.1805 rad、`Y_top60`が0.2245 rad、`Y_top80`が0.2985 radです。

`Y_top20`は最小のtraining lossを示しますが、内部rear yaw meanは23.212°であり、YawPoseを使用しない`Y_base`の22.811°より0.401°高い値です。採用範囲を広げた`Y_top60`は22.795°、`Y_top80`は22.485°です。

`Y_top20`では2,810件のsubsetを1 epochあたり451,072回drawするため、1画像あたり平均160.5回提示されます。また、そのsubsetの97.6%が120–150°です。この条件では、少数の120–150°サンプルへYawPose教師信号が高頻度で集中します。

この学習条件と、120–150°の改善および150–180°の悪化が同時に観測されることは整合しています。ただし、adoption ratioを変えるとreliability score、yaw角度分布、生成source分布、1画像あたり反復回数が同時に変わるため、反復回数だけを原因として分離することはできません。

### 外部domain

AFLW2000ではYawPoseを使用する全条件のSO(3) meanが`Y_base`より高く、yaw side meanも全条件で`Y_base`より高い値です。300W-LPでは`Y_top20`のside yawが悪化する一方、`Y_top40`以降ではoverall yawが`Y_base`より低下しています。

AGORA-HPEではrear yaw meanがYawPose条件で`Y_base`より0.186–0.334°低下していますが、rear P90の変化は小さい範囲です。

したがって、YawPose yaw-only教師信号による変化はrear meanだけに限定されず、外部domainによってfront / sideを含む挙動が異なります。

### teacher ensemble

WHENetの符号較正後平均誤差は29.677°ですが、SemiUHPEは122.134°です。reliability scoreは3 teacherの関係を等重みのpercentile rankで統合するため、YawPose後方domainに対する整合度が低いteacherもrankingへ寄与しています。

この事実はreliability scoreの識別力を解釈する際の制約です。ただし、本実験ではSemiUHPEを除いたreliability rankingを別条件として作成していないため、SemiUHPEの寄与量を結果から分離することはできません。

## 制約

本実験の解釈に関係する制約を以下に示します。

- reliability scoreとyaw角度分布が交絡しているため、ラベル信頼度だけの効果を分離できません。
- adoption ratioと生成source分布が同時に変化するため、特定sourceの影響を分離できません。
- YawPose総draw数を固定したため、subset sizeと1画像あたりの反復回数が交絡しています。
- reliability score、yaw角度分布、生成source分布、反復回数が同時に変わるため、rear-near改善またはrear-deep悪化を単一要因へ帰属できません。
- SemiUHPEはYawPose後方集合に対する符号較正後平均誤差が122.134°であり、teacher ensembleの一部に大きなdomain mismatchがあります。
- YawPose rear candidateの判定自体がcanonical yawに依存するため、canonical yawが大きく誤っている場合にはcandidate集合の構成も影響を受けます。
- internal developmentのyawはrotation matrixから求めたhead-forward yawを使用し、外部評価ではmanifestのsource yawを使用するため、データセット間のyaw絶対値を同一規約のサンプル値として直接比較しません。
- DAD-3DHeads validationのrearは47件であり、rear 4区分では一部が1件または7件になります。
- `Y_top100`の固定epoch 10内部metricファイルは0 byteであり、内部developmentのepoch 10値を保存成果物から検証できません。
- 本実験は単一seed 42で実行されており、seed間分散は測定していません。

## 結論

初期checkpointから`Y_base`への既存HPE fine-tuningだけで、内部rear yaw meanは28.819°から22.811°へ、AGORA-HPE rear yaw meanは30.800°から27.720°へ低下しました。YawPose追加によるAGORA-HPE rear yaw meanの`Y_base`からの追加変化は最大0.334°であり、主要な後方改善量は`Y_base`までのfine-tuningで発生しています。

reliability rankingはyaw帯と強く関連しており、top20では97.6%が120–150°でした。`Y_top20`では内部developmentとAGORA-HPEの双方で120–150°が`Y_base`より低下する一方、150–180°は両符号で増加しました。採用範囲を広げてrear-deepの構成比を増やすと、150–180°の悪化量は縮小しました。

固定YawPose draw数によって`Y_top20`では1画像あたり平均160.5回/epochの提示が発生し、`Y_top100`の約5倍になります。この反復集中は浅い角度へのsubset偏りと同時に発生しているため、観測された角度帯別変化と整合する一方、反復だけを独立した原因として確定することはできません。

AFLW2000ではYawPose条件が`Y_base`より悪化し、300W-LPでは採用率によって変化方向が異なりました。YawPose追加の影響は後方yawだけに局所化せず、外部domain generalizationにも現れています。

## 保存された成果物

本レポートの根拠となる主要成果物は次の場所に保存されています。

- 実験計画は`docs/experiments/yawpose_rear_yaw_experiment_plan.md`です。
- reliability設定は`experiments/runs/yawpose_reliability/config.json`です。
- reliability provenanceは`experiments/runs/yawpose_reliability/provenance.json`です。
- reliability集計は`experiments/runs/yawpose_reliability/metrics/summary.json`です。
- 条件別学習設定は`experiments/runs/yawpose_rear_search/conditions/<condition>/config.json`です。
- 条件別内部学習結果は`experiments/runs/yawpose_rear_search/conditions/<condition>/metrics/`です。
- 条件別外部評価は`eval/yawpose_rear_search/conditions/`です。
- baselineとの外部SO(3)比較は`eval/yawpose_rear_search/comparisons/`です。
- full-range yaw比較は`eval/yawpose_rear_search/yaw_comparisons/`です。
- AGORA-HPE、AFLW2000、300W-LPのyaw集計は`eval/yawpose_rear_search/yaw_summary_baseline_fp32_final.json`です。
- DAD-3DHeads validationのyaw集計は`eval/yawpose_rear_search/yaw_summary_baseline_dad_fp32_final_dad.json`です。
