# YawPose後方Yaw追加学習 実験計画

## 1. 目的

本実験の目的は、合成Head Pose EstimationデータセットであるYawPoseを追加学習データとして利用し、SixDRepNet360-ResNet50の後方姿勢におけるyaw推定精度を改善できるかを検証することです。

YawPoseはyaw推定を主目的として構成された合成データセットです。pitchは一部サンプルにのみ存在し、roll教師はありません。本実験ではこの不完全な姿勢教師を補完せず、YawPose由来サンプルからはyawだけを学習対象とします。VGGHeadsおよびDAD-3DHeads由来の既存HPE学習データについては、yaw・pitch・rollを含む完全な姿勢教師を使用します。

YawPoseには、画像の左右反転とyawアノテーションの不一致、ならびに画像内容とyawアノテーションの角度ずれが含まれることが手動確認で判明しています。後方帯について行った限定的な目視確認では、確認対象の3〜4割程度に何らかの問題が見つかっています。この比率は系統的な無作為抽出による推定値ではなく、データ品質上の問題を把握するための観察値です。

また、YawPoseの全角度帯をyaw教師として追加したローカル先行実験ではAFLW2000の評価性能低下が確認されています。この先行実験は本リポジトリの正式な実験成果物には含まれておらず、本計画の後方限定・信頼度選択条件とも異なるため、YawPoseを無条件に追加した場合のリスクを示す動機としてのみ扱います。

このため、本実験ではYawPoseを一律に教師データとして使用しません。複数の既存HPEモデルを固定teacherとして使用し、YawPoseのcanonical yawをどの程度信頼できるかを相対順位として学習前に事前計算します。その順位の上位20%、40%、60%、80%、100%をそれぞれ独立した学習条件として比較し、後方yaw改善と既存性能保持の関係を評価します。

teacherの予測yawは擬似GTとして使用しません。teacherはYawPoseラベルの採否を決めるための信頼度導出にのみ使用します。

## 2. 背景

### 2.1 対象モデル

対象モデルはSixDRepNet360-ResNet50です。モデル構造は変更せず、ResNet50の`layer4`とrotation regression headを更新対象とします。

初期checkpointには次の配布checkpointを使用します。

| 項目 | 値 |
|---|---|
| path | `checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` |
| SHA-256 | `3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6` |

このcheckpointは`docs/experiments/rear_sampling_flip_consistency_tradeoff.md`で比較した9条件と同じ初期checkpointです。

### 2.2 既存HPE学習データ

基礎学習データにはVGGHeadsとDAD-3DHeads trainを使用します。これらのデータでは完全な回転行列教師が存在するため、SO(3) geodesic lossで学習します。

rear samplingとflip-consistencyの比較結果は`docs/experiments/rear_sampling_flip_consistency_tradeoff.md`に記録されています。この実験では、VGGHeadsとDAD-3DHeads trainを合わせた学習データにおける自然な後方比率は1.743%でした。

自然後方比率にSO(3) flip-consistency重み0.2を組み合わせた条件では、AGORA-HPE後方の平均SO(3) geodesic errorが37.900°から34.819°へ低下しました。一方、既存rearを5%または25%までoversamplingするとAGORA-HPE後方はさらに低下しましたが、AGORA-HPE正面や300W-LP側面の誤差増加も確認されました。

YawPose自体が追加の後方学習信号になるため、本実験では既存HPEデータ側のrear samplingを自然比率1.743%に固定します。

### 2.3 YawPose

YawPoseは`PINTO0309/YawNet`の`resources` releaseで公開されている合成頭部yawデータセットです。本実験では、次の公開アーカイブを入力データの基準とします。

| 項目 | 値 |
|---|---|
| upstream repository | `https://github.com/PINTO0309/YawNet` |
| dataset documentation revision | `4af7fa9d73e94790688c518376d3be7c41c74cc3` |
| release | `resources` |
| archive | `yawpose.tar.gz` |
| archive SHA-256 | `08df8f2e5df8d2c5c0509475688cf1366c69b96b393f1cbbbf87aeb9ced9118a` |
| published cleaned labels | `labels_fixed.jsonl`、42,135件 |
| image crops | 43,258件 |
| image size | 320×320 |

YawPoseの公開yawは`yaw_deg ∈ [0, 360)`で、0°が正面、+90°がviewer-leftです。公開データは`synthetic_001`から`synthetic_006`までの生成sourceを含みます。本計画でいう`YawPose生成source`または`source`は、この公開source区分を指します。

本実験の未修正ラベルの基準値には`labels_fixed.jsonl`の`yaw_deg`を使用します。公開`labels_fixed.jsonl`にはYawNet側で実施されたラベルcleaningが既に反映されており、`synthetic_001`の`intent_rear` 5,890件のうち2,655件は公開データ作成時にDINOv3 teacherで再ラベルされています。この上流処理は公開データセットのprovenanceとして扱い、本実験のteacher ensembleによる追加のラベル置換とは区別します。本実験では`teacher_yaw`または`yaw_deg_orig`を用いて公開ラベルを再置換しません。

人手修正は`birdManIkioiShota/yawpose-annotator`が生成する`manual_corrections.jsonl`として公開ラベルとは別に保持します。信頼度事前計算を開始する時点で使用するcorrection manifestを固定し、SHA-256をprovenanceへ保存します。correction manifestが空の場合も空manifestを明示的に入力として固定します。

YawPoseにはpitch情報を持つサンプルがありますが、本実験では使用しません。YawPose streamから学習に使用する教師信号はcanonical yawだけです。

公開データセットの仕様は`https://github.com/PINTO0309/YawNet/blob/4af7fa9d73e94790688c518376d3be7c41c74cc3/docs/yawpose_dataset.md`を参照します。手動修正ツールは`https://github.com/birdManIkioiShota/yawpose-annotator`です。

## 3. 実験仮説

本実験で検証する中心仮説は、YawPose後方データによる性能低下の一因がyawラベルノイズであり、複数teacherとYawPose canonical yawの整合度から相対的に信頼できるサンプルを優先して学習へ投入することで、YawPoseの後方画像をrear yaw改善へ利用できるというものです。

この仮説を検証するため、teacher予測による再ラベルは行わず、YawPose yaw lossの重み、既存HPE学習条件、optimizer、epoch数を固定します。主に変更する変数は、事前計算した信頼度順位のどこまでを学習へ採用するかという上位割合です。

## 4. 用語

本実験で使用する主要な用語を定義します。

| 用語 | 定義 |
|---|---|
| HPE | Head Pose Estimationの略称です。画像から頭部姿勢を推定するタスクを指します。 |
| yaw | 本実験では頭部のhead-forward方向の水平方位角を指します。具体的な座標規約は第5.1節で定義します。 |
| pitch / roll | 頭部姿勢を構成するyaw以外の回転成分です。existing HPE dataでは学習対象ですが、YawPose由来データでは教師信号として使用しません。 |
| existing HPE data | VGGHeadsおよびDAD-3DHeads train由来の学習データです。完全な回転行列教師を使用します。 |
| SO(3) | 3次元回転を表す回転群です。本実験のSO(3) geodesic loss / errorは、二つの3次元回転の間の角距離を表します。 |
| base model | 学習開始時checkpointをロードしたSixDRepNet360です。retention distillationでは、このモデルを固定teacherとして使用します。 |
| retention distillation | existing HPE dataの非後方サンプルについて、学習対象モデルの予測を固定base modelの予測へ近づけ、非後方性能の変化を抑える損失です。 |
| flip-consistency | 元画像と水平反転画像から得た予測を同一座標系へ戻し、両者の整合性を要求する損失です。本実験ではexisting HPE dataの後方だけにSO(3)版を適用します。 |
| YawPose rear candidate | YawPoseのうち、canonical yawが共通yaw規約で`|yaw| >= 120°`となるサンプルです。 |
| teacher | YawPoseの学習には使用せず、YawPose canonical yawの信頼度導出にのみ使用する固定済みHPEモデルです。 |
| canonical yaw | 学習時にYawPoseサンプルへ適用するyaw教師です。人手修正が存在する場合は修正値を使用し、それ以外はYawPoseの元yawを使用します。 |
| reliability metric | teacher予測とcanonical yawの整合度、ならびにteacher間の一致度を表す事前計算値です。 |
| reliability score | 複数のreliability metricの相対順位を統合した0〜1の順位scoreです。確率ではなく、小さいほど相対的に信頼度が高い値とします。 |
| reliability ranking | reliability scoreの昇順でYawPose rear candidateを並べた相対順位です。 |
| adoption ratio | reliability rankingの上位から学習へ採用する割合です。本実験では20%、40%、60%、80%、100%を比較します。 |
| retention | 後方性能改善によって正面・側面性能を悪化させないための保持条件です。 |

## 5. yaw規約と姿勢区分

### 5.1 共通yaw規約

YawPoseの公開`yaw_deg`を本実験のsigned yawへ変換する規則を固定します。

`yaw_signed = ((yaw_deg + 180) mod 360) - 180`

値域は`[-180°, 180°)`です。0°は正面、+90°はviewer-left、-90°はviewer-rightです。公開YawPoseの`yaw_deg`を使用する場合、この変換以外の符号反転やオフセット補正を追加しません。

SixDRepNet360など回転行列を出力するモデルでは、head-localの`+Z`軸をhead-forward vectorとして取り出します。

`v = R * [0, 0, 1]^T`

`v = [v_x, v_y, v_z]^T`としたとき、raw azimuthを`atan2(v_x, v_z)`で求めます。teacher adapterは、このraw azimuthまたは各モデル固有のyaw出力をYawPoseのsigned yaw規約へ変換します。

teacherごとのadapterは、少なくとも0°、+90°、-90°、+150°、-150°の既知姿勢fixtureで符号と周期境界を検証します。fixtureと変換結果をteacher manifestへ保存し、YawPose signed yawとの方向対応が確認できないteacherは信頼度事前計算へ使用しません。

### 5.2 YawPoseの学習対象

YawPoseでは後方帯だけを学習対象とします。canonical yawを共通yaw規約へ変換した後、`|yaw| >= 120°`をYawPose rear candidateとします。

前方および側面のYawPoseサンプルは本実験では使用しません。

rear candidateの判定自体もcanonical yawに依存するため、canonical yawが大きく誤っている場合には、本来rearではないサンプルが候補へ入る、または本来rearであるサンプルが候補から外れる可能性があります。teacherによる再ラベルを行わない本実験では、この点を既知の制約として扱います。

### 5.3 後方診断区分

後方内部の偏りを確認するため、YawPose rear candidateを4区分に分けて統計を記録します。

| 区分 | yaw範囲 |
|---|---|
| negative rear-near | `-150° < yaw <= -120°` |
| negative deep-rear | `-180° <= yaw <= -150°` |
| positive rear-near | `120° <= yaw < 150°` |
| positive deep-rear | `150° <= yaw <= 180°` |

この4区分は信頼度順位や学習後性能に左右方向・角度帯の偏りが発生していないかを確認するために使用します。

## 6. 学習損失

総損失は概念的に次式で構成します。

`L_total = L_original_supervised + 1.0 * L_distillation + 0.2 * L_original_flip + 0.2 * L_yawpose`

各損失項の意味を示します。

| 損失 | 適用対象 | 内容 |
|---|---|---|
| `L_original_supervised` | existing HPE data全体 | 正解回転行列に対するSO(3) geodesic lossです。 |
| `L_distillation` | existing HPE dataの非後方 | 学習開始時のbase model予測に対するretention distillationです。 |
| `L_original_flip` | existing HPE dataの後方 | 左右反転前後の予測回転に対するSO(3) flip-consistency lossです。 |
| `L_yawpose` | 選択されたYawPose rear candidate | canonical yawだけを教師にしたyaw-only lossです。 |

YawPose yaw lossの重み`lambda_yawpose`は0.2に固定します。この0.2はYawPoseサンプルを20%採用することを意味せず、YawPose yaw lossを総損失へ加える際の損失重み係数です。

YawPose streamから計算する損失は`L_yawpose`だけです。既存HPE dataに適用するSO(3) supervised loss、retention distillation、rear flip-consistencyはYawPose sampleには適用しません。

## 7. YawPose yaw loss

SixDRepNet360の出力形式は変更しません。モデルが出力した回転行列`R_pred`から第5.1節と同じhead-forward azimuthを求め、予測yawとして使用します。

予測yawを`y_pred`、canonical yawを`y_gt`としたとき、周期境界を考慮した角度差を次式で求めます。

`delta = atan2(sin(y_pred - y_gt), cos(y_pred - y_gt))`

YawPose yaw lossは次式で定義します。

`L_yawpose = mean(abs(delta))`

loss計算ではradianを使用します。

YawPoseサンプルからpitchおよびroll targetは生成しません。YawPose由来の教師信号はyawだけです。

## 8. teacher ensemble

### 8.1 目的

teacher ensembleの目的はYawPose canonical yawを書き換えることではなく、各YawPose rear candidateのcanonical yawが相対的にどの程度信頼できるかを評価することです。

### 8.2 teacher構成

信頼度事前計算に使用するteacher ensembleは3モデルに固定します。YawPoseで学習されたモデルは、監査対象ラベルとの独立性を保つためteacher ensembleへ含めません。

| teacher ID | モデル | 固定する実装 | checkpoint |
|---|---|---|---|
| `sixdrepnet360_base` | SixDRepNet360-ResNet50 | 本リポジトリの評価実装 | `checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth`、SHA-256 `3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6` |
| `semiuhpe_effnetv2s` | SemiUHPE EfficientNetV2-S full-range | `hnuzhy/SemiUHPE@c8f67102bf5aba8869b3f23453ac67599f21aa1f` | upstream公開名 `DAD-WildHead-EffNetV2-S-best.pth` |
| `whenet` | WHENet | `Ascend-Research/HeadPoseEstimation-WHENet@a0d7bdfb5e2ac97ae6b0ae3eef79fdcf4075ab82` | upstream `WHENet.h5` |

SemiUHPEとWHENetのcheckpointは取得後にSHA-256を計算し、teacher manifestへ固定します。信頼度事前計算は、3モデルすべての実装revision、checkpoint SHA-256、入力前処理、yaw adapter、fixture検証結果がmanifestへ記録されるまで開始しません。以後の全YawPose条件は同じteacher manifestを使用します。

3モデルは、6D rotation regression、SemiUHPEのfull-range unconstrained HPE、WHENetのwide-range Euler yawという異なるモデル系統を含めます。reliability scoreはこの3モデルの出力を同一重みで扱います。

### 8.3 座標規約

teacherごとにyawの符号、範囲、Euler角の定義が異なる可能性があるため、teacher出力を直接比較しません。

各teacherの出力は第5.1節の共通yaw規約へ変換し、少数の既知姿勢サンプルで正面、左右側面、左右後方の対応を確認します。

## 9. 信頼度の事前計算

### 9.1 実行時期

信頼度導出は学習前に一度だけ実施します。

全YawPose rear candidateについて固定teacher ensembleによる推論を完了し、その結果からreliability scoreとreliability rankingを作成します。

学習中にteacher推論を実行しません。学習対象モデルの予測、学習loss、epoch、checkpointによって信頼度を更新することもありません。

すべてのYawPose学習条件は同一の事前計算成果物を参照します。

### 9.2 角度距離

YawPose sample `i`のcanonical yawを`y_i`、teacher `j`のyaw予測を`p_ij`とします。

角度差には円周距離を使用します。

`d_circ(a, b) = abs(atan2(sin(a - b), cos(a - b)))`

### 9.3 事前計算指標

各YawPose rear candidateについて少なくとも次の値を計算します。

| 指標 | 定義 |
|---|---|
| `gt_median_error` | 各teacherとcanonical yawの角度差のmedianです。 |
| `gt_max_error` | 各teacherとcanonical yawの角度差の最大値です。 |
| `teacher_dispersion` | teacher同士のpairwise yaw差のmedianです。 |
| `agree_10` | canonical yawから10°以内に入るteacherの割合です。 |
| `agree_20` | canonical yawから20°以内に入るteacherの割合です。 |
| `agree_30` | canonical yawから30°以内に入るteacherの割合です。 |

teacher ensembleの平均yaw、median yaw、その他の集約yawを学習用GTとして使用しません。

## 10. reliability scoreとranking

### 10.1 性質

reliability scoreは確率ではありません。異なる単位・分布を持つ複数のreliability metricを、YawPose rear candidate集合内の相対順位へ変換して統合するためのscoreです。

### 10.2 metricごとのpercentile rank

`gt_median_error`、`teacher_dispersion`、`gt_max_error`について、YawPose rear candidate全体の中で小さい値を高信頼側としてpercentile rankへ変換します。

各percentile rankを`r_median`、`r_dispersion`、`r_max`とし、値域を`[0, 1]`とします。0に近いほど、そのmetricについて相対的に良好なサンプルです。

### 10.3 score

reliability scoreは次式で定義します。

`reliability_score = (r_median + r_dispersion + r_max) / 3`

3指標を等重みとし、teacherとcanonical yawの典型的な一致度、teacher同士の一致度、単一teacherの大きな外れを同時にrankingへ反映します。

`agree_10`、`agree_20`、`agree_30`はscoreには含めず、rankingの診断指標として保存します。

### 10.4 ranking

reliability rankingは`reliability_score`の昇順で決定します。同一scoreの場合は`gt_median_error`、`teacher_dispersion`、`gt_max_error`、sample IDの順でtie-breakし、順位を再現可能にします。

最終順位は全rear candidate数に対するranking percentileとしても保存します。このpercentileも信頼確率ではなく、単なる順位位置です。

## 11. 人手修正データ

canonical yawは、公開`labels_fixed.jsonl`の`yaw_deg`を第5.1節の規約へ変換した値を基準とし、固定した`manual_corrections.jsonl`に同一sampleのyaw修正がある場合だけ修正後yawで上書きします。

人手修正済みサンプルもteacher ensembleによる同じ事前計算を行い、自動的にranking最上位へ配置する特別処理は行いません。

人手修正済み集合は、主として問題が疑われたサンプルを含む可能性があるため、YawPose全体のclean/noisy比率を推定する代表標本とはみなしません。rankingの上位割合ごとに、人手修正前後のyaw差や既知問題サンプルがどの順位へ配置されたかを診断するために使用します。

公開`labels_fixed.jsonl`、固定した`manual_corrections.jsonl`、両者を統合して生成するcanonical manifestのSHA-256をすべて保存し、学習条件間でcanonical yawが変化しないことを検査します。

## 12. 事前計算成果物

信頼度事前計算は学習runとは独立した成果物として保存します。

サンプル単位では少なくとも次の情報を保存します。

- sample ID
- image path
- YawPose生成source
- original yaw
- canonical yaw
- human-corrected flag
- rear bucket
- 各teacherのyaw予測
- 各teacherのcanonical yawとの差
- `gt_median_error`
- `gt_max_error`
- `teacher_dispersion`
- `agree_10`
- `agree_20`
- `agree_30`
- `r_median`
- `r_dispersion`
- `r_max`
- reliability score
- reliability rank
- reliability percentile

事前計算runのprovenanceには少なくとも次の情報を保存します。

- YawPose入力manifest hash
- `manual_corrections.jsonl` hash
- teacher一覧
- teacher checkpoint hash
- teacher実装revision
- yaw座標変換規約
- 信頼度計算コードrevision
- ranking規則
- 生成日時

学習runではこの成果物を読み込み、teacherをロードしません。

## 13. YawPose採用条件

信頼度順位の上位割合を20%ずつ変化させます。

| 条件ID | YawPose採用条件 |
|---|---|
| `Y_base` | YawPoseを使用しません。 |
| `Y_top20` | reliability ranking上位20%を使用します。 |
| `Y_top40` | reliability ranking上位40%を使用します。 |
| `Y_top60` | reliability ranking上位60%を使用します。 |
| `Y_top80` | reliability ranking上位80%を使用します。 |
| `Y_top100` | YawPose rear candidateをすべて使用します。 |

subsetは必ずnested subsetにします。

`Y_top20 ⊂ Y_top40 ⊂ Y_top60 ⊂ Y_top80 ⊂ Y_top100`

この構造により、順位の低い20%区間を追加するたびに性能がどのように変化するかを比較できます。

## 14. 学習時のサンプリング

### 14.1 データstream

existing HPE dataとYawPoseは別streamとして扱います。

各micro-batchではexisting HPE dataから64サンプルを取得します。YawPoseを使用する条件では、同時に選択済みYawPose subsetから64サンプルを取得し、yaw-only lossを計算します。

概念的には各micro-batchで次式を計算します。

`L_micro = L_original_batch + 0.2 * L_yawpose_batch`

gradient accumulationを2とするため、1 optimizer updateあたりのeffective sample数はexisting HPE dataが128、YawPoseを使用する条件ではYawPoseも128です。

YawPose batch内では`L_yawpose`をmean reductionします。

### 14.2 条件間の学習量

`Y_top20`から`Y_top100`まで、1 optimizer updateあたりのYawPose batch sizeとYawPose loss weightを固定します。

したがって、採用割合を増やしても1 updateあたりのYawPose loss量を意図的に増やしません。条件間で主に変化するのはYawPose候補poolの範囲です。

高信頼度条件ほどpoolが小さくなるため、同一draw数ではrepeat回数が増加します。各runでは次の値を必ず記録します。

- selected unique samples
- total YawPose draws
- repeated draws
- repeat ratio
- rear bucket別draw数
- source別draw数

## 15. 共通学習条件

YawPose採用割合以外の主要条件を固定します。

| 項目 | 設定 |
|---|---|
| model | SixDRepNet360-ResNet50 |
| initial checkpoint | `checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` |
| existing train data | VGGHeads + DAD-3DHeads train |
| existing rear sampling | 自然比率1.743% |
| update scope | ResNet50 `layer4` + regression head |
| epochs | 10 |
| existing HPE micro-batch size | 64 |
| YawPose micro-batch size | 64 |
| gradient accumulation | 2 |
| existing HPE effective batch size | 128 |
| YawPose effective batch size | 128 |
| precision | BF16 |
| optimizer | AdamW |
| backbone learning rate | `1e-6` |
| regression head learning rate | `3e-5` |
| weight decay | `1e-4` |
| warmup updates | 100 |
| gradient clip norm | 1.0 |
| original supervised weight | 1.0 |
| retention distillation weight | 1.0 |
| existing rear flip-consistency weight | 0.2 |
| YawPose yaw loss weight | 0.2 |
| seed | 42 |

すべての条件は同一の初期checkpointから独立に開始します。

## 16. baseline条件

`Y_base`はYawPose対応後の同一コードパスでYawPose streamを無効化して実行します。

`Y_base`が`docs/experiments/rear_sampling_flip_consistency_tradeoff.md`の自然後方比率・flip-consistency重み0.2条件と整合することを確認し、YawPose対応実装そのものによる性能変化とYawPose追加による性能変化を分離します。

## 17. checkpoint比較

主比較には全条件の固定epoch 10 checkpointを使用します。

条件ごとに異なるbest epochを主比較へ使用すると、YawPose採用割合の効果とcheckpoint選択時期の効果が混在するためです。

内部best checkpointは補助成果物として第18節のselection規則に従って保存します。

## 18. 内部checkpoint selection

内部development splitにはVGGHeadsおよびDAD-3DHeads trainから分離した既存developmentデータを使用します。rear sampling / flip-consistency比較実験と同じsplitを再利用し、YawPoseを内部developmentへ混入させません。

学習開始前baselineに対して、候補checkpointには次のretention条件を要求します。

- front meanが悪化しないこと
- front P90が悪化しないこと
- side meanが悪化しないこと
- side P90が悪化しないこと

保持条件を満たすcheckpointは次の優先順位で比較します。

1. rear 4区分のworst P90
2. rear全体P90
3. rear 4区分のworst `>90° rate`
4. rear全体mean

外部benchmarkはcheckpoint selectionには使用しません。

## 19. 評価

### 19.1 rear yaw評価

YawPoseはyawだけを追加教師として使用するため、SO(3) geodesic errorだけではなくyaw errorを独立して記録します。

internal development splitおよびAGORA-HPEでは少なくとも次の指標を記録します。

- rear yaw mean absolute angular error
- rear yaw median
- rear yaw P90
- rear yaw `>30° rate`
- rear yaw `>60° rate`
- rear yaw `>90° rate`
- rear 4区分ごとのyaw mean
- rear 4区分ごとのyaw P90

internal developmentでは回転行列から第5.1節のhead-forward azimuthを算出します。外部benchmarkでは各評価manifestに保存されたsource yawを使用するため、internal developmentのazimuthと外部source yawは数値規約が完全に一致するとは限りません。両者は同一サンプル単位の角度値として直接比較せず、それぞれのデータセット内でcandidateとbaselineを比較します。

### 19.2 rotation全体の評価

yaw改善がpitch・rollを含む回転全体の悪化を伴っていないかを確認するため、SO(3)指標も記録します。

- rear geodesic mean
- rear geodesic P90
- rear geodesic `>90° rate`
- front mean / P90
- side mean / P90

### 19.3 外部評価データ

外部評価には次のデータセットを使用します。

| データセット | 主な用途 |
|---|---|
| AGORA-HPE | 全周姿勢を含むため、rear yawおよびrear SO(3)性能の主要評価に使用します。 |
| AFLW2000 | 主としてfront / side性能保持の確認に使用します。 |
| 300W-LP | 主としてfront / side性能保持の確認に使用します。 |
| DAD-3DHeads validation | 独立validationとして全体、front、side、rearを確認します。 |

DAD-3DHeads validationのrearサンプル数は少ないため、rear単独値は補助結果として扱います。

外部評価データは学習、信頼度計算、checkpoint selectionには使用しません。

## 20. YawPose subset診断

各採用条件について、学習前にsubset自体の統計を保存します。

| 統計 | 確認目的 |
|---|---|
| selected sample count | 採用データ量を確認します。 |
| rear bucket別件数 | 角度帯の偏りを確認します。 |
| positive / negative比率 | 左右方向の偏りを確認します。 |
| source別件数 | `synthetic_001`〜`synthetic_006`の公開生成sourceへの偏りを確認します。 |
| `gt_median_error`分布 | teacherとcanonical yawの一致度を確認します。 |
| `teacher_dispersion`分布 | teacher間一致度を確認します。 |
| `gt_max_error`分布 | 一部teacherだけが大きく外れるケースを確認します。 |
| `reliability_score`分布 | 各採用条件がどの順位範囲を含むかを確認します。 |
| manual correctionとの関係 | rankingが既知のラベル問題をどの程度下位へ送れているかを確認します。 |

上位割合を絞ることで特定yaw帯や特定sourceだけが残っていないかも確認します。

## 21. 結果の比較方法

主要比較対象は`Y_base`、`Y_top20`、`Y_top40`、`Y_top60`、`Y_top80`、`Y_top100`の固定epoch 10 checkpointです。

結果を単一の総合scoreへ集約せず、次の関係を個別に確認します。

`adoption ratio -> YawPose subset quality -> rear yaw -> rear SO(3) -> front/side retention`

たとえば`Y_top20`から`Y_top60`までrear yawが改善し、`Y_top80`で悪化した場合は、60〜80%区間で追加された低順位データの利益よりラベルノイズの影響が大きくなった可能性を検討します。

`Y_top100`まで単調に改善した場合は、低順位データを含めても追加rear dataの効果が優勢だったことを示します。

`Y_top20`が`Y_base`より悪化した場合は、yawラベル品質だけでは説明できず、合成画像domain shift、YawPose exposure、loss weightなどを別要因として切り分けます。

## 22. 実験成果物

本実験では少なくとも次の成果物を保存します。

### 信頼度事前計算

- teacher ensemble manifest
- teacher checkpoint hashes
- YawPose input manifest hash
- `manual_corrections.jsonl` hash
- sample-level teacher predictions
- sample-level reliability metrics
- sample-level reliability score
- final reliability ranking
- adoption ratioごとのsubset manifest
- subset summary

### 学習run

- config
- provenance
- epochごとのtrain metrics
- epochごとのdevelopment metrics
- fixed epoch 10 checkpoint
- internal best checkpoint
- YawPose draw / repeat statistics

### 外部評価

- AGORA-HPE evaluation
- AFLW2000 evaluation
- 300W-LP evaluation
- DAD-3DHeads validation evaluation
- baselineとのpaired comparison

## 23. 判定対象

本実験では、単一の数値で採用条件を自動決定しません。

各adoption ratioについて、少なくとも次の三つの観点を分離して比較します。

1. rear yaw errorがどの程度変化したかを確認します。
2. rear SO(3) errorがどの程度変化したかを確認します。
3. front / sideおよび外部データセットに退行が発生したかを確認します。

これにより、YawPoseの信頼度順位をどこまで広げると後方yaw改善が得られ、どこからラベルノイズまたはdomain shiftの影響が支配的になるかを評価します。
