# YawPose後方Yaw 層別single-draw実験計画

## 目的

本実験では、YawPose後方yaw追加学習において、reliability採用率とyaw角度分布、同一サンプルの反復回数が混在していた既存実験を再設計します。

YawPoseは後方サンプルだけを使用し、教師信号にはcanonical yawだけを使用します。teacher ensembleはcanonical yawの置換には使用せず、サンプルの相対的なreliability rankingを作るためだけに使用します。

また、既存HPE学習データではVGGHeadsを全条件で使用し、DAD-3DHeads trainを追加する条件と追加しない条件を分けます。これにより、DAD-3DHeadsを学習データから除外した場合の性能差を同一実験内で確認します。

## 初期モデル

全条件は同一のaudited base checkpointから独立に開始します。

- model: SixDRepNet360-ResNet50
- update scope: ResNet50 layer4 + regression head
- YawPose loss weight: 0.2
- non-rear retention distillation weight: 1.0
- existing-HPE rear flip-consistency weight: 0.2
- epochs: 10
- batch size: 64
- gradient accumulation: 2

固定epoch 10 checkpointを主要比較対象とします。

## YawPose候補

YawPoseのcanonical yawが `|yaw| >= 120°` のサンプルだけを候補とします。

yaw規約は `[-180°, 180°)` とし、後方候補を次の8区分へ分けます。

| 区分 | yaw範囲 |
|---|---|
| negative 165–180 | `[-180°, -165°)` |
| negative 150–165 | `[-165°, -150°)` |
| negative 135–150 | `[-150°, -135°)` |
| negative 120–135 | `[-135°, -120°]` |
| positive 120–135 | `[120°, 135°)` |
| positive 135–150 | `[135°, 150°)` |
| positive 150–165 | `[150°, 165°)` |
| positive 165–180 | `[165°, 180°)` |

`180°` はsigned yaw変換後に `-180°` としてnegative 165–180区分へ入ります。

## Reliability ranking

teacher構成、teacher出力のyaw変換、raw reliability metricは既存YawPose実験と同じものを使用します。

各サンプルについて次の3指標を使用します。

- `gt_median_error_deg`
- `teacher_dispersion_deg`
- `gt_max_error_deg`

percentile rankは14,049件全体では計算しません。8個のyaw区分ごとに独立して計算します。

各区分内でreliability scoreを昇順に並べ、top20、top40、top60、top80、top100をそれぞれ選択します。最終subsetは8区分の選択結果の和集合とします。

各yaw区分について次のnested関係を維持します。

`top20 ⊂ top40 ⊂ top60 ⊂ top80 ⊂ top100`

## YawPose sampling

選択されたYawPoseサンプルは、各epochで1回だけdrawします。

subsetの大きさに合わせて1 epochのYawPose総draw数を変化させ、総draw数を条件間で固定しません。同一epoch内で同じYawPoseサンプルを反復しません。

各epochではsubset全体をseed付きでshuffleします。YawPose batchはexisting-HPE epoch全体へ決定論的に分散させます。

最後のYawPose batchが64件未満の場合、yaw lossの寄与を実サンプル数に比例させます。これにより、最後の不完全batchに含まれる各サンプルの重みを完全batchのサンプルと揃えます。

各epochで次のsampling情報を保存します。

- selected unique samples
- unique samples drawn
- total draws
- repeated draws
- repeat ratio
- source別draw数
- 15度yaw区分別draw数
- YawPose batch配置位置

## Existing-HPE data

VGGHeadsは全条件で使用します。

existing-HPE data regimeは次の2種類です。

| regime | 学習データ |
|---|---|
| `vgg_only` | VGGHeads train |
| `vgg_plus_dad` | VGGHeads train + DAD-3DHeads train |

existing-HPE側の1 epoch学習量は、VGGHeads train件数をeffective batch境界へ切り下げた値に固定します。DAD-3DHeadsを追加してもoptimizer update数は増加させません。

rear samplingは各regimeの学習poolにおける自然rear比率を使用します。

## 内部development評価

checkpoint selectionに使用する内部development集合は、両regimeともVGGHeads devへ固定します。

DAD-3DHeads devはcheckpoint selectionに使用しません。

これにより、DAD-3DHeadsの採否によって内部評価集合自体が変化することを避けます。

## 条件行列

各existing-HPE regimeについて次の6条件を実行し、合計12条件とします。

| YawPose条件 | 採用範囲 |
|---|---|
| `Y_base` | YawPoseなし |
| `Y_top20` | 各15度yaw区分のreliability上位20% |
| `Y_top40` | 各15度yaw区分のreliability上位40% |
| `Y_top60` | 各15度yaw区分のreliability上位60% |
| `Y_top80` | 各15度yaw区分のreliability上位80% |
| `Y_top100` | 全YawPose rear candidate |

生成されるcondition IDにはexisting-HPE regimeを含めます。

## 外部評価

既存のSO(3)評価、front / side / rear評価、120–150° / 150–180°の後方診断は維持します。

AGORA-HPEについては、追加で全周yawを15度単位の24区分へ分割します。

各15度区分について少なくとも次の値を保存します。

- sample count
- yaw absolute error mean
- yaw absolute error median
- yaw absolute error P90
- SO(3) geodesic error mean

AGORA-HPEでは24区分すべてを出力し、該当サンプルがない区分もcount 0として残します。

15度集計からpolar形式の誤差可視化も生成します。

AFLW2000、300W-LP、DAD-3DHeads official validationについては、15度全周集計とpolar可視化を追加しません。

## 成果物

既存実験成果物は変更しません。新しい実験は次のrun IDを標準値とします。

- reliability: `yawpose_reliability_stratified15`
- training search: `yawpose_rear_stratified_search`

学習成果物は `experiments/runs/yawpose_rear_stratified_search/`、外部評価成果物は `eval/yawpose_rear_stratified_search/` に保存します。
