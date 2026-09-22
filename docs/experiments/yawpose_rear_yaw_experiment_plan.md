# YawPose後方Yaw追加学習 実験計画

## 1. 目的

本実験の目的は、合成Head Pose EstimationデータセットであるYawPoseを追加学習データとして利用し、SixDRepNet360-ResNet50の後方姿勢におけるyaw推定精度を改善できるかを検証することです。

YawPoseはyaw推定を主目的として構成された合成データセットです。pitchは一部サンプルにのみ存在し、roll教師はありません。本実験ではこの不完全な姿勢教師を補完せず、YawPose由来サンプルからはyawだけを学習対象とします。VGGHeadsおよびDAD-3DHeads由来の既存HPE学習データについては、従来どおりyaw・pitch・rollを含む完全な姿勢教師を使用します。

YawPoseには、画像の左右反転とyawアノテーションの不一致、ならびに画像内容とyawアノテーションの角度ずれが相当数含まれることが手動確認で判明しています。特に後方帯では、目視確認した範囲で3〜4割程度に何らかの問題が見つかっています。また、YawPose全体をyaw教師として追加した先行実験ではAFLW2000の評価性能低下が確認されています。

このため、本実験ではYawPoseを一律に教師データとして使用しません。複数の既存HPEモデルを固定teacherとして使用し、YawPoseの元yawラベルをどの程度信頼できるかを相対順位として事前計算します。その順位の上位20%、40%、60%、80%、100%をそれぞれ独立した学習条件として比較し、後方yaw改善と既存性能保持の関係を評価します。

teacherの予測yawは擬似GTとして使用しません。teacherはYawPoseラベルの採否を決めるための信頼度導出にのみ使用します。

## 2. 背景

### 2.1 対象モデル

対象モデルはSixDRepNet360-ResNet50です。モデル構造は変更せず、既存の後方改善実験と同様にResNet50の`layer4`とrotation regression headを更新対象とします。

初期checkpointには、既存のrear sampling / flip-consistency比較実験と同じ配布checkpointを使用します。

### 2.2 既存HPE学習データ

基礎学習データにはVGGHeadsとDAD-3DHeads trainを使用します。これらのデータでは完全な回転行列教師が存在するため、従来どおりSO(3) geodesic lossで学習します。

既存のrear sampling / flip-consistency比較結果は`docs/experiments/rear_sampling_flip_consistency_tradeoff.md`に記録されています。この実験では、VGGHeadsとDAD-3DHeads trainを合わせた学習データにおける自然な後方比率は1.743%でした。

自然後方比率にSO(3) flip-consistency重み0.2を組み合わせた条件では、AGORA-HPE後方の平均SO(3) geodesic errorが37.900°から34.819°へ改善しました。一方、既存rearを5%または25%までoversamplingするとAGORA-HPE後方はさらに改善しましたが、AGORA-HPE正面や300W-LP側面の退行量も増加しました。

YawPose自体が追加の後方学習信号になるため、本実験では既存HPEデータ側のrear samplingを自然比率1.743%に固定します。

### 2.3 YawPose

YawPoseは約4万件規模の合成頭部画像を含む、yaw推定を主目的としたデータセットです。pitchは部分的に存在しますが、本実験では使用しません。roll教師は存在せず、本実験でも生成しません。

YawPoseの手動修正には次の補助ツールを使用できます。

https://github.com/birdManIkioiShota/yawpose-annotator

ただし、全件を人手修正することはデータ件数上現実的ではないため、本実験の基本方針は自動修正ではなく、信頼度順位に基づくサンプル選択とします。

## 3. 実験仮説

本実験で検証する中心仮説は、YawPose後方データによる性能低下の一因がyawラベルノイズであり、複数teacherとYawPose GTの整合度から相対的に信頼できるサンプルを優先して学習へ投入することで、YawPoseの後方画像をrear yaw改善へ利用できるというものです。

この仮説を検証するため、teacher予測による再ラベルは行わず、YawPose yaw lossの重み、既存HPE学習条件、optimizer、epoch数を固定します。主に変更する変数は、事前計算した信頼度順位のどこまでを学習へ採用するかという上位割合です。

## 4. 用語

本実験で使用する主要な用語を定義します。

| 用語 | 定義 |
|---|---|
| existing HPE data | VGGHeadsおよびDAD-3DHeads train由来の学習データです。完全な回転行列教師を使用します。 |
| YawPose rear candidate | YawPoseのうち、共通yaw規約へ変換後に`|yaw| >= 120°`となるサンプルです。 |
| teacher | YawPoseの学習には使用せず、YawPose GTの信頼度導出にのみ使用する固定済みHPEモデルです。 |
| canonical yaw | 学習時にYawPoseサンプルへ適用するyaw教師です。人手修正が存在する場合は修正値を使用し、それ以外はYawPoseの元yawを使用します。 |
| reliability ranking | teacherとcanonical yawの整合度から事前計算する、YawPose rear candidate間の相対順位です。確率ではありません。 |
| adoption ratio | reliability rankingの上位から学習へ採用する割合です。本実験では20%、40%、60%、80%、100%を比較します。 |
| retention | 後方性能改善によって正面・側面性能を悪化させないための保持条件です。 |

## 5. 姿勢区分

### 5.1 YawPoseの学習対象

YawPoseでは後方帯だけを学習対象とします。共通のsigned yawへ変換した後、`|yaw| >= 120°`をYawPose rear candidateとします。

前方および側面のYawPoseサンプルは本実験では使用しません。

### 5.2 後方診断区分

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

YawPose由来データにはflip-consistencyを適用しません。yaw-only flip-consistencyも使用しません。

## 7. YawPose yaw loss

SixDRepNet360の出力形式は変更しません。モデルが出力した回転行列`R_pred`からhead-localの`+Z`方向をhead-forward vectorとして取り出し、そのXZ平面上のazimuthを予測yawとして使用します。

予測yawを`y_pred`、canonical yawを`y_gt`としたとき、周期境界を考慮した角度差を次式で求めます。

`delta = atan2(sin(y_pred - y_gt), cos(y_pred - y_gt))`

YawPose yaw lossは次式で定義します。

`L_yawpose = mean(abs(delta))`

loss計算ではradianを使用します。

YawPoseサンプルからpitchおよびroll targetは生成しません。YawPose由来の教師信号はyawだけです。

## 8. teacher ensemble

### 8.1 目的

teacher ensembleの目的はYawPose GTを書き換えることではなく、各YawPose rear candidateのcanonical yawが相対的にどの程度信頼できるかを評価することです。

### 8.2 teacher構成

3〜5種類のfull-rangeまたはwide-range HPEモデルを使用します。teacher間の誤差相関を下げるため、可能な範囲でarchitecture、学習データ、pose representationが異なるモデルを組み合わせます。

teacher集合は信頼度事前計算を開始する前に固定します。各teacherについてモデル名、実装revision、checkpoint、checkpoint hash、入力前処理、yaw変換規約を記録します。

YawPoseを学習データとして使用してfine-tuneされたモデルは、独立teacherとしては使用しません。

### 8.3 座標規約

teacherごとにyawの符号、範囲、Euler角の定義が異なる可能性があるため、teacher出力を直接比較しません。

各teacherの出力は事前に本プロジェクトの共通signed yaw規約へ変換し、少数の既知姿勢サンプルで正面、左右側面、左右後方の対応を確認します。

## 9. 信頼度の事前計算

### 9.1 実行時期

信頼度導出は学習前に一度だけ実施します。

全YawPose rear candidateについて固定teacher ensembleによる推論を完了し、その結果からreliability rankingを作成します。

学習中にteacher推論を実行しません。studentの予測、学習loss、epoch、checkpointによって信頼度を更新することもありません。

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

## 10. reliability ranking

### 10.1 性質

reliability rankingは確率ではありません。YawPose rear candidateを「canonical yawを相対的に信頼しやすい順」に並べるための順序です。

### 10.2 順位規則

基本順位は次の辞書式ソートで決定します。

1. `gt_median_error`が小さいサンプルを上位とします。
2. `gt_median_error`が同値の場合は`teacher_dispersion`が小さいサンプルを上位とします。
3. さらに同値の場合は`gt_max_error`が小さいサンプルを上位とします。
4. すべて同値の場合はsample IDで決定し、順位を再現可能にします。

この規則により、複数teacherがcanonical yawと近く、かつteacher同士も一致しているサンプルが上位になります。

最終順位は全rear candidate数に対するpercentileとしても保存します。このpercentileは信頼確率ではなく、単なる順位位置です。

## 11. 人手修正データ

`yawpose-annotator`で人手修正済みのサンプルについては、修正後yawをcanonical yawとして使用します。

人手修正済みサンプルもteacher ensembleによる同じ事前計算を行い、自動的にranking最上位へ配置する特別処理は行いません。

人手修正済みサンプルは、reliability rankingと実際のラベル品質の関係を診断するためにも使用します。rankingの上位割合ごとに、人手修正前後のyaw差や問題サンプル含有率を集計します。

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
- reliability rank
- reliability percentile

事前計算runのprovenanceには少なくとも次の情報を保存します。

- YawPose入力manifest hash
- manual correction file hash
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

`Y_top20 subset Y_top40 subset Y_top60 subset Y_top80 subset Y_top100`

この構造により、順位の低い20%区間を追加するたびに性能がどのように変化するかを比較できます。

## 14. 学習時のサンプリング

### 14.1 データstream

existing HPE dataとYawPoseは別streamとして扱います。

各optimizer updateではexisting HPE batchから従来損失を計算し、YawPoseを使用する条件では別のYawPose batchからyaw-only lossを計算します。

概念的には次式になります。

`L_step = L_original_batch + 0.2 * L_yawpose_batch`

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

## 15. YawPoseで使用しない処理

本実験ではYawPose由来データに対して次の処理を行いません。

- pitch教師を使用しません。
- roll教師を生成しません。
- SO(3) supervised lossを使用しません。
- SO(3) flip-consistencyを使用しません。
- yaw-only flip-consistencyを使用しません。
- teacher予測yawへのpseudo-label置換を行いません。
- teacher予測だけを根拠にした自動yaw修正を行いません。
- teacher予測だけを根拠にした左右反転修正を行いません。
- 学習中にreliability rankingを更新しません。

## 16. 共通学習条件

YawPose採用割合以外の主要条件を固定します。

| 項目 | 設定 |
|---|---|
| model | SixDRepNet360-ResNet50 |
| initial checkpoint | rear sampling / flip-consistency既存実験と同一 |
| existing train data | VGGHeads + DAD-3DHeads train |
| existing rear sampling | 自然比率1.743% |
| update scope | ResNet50 `layer4` + regression head |
| epochs | 10 |
| batch size | 64 |
| gradient accumulation | 2 |
| effective batch size | 128 |
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

## 17. baseline条件

`Y_base`はYawPose対応後の同一コードパスでYawPose streamを無効化して実行します。

`Y_base`が既存の自然後方比率・flip-consistency重み0.2条件と整合することを確認し、YawPose対応実装そのものによる性能変化とYawPose追加による性能変化を分離します。

## 18. checkpoint比較

主比較には全条件の固定epoch 10 checkpointを使用します。

条件ごとに異なるbest epochを主比較へ使用すると、YawPose採用割合の効果とcheckpoint選択時期の効果が混在するためです。

内部best checkpointは補助成果物として既存のselection規則に従って保存します。

## 19. 内部checkpoint selection

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

## 20. 評価

### 20.1 rear yaw評価

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

### 20.2 rotation全体の評価

yaw改善がpitch・rollを含む回転全体の悪化を伴っていないかを確認するため、従来のSO(3)指標も記録します。

- rear geodesic mean
- rear geodesic P90
- rear geodesic `>90° rate`
- front mean / P90
- side mean / P90

### 20.3 外部評価データ

外部評価には次のデータセットを使用します。

| データセット | 主な用途 |
|---|---|
| AGORA-HPE | 全周姿勢を含むため、rear yawおよびrear SO(3)性能の主要評価に使用します。 |
| AFLW2000 | 主としてfront / side性能保持の確認に使用します。 |
| 300W-LP | 主としてfront / side性能保持の確認に使用します。 |
| DAD-3DHeads validation | 独立validationとして全体、front、side、rearを確認します。 |

DAD-3DHeads validationのrearサンプル数は少ないため、rear単独値は補助結果として扱います。

## 21. YawPose subset診断

各採用条件について、学習前にsubset自体の統計を保存します。

| 統計 | 確認目的 |
|---|---|
| selected sample count | 採用データ量を確認します。 |
| rear bucket別件数 | 角度帯の偏りを確認します。 |
| positive / negative比率 | 左右方向の偏りを確認します。 |
| source別件数 | 特定生成sourceへの偏りを確認します。 |
| `gt_median_error`分布 | teacherとcanonical yawの一致度を確認します。 |
| `teacher_dispersion`分布 | teacher間一致度を確認します。 |
| `gt_max_error`分布 | 一部teacherだけが大きく外れるケースを確認します。 |
| manual correctionとの関係 | rankingが既知のラベル問題をどの程度下位へ送れているかを確認します。 |

上位割合を絞ることで特定yaw帯や特定sourceだけが残っていないかも確認します。

## 22. 結果の比較方法

主要比較対象は`Y_base`、`Y_top20`、`Y_top40`、`Y_top60`、`Y_top80`、`Y_top100`の固定epoch 10 checkpointです。

結果を単一の総合scoreへ集約せず、次の関係を個別に確認します。

`adoption ratio -> YawPose subset quality -> rear yaw -> rear SO(3) -> front/side retention`

たとえば`Y_top20`から`Y_top60`までrear yawが改善し、`Y_top80`で悪化した場合は、60〜80%区間で追加された低順位データの利益よりラベルノイズの影響が大きくなった可能性を検討します。

`Y_top100`まで単調に改善した場合は、低順位データを含めても追加rear dataの効果が優勢だったことを示します。

`Y_top20`が`Y_base`より悪化した場合は、yawラベル品質だけでは説明できず、合成画像domain shift、YawPose exposure、loss weightなどを別要因として切り分けます。

## 23. 実験成果物

本実験では少なくとも次の成果物を保存します。

### 信頼度事前計算

- teacher ensemble manifest
- teacher checkpoint hashes
- YawPose input manifest hash
- manual correction hash
- sample-level teacher predictions
- sample-level reliability metrics
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

## 24. 判定対象

本実験では、単一の数値で採用条件を自動決定しません。

各adoption ratioについて、少なくとも次の三つの観点を分離して比較します。

1. rear yaw errorがどの程度変化したかを確認します。
2. rear SO(3) errorがどの程度変化したかを確認します。
3. front / sideおよび外部データセットに退行が発生したかを確認します。

これにより、YawPoseの信頼度順位をどこまで広げると後方yaw改善が得られ、どこからラベルノイズまたはdomain shiftの影響が支配的になるかを評価します。
