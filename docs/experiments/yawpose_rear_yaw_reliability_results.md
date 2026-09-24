# YawPose後方yaw信頼度フィルタリング実験結果

## 目的

本実験は、SixDRepNet360-ResNet50の全周頭部姿勢推定に対して、YawPoseの後方合成画像をyaw-only教師信号として追加したときの効果を評価するものです。

YawPoseにはyawラベルの揺らぎや画像品質のばらつきがあるため、複数teacherの予測とYawPose canonical yawの一致度からreliability scoreを事前計算し、信頼度順位の上位20%、40%、60%、80%、100%を段階的に学習へ投入しました。

本実験では、YawPoseの追加が後方yaw、回転全体のSO(3)誤差、前方・側方性能へ与える影響を確認するとともに、reliability rankingが実際にどのようなサンプルを上位へ集めたかを診断します。

## 対象モデルとデータ

対象モデルはSixDRepNet360-ResNet50です。初期checkpointは次の固定checkpointです。

| 項目 | 値 |
|---|---|
| checkpoint | `checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` |
| SHA-256 | `3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6` |
| update scope | ResNet50 `layer4` + regression head |
| epochs | 10 |
| precision | BF16 |
| optimizer | AdamW |
| backbone LR | `1e-6` |
| head LR | `3e-5` |
| weight decay | `1e-4` |
| seed | 42 |

既存HPE学習データにはVGGHeadsとDAD-3DHeads trainを使用します。既存HPE streamでは自然な後方比率1.743%を維持し、後方サンプルにはflip-consistency loss、後方以外にはbase modelへのretention distillationを適用します。

YawPoseでは`|yaw| >= 120°`の後方候補だけを使用します。YawPose由来の教師信号はyawだけであり、pitchとrollは教師信号として使用しません。

総損失は概念的に次式です。

`L_total = L_original_supervised + 1.0 * L_distillation + 0.2 * L_original_flip + 0.2 * L_yawpose`

YawPoseを使用する条件では、existing HPE micro-batch 64件に対してYawPose micro-batchも64件を毎回投入します。YawPose採用率を変更してもYawPose batch sizeとloss weightは変更しません。

## reliability score

reliability scoreは確率ではなく、YawPose後方候補14,049件の中でteacher一致度を相対順位へ変換したscoreです。値が小さいほど信頼度順位が上位です。

各サンプルについて、3つのteacherとcanonical yawの関係から次の3指標を計算します。

- `gt_median_error`は、各teacherとcanonical yawの円周角度差のmedianです。
- `gt_max_error`は、各teacherとcanonical yawの円周角度差の最大値です。
- `teacher_dispersion`は、teacher同士のpairwise yaw差のmedianです。

3指標をそれぞれpercentile rankへ変換し、等重み平均した値をreliability scoreとします。

teacher ensembleはSixDRepNet360 base、SemiUHPE EfficientNetV2-S、WHENetの3モデルです。SemiUHPEとWHENetのyaw符号は、方向が検証されたYawPose後方サンプル9,138件を使って固定しています。

符号較正時の選択符号における平均yaw誤差はWHENetが29.677°、SemiUHPEが122.134°です。したがって、SemiUHPEについては符号方向の選択自体は可能である一方、今回のYawPose後方集合に対するyaw予測の絶対的な整合度は低い状態です。

## YawPose採用条件

比較条件は次の6条件です。

| 条件 | YawPose採用範囲 |
|---|---|
| `Y_base` | YawPoseを使用しません。 |
| `Y_top20` | reliability ranking上位20%を使用します。 |
| `Y_top40` | reliability ranking上位40%を使用します。 |
| `Y_top60` | reliability ranking上位60%を使用します。 |
| `Y_top80` | reliability ranking上位80%を使用します。 |
| `Y_top100` | 後方候補14,049件をすべて使用します。 |

subsetはnested subsetであり、`Y_top20 ⊂ Y_top40 ⊂ Y_top60 ⊂ Y_top80 ⊂ Y_top100`の関係です。

## reliability subsetの角度分布

reliability上位集合の角度分布は均一ではありません。各subsetの120–150°と150–180°の構成は次のとおりです。

| subset | 件数 | 120–150° | 150–180° | 150–180°比率 |
|---|---:|---:|---:|---:|
| top20 | 2,810 | 2,742 | 68 | 2.4% |
| top40 | 5,620 | 4,776 | 844 | 15.0% |
| top60 | 8,430 | 5,826 | 2,604 | 30.9% |
| top80 | 11,240 | 6,768 | 4,472 | 39.8% |
| top100 | 14,049 | 7,576 | 6,473 | 46.1% |

top20では2,810件中2,742件、97.6%が120–150°に集中しています。採用範囲を広げるにつれて150–180°が増加し、top100では150–180°が46.1%を占めます。

したがって、本実験のreliability rankingはラベル一致度だけを並べ替えたものではなく、**120–150°のサンプルを上位へ強く集中させる角度分布を持っています。**

## YawPoseサンプリング

YawPoseを使用する全条件では、subsetの大きさにかかわらず1 epochあたり451,072件をYawPose streamからdrawします。そのため、subsetが小さい条件ほど同じ画像が高頻度で反復されます。

| 条件 | selected unique | total draws / epoch | 平均draw / image / epoch | repeat ratio |
|---|---:|---:|---:|---:|
| `Y_top20` | 2,810 | 451,072 | 160.5 | 99.38% |
| `Y_top40` | 5,620 | 451,072 | 80.3 | 98.75% |
| `Y_top60` | 8,430 | 451,072 | 53.5 | 98.13% |
| `Y_top80` | 11,240 | 451,072 | 40.1 | 97.51% |
| `Y_top100` | 14,049 | 451,072 | 32.1 | 96.89% |

`Y_top20`では1画像が平均約160.5回/epoch、10 epochでは単純平均で約1,605回提示されます。`Y_top100`では約32.1回/epochであるため、1画像あたりの平均提示回数には約5倍の差があります。

loss coefficientは全YawPose条件で0.2に固定されているため、YawPose stream全体のloss weightは一定です。一方、1サンプルあたりに配分される累積的な学習機会はsubset sizeによって大きく異なります。

## 内部development評価

内部development splitはVGGHeadsとDAD-3DHeadsのdevelopmentデータ55,719件です。YawPoseは内部developmentへ含めません。

学習開始前の初期checkpointと固定epoch 10の内部結果は次のとおりです。

| 条件 | SO(3) overall mean | SO(3) rear mean | SO(3) rear P90 | yaw overall mean | yaw rear mean | yaw rear P90 |
|---|---:|---:|---:|---:|---:|---:|
| 初期checkpoint | 22.640° | 35.462° | 71.487° | 19.776° | 28.819° | 67.205° |
| `Y_base` | 21.163° | 26.593° | 53.863° | 18.501° | 22.811° | 50.342° |
| `Y_top20` | 21.006° | 26.900° | 53.179° | 18.373° | 23.212° | 50.236° |
| `Y_top40` | 20.981° | 26.746° | 53.138° | 18.346° | 23.014° | 50.124° |
| `Y_top60` | 20.983° | 26.575° | 52.614° | 18.343° | 22.795° | 49.875° |
| `Y_top80` | 20.993° | 26.294° | 52.456° | 18.350° | 22.485° | 49.848° |

初期checkpointから`Y_base`への変化だけで、rear SO(3) meanは35.462°から26.593°へ8.869°低下し、rear yaw meanは28.819°から22.811°へ6.008°低下しています。したがって、内部developmentで観測される後方改善の大部分はYawPoseを使用しない既存HPE fine-tuningで発生しています。

YawPose追加分だけを`Y_base`と比較すると、`Y_top20`ではrear yaw meanが0.401°増加し、`Y_top40`では0.203°増加しています。`Y_top60`ではほぼ同値となり、`Y_top80`では22.485°となって`Y_base`より0.326°低下しています。

`Y_top100`の`experiments/runs/yawpose_rear_search/conditions/Y_top100/metrics/epoch_010.json`はリポジトリ上で0 byteのため、固定epoch 10の内部development値はこの成果物から検証できません。外部評価成果物は正常に保存されています。

## 外部SO(3)評価

外部評価では固定epoch 10 checkpointを使用し、FP32、AMP無効、batch size 256、FP64 metric、deterministic algorithms有効の条件で評価しています。

データセット全体のSO(3) geodesic meanは次のとおりです。

| 条件 | AGORA-HPE | AFLW2000 | 300W-LP | DAD-3DHeads validation |
|---|---:|---:|---:|---:|
| 初期checkpoint | 45.475° | 6.562° | 5.845° | 31.482° |
| `Y_base` | 44.402° | 6.534° | 5.803° | 29.385° |
| `Y_top20` | 44.140° | 7.011° | 5.931° | 29.086° |
| `Y_top40` | 44.128° | 7.030° | 5.669° | 29.054° |
| `Y_top60` | 44.092° | 6.923° | 5.618° | 29.071° |
| `Y_top80` | 44.159° | 6.823° | 5.637° | 29.112° |
| `Y_top100` | 44.275° | 6.754° | 5.650° | 29.135° |

AGORA-HPE、300W-LP、DAD-3DHeads validationでは、一部のYawPose条件が`Y_base`より低いSO(3) meanを示しています。一方、AFLW2000ではYawPoseを追加した全条件が`Y_base`より高いSO(3) meanです。

この結果から、YawPose yaw-only教師信号による変化はデータセット間で同方向ではなく、外部domainによって異なることが確認できます。

## AGORA-HPE後方yaw評価

AGORA-HPEには全周yawが含まれるため、外部rear yawの主要評価として使用します。ここでのyaw groupとGT yawはevaluation manifestに保存されたsource yawを使用します。

rear `|yaw| >= 120°`は2,644件です。初期checkpointと各条件のrear yaw meanは次のとおりです。

| 条件 | rear yaw mean | 初期checkpointとの差 | `Y_base`との差 |
|---|---:|---:|---:|
| 初期checkpoint | 30.800° | 0.000° | +3.080° |
| `Y_base` | 27.720° | -3.080° | 0.000° |
| `Y_top20` | 27.534° | -3.266° | -0.186° |
| `Y_top40` | 27.479° | -3.322° | -0.241° |
| `Y_top60` | 27.434° | -3.366° | -0.286° |
| `Y_top80` | 27.385° | -3.415° | -0.334° |
| `Y_top100` | 27.472° | -3.329° | -0.248° |

初期checkpointから`Y_base`までの改善量は3.080°です。YawPose追加による`Y_base`からの変化は0.186–0.334°の範囲であり、平均rear yawに対する追加効果は`Y_base`までの改善より小さい値です。

## AGORA-HPE後方角度帯

AGORA-HPEのrear yawを符号と角度帯で4分割したmeanは次のとおりです。

| 条件 | negative 120–150° | negative 150–180° | positive 120–150° | positive 150–180° |
|---|---:|---:|---:|---:|
| `Y_base` | 31.155° | 19.309° | 39.005° | 20.763° |
| `Y_top20` | 29.270° | 20.800° | 37.075° | 22.392° |
| `Y_top40` | 29.203° | 20.790° | 36.960° | 22.365° |
| `Y_top60` | 29.227° | 20.578° | 37.132° | 22.192° |
| `Y_top80` | 29.484° | 20.235° | 37.372° | 21.836° |
| `Y_top100` | 30.363° | 19.923° | 37.846° | 21.142° |

`Y_top20`を`Y_base`と比較すると、120–150°ではnegative側が1.885°、positive側が1.930°低下しています。一方、150–180°ではnegative側が1.491°、positive側が1.629°増加しています。

採用範囲を広げるにつれて150–180°の悪化量は縮小しています。`Y_top100`では`Y_base`との差がnegative 150–180°で+0.614°、positive 150–180°で+0.380°まで縮小しています。

この角度帯別結果は、reliability上位集合が120–150°へ集中していたsubset分布と同じ方向の変化を示しています。すなわち、上位subsetを強く使用する条件では120–150°が改善する一方、150–180°は`Y_base`より悪化しています。

## DAD-3DHeads validation後方yaw評価

DAD-3DHeads validationのrear `|yaw| >= 120°`は47件です。rear yaw meanは初期checkpointの55.602°に対し、`Y_base`が51.564°、YawPose条件が49.438–49.793°です。

| 条件 | rear yaw mean | 95% bootstrap CIによる初期checkpointとの差 |
|---|---:|---:|
| `Y_base` | 51.564° | [-9.679°, +0.706°] |
| `Y_top20` | 49.438° | [-11.364°, -1.946°] |
| `Y_top40` | 49.623° | [-10.834°, -2.035°] |
| `Y_top60` | 49.652° | [-10.853°, -1.978°] |
| `Y_top80` | 49.793° | [-11.203°, -1.360°] |
| `Y_top100` | 49.519° | [-12.110°, -1.159°] |

rear件数が47件と少なく、150–180°の一部区分は1件または7件しかありません。そのため、DAD-3DHeads validationのrear細分化値は補助結果として扱います。

## 反復集中と学習挙動

固定YawPose draw数により、subsetが小さい条件ほど同じ画像が高頻度で再利用されます。この設計では、reliability閾値を変更するとサンプル品質だけでなく、1画像あたりの平均提示回数も同時に変化します。

epoch 10のYawPose training lossは次のとおりです。

| 条件 | YawPose loss |
|---|---:|
| `Y_top20` | 0.1099 rad |
| `Y_top40` | 0.1805 rad |
| `Y_top60` | 0.2245 rad |
| `Y_top80` | 0.2985 rad |

`Y_top20`はYawPose lossが最も低い一方、内部rear yaw meanは23.212°で、YawPoseを使用しない`Y_base`の22.811°より0.401°高い値です。採用範囲を広げた`Y_top60`では22.795°、`Y_top80`では22.485°となっています。

この結果だけではmemorizationを直接測定できません。top20は高信頼側のサンプルで構成されるため、training lossが低い理由にはサンプル自体の推定容易性も含まれます。

一方、`Y_top20`では2,810枚を1 epochあたり451,072回drawし、角度分布の97.6%が120–150°です。その条件でAGORA-HPEの120–150°が改善し、150–180°が悪化しているため、**少数かつ浅い後方角へYawPose lossが集中したことと整合する学習挙動**が確認できます。

ただし、採用率を変えるとreliability score、yaw角度分布、1画像あたりの反復回数が同時に変化します。そのため、今回の結果だけから反復回数を独立した原因として確定することはできません。

## reliability rankingの解釈

本実験では、reliability scoreが低いサンプルほどteacherとcanonical yawが一致しやすいように設計されています。しかし、実際のrankingでは上位20%の97.6%が120–150°となり、150–180°は2.4%しか含まれませんでした。

したがって、今回のreliability rankingは少なくとも次の二つの性質を同時に含んでいます。

- teacherとcanonical yawの一致度によるラベル信頼度の順位を含みます。
- teacherが一致しやすい120–150°を上位へ集める姿勢難易度の順位も含みます。

このため、`Y_top20`から`Y_top100`までの差を、ラベル品質だけの効果として解釈することはできません。採用率の変更によって、後方内部の角度分布も大きく変化しています。

また、SemiUHPEのYawPose後方集合に対する符号較正後平均誤差が122.134°であるため、teacher ensembleの一部が今回の対象domainへ十分に整合していない可能性があります。この点もreliability scoreの識別力を解釈する際の制約です。

## 外部domainへの影響

YawPose追加による変化は外部データセット間で異なります。

AGORA-HPEのSO(3) meanは`Y_base`の44.402°に対して`Y_top20`から`Y_top100`まで44.092–44.275°です。300W-LPでは`Y_top20`が5.931°で`Y_base`の5.803°より高く、`Y_top40`以降は5.619–5.669°まで低下しています。

AFLW2000では`Y_base`の6.534°に対してYawPose条件が6.754–7.030°であり、すべてのYawPose条件でSO(3) meanが増加しています。

したがって、YawPose yaw-only教師信号は後方yawだけに局所的な影響を与えているわけではなく、モデルの外部domain generalizationにも変化を与えています。

## 実験結果の整理

本実験で確認された主要な観測結果は次のとおりです。

1. 初期checkpointから`Y_base`への既存HPE fine-tuningだけで、内部rear yaw meanは28.819°から22.811°へ低下し、AGORA-HPE rear yaw meanは30.800°から27.720°へ低下しました。
2. YawPose追加によるAGORA-HPE rear yaw meanの追加変化は`Y_base`比で最大0.334°であり、初期checkpointから`Y_base`までの変化より小さい値でした。
3. reliability上位20%は97.6%が120–150°であり、reliability scoreと後方角度帯が強く結び付いていました。
4. `Y_top20`ではAGORA-HPEの120–150°が`Y_base`より改善した一方、150–180°は両符号で悪化しました。
5. subsetを広げて150–180°の比率を増やすと、AGORA-HPE deep-rearの`Y_base`からの悪化量は縮小しました。
6. `Y_top20`では1画像あたり平均160.5 draw/epochとなり、`Y_top100`の約5倍の反復が発生します。固定YawPose loss設計により、採用率は1サンプルあたりの実効的な露出量も同時に変更しています。
7. AFLW2000ではYawPoseを追加した全条件が`Y_base`より高いSO(3) meanとなり、外部domainによるtrade-offが確認されました。

## 制約

本実験の解釈には次の制約があります。

- reliability scoreとyaw角度分布が交絡しているため、ラベル信頼度だけの効果を分離できません。
- YawPose総draw数を固定したため、subset sizeと1画像あたりの反復回数が交絡しています。
- reliability score、角度分布、反復回数の3要因が同時に変化するため、deep-rear悪化の原因を単一要因へ帰属できません。
- SemiUHPEはYawPose後方集合に対する符号較正後平均誤差が122.134°であり、teacher ensembleの一部にdomain mismatchが残っています。
- internal developmentのyawはrotation matrixから求めたhead-forward azimuthを使用し、外部評価のyawは各manifestのsource yawを使用するため、異なるデータセット間の絶対値を同一サンプル規約として直接比較しません。
- DAD-3DHeads validationのrearは47件であり、rear細分化後のサンプル数はさらに少なくなります。
- `Y_top100`の固定epoch 10内部metricファイルは0 byteであり、内部development値を検証できません。

## 結論

今回のreliability scoreは、teacherとcanonical yawの一致度だけでなく、teacherとcanonical yawが一致しやすい120–150°の後方画像を上位へ集める性質を示しました。top20では97.6%が120–150°であり、この角度分布に対応してAGORA-HPEの120–150°は改善する一方、150–180°は`Y_base`より悪化しました。

また、YawPose streamの総draw数を全条件で固定したため、top20の少数画像は1 epochあたり平均160.5回再利用されました。top20のYawPose training lossが低い一方で内部rear yawが`Y_base`より悪化し、外部評価でも浅い後方角の改善とdeep-rearの悪化が同時に現れていることは、少数かつ浅い角度の画像へ教師信号が集中したという学習条件と整合します。

ただし、reliability score、角度分布、反復回数が同時に変化しているため、今回の実験から反復集中だけをdeep-rear悪化の原因として確定することはできません。

## 保存された成果物

本レポートの主要な根拠となる成果物は次の場所に保存されています。

- 実験計画は`docs/experiments/yawpose_rear_yaw_experiment_plan.md`です。
- reliability集計は`experiments/runs/yawpose_reliability/metrics/summary.json`です。
- 条件別内部学習結果は`experiments/runs/yawpose_rear_search/conditions/<condition>/metrics/`です。
- 条件別外部評価は`eval/yawpose_rear_search/conditions/`です。
- baselineとの外部SO(3)比較は`eval/yawpose_rear_search/comparisons/`です。
- full-range yaw比較は`eval/yawpose_rear_search/yaw_comparisons/`です。
- AGORA-HPE、AFLW2000、300W-LPのyaw集計は`eval/yawpose_rear_search/yaw_summary_baseline_fp32_final.json`です。
- DAD-3DHeads validationのyaw集計は`eval/yawpose_rear_search/yaw_summary_baseline_dad_fp32_final_dad.json`です。
