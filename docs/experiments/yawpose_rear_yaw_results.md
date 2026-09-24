# YawPose後方Yaw信頼度選択 実験結果

## 目的

本実験の目的は、合成Head Pose EstimationデータセットであるYawPoseの後方画像をyaw-only教師として追加し、SixDRepNet360-ResNet50の後方yaw推定を改善できるかを確認することです。

YawPoseには画像内容とyawラベルの不一致が含まれるため、3種類の固定teacherとYawPose canonical yawの一致度からreliability rankingを学習前に作成しました。学習ではteacher予測を擬似GTとして使用せず、ranking上位20%、40%、60%、80%、100%をそれぞれ独立した条件として比較しました。

本レポートでは、reliability rankingの分布、学習時のYawPose反復量、内部development split、AGORA-HPE・AFLW2000・300W-LP・DAD-3DHeads validationの外部評価をまとめます。

## 対象モデルと学習条件

対象モデルはSixDRepNet360-ResNet50です。初期checkpointは次の配布checkpointです。

| 項目 | 値 |
|---|---|
| path | `checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` |
| SHA-256 | `3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6` |

既存HPE学習データにはVGGHeadsとDAD-3DHeads trainを使用し、YawPoseとは別streamとして扱いました。既存HPE側のrear samplingは自然比率1.743%のままとし、後方サンプルにはSO(3) horizontal-flip consistencyを適用しています。

主要な学習条件は次のとおりです。

| 項目 | 設定 |
|---|---|
| epochs | 10 |
| samples / epoch | 451,072 |
| update scope | ResNet50 `layer4` + regression head |
| precision | BF16 |
| existing HPE micro-batch | 64 |
| YawPose micro-batch | 64 |
| gradient accumulation | 2 |
| optimizer | AdamW |
| backbone learning rate | `1e-6` |
| regression head learning rate | `3e-5` |
| weight decay | `1e-4` |
| retention distillation weight | 1.0 |
| existing rear flip-consistency weight | 0.2 |
| YawPose yaw loss weight | 0.2 |
| seed | 42 |

YawPoseから使用する教師信号はyawだけです。pitchとrollはYawPose教師として使用していません。YawPoseの学習対象はcanonical yawで`|yaw| >= 120°`となる14,049件です。

比較条件は次の6条件です。

| 条件 | YawPose採用範囲 |
|---|---|
| `Y_base` | YawPoseを使用しません。 |
| `Y_top20` | reliability ranking上位20%を使用します。 |
| `Y_top40` | reliability ranking上位40%を使用します。 |
| `Y_top60` | reliability ranking上位60%を使用します。 |
| `Y_top80` | reliability ranking上位80%を使用します。 |
| `Y_top100` | rear candidateをすべて使用します。 |

主比較には固定epoch 10 checkpointを使用しています。

## Reliability ranking

reliability scoreは、各サンプルについて計算した`gt_median_error`、`teacher_dispersion`、`gt_max_error`をそれぞれpercentile rankへ変換し、その平均として定義しています。scoreは確率ではなく、小さいほどranking上位となる相対値です。

teacher ensembleにはSixDRepNet360 base、SemiUHPE EfficientNetV2-S、WHENetを使用しています。YawPose canonical yawはteacher予測へ置換していません。

### 採用subsetの角度分布

reliability rankingとyaw角度帯の間には強い偏りがありました。各subsetの120–150°と150–180°の構成は次のとおりです。

| subset | 件数 | 120–150° | 150–180° | 150–180°比率 |
|---|---:|---:|---:|---:|
| top20 | 2,810 | 2,742 | 68 | 2.4% |
| top40 | 5,620 | 4,776 | 844 | 15.0% |
| top60 | 8,430 | 5,826 | 2,604 | 30.9% |
| top80 | 11,240 | 6,768 | 4,472 | 39.8% |
| top100 | 14,049 | 7,576 | 6,473 | 46.1% |

ranking上位20%では97.6%が120–150°に集中しています。採用率を広げるほど150–180°が増えるため、top20からtop100への比較ではreliability thresholdだけでなく、学習対象yawの分布も同時に変化しています。

各subsetで`human_corrected`として記録されたサンプルは0件です。

### Teacherのyaw較正

SemiUHPEとWHENetのyaw符号は、方向QAを通過したYawPose後方サンプル9,138件で較正されています。

| teacher | 選択された符号でのmean yaw error | 反対符号でのmean yaw error |
|---|---:|---:|
| SemiUHPE EfficientNetV2-S | 122.134° | 152.551° |
| WHENet | 29.677° | 65.465° |

両teacherとも符号選択では同じ向きが選ばれましたが、SemiUHPEは選択後も較正集合に対するmean yaw errorが122.134°でした。したがって、本実験のreliability rankingは、YawPose後方帯でcanonical yawとの一致が弱いteacherを含むensembleから導出されています。

## YawPoseの反復量

`Y_top20`から`Y_top100`までYawPose micro-batch sizeとYawPose loss weightを固定し、各epochで451,072件をYawPose streamからdrawしました。この設計ではsubsetが小さいほど、同じ画像が1 epoch内で多く再利用されます。

| 条件 | 選択画像数 | 平均draw/画像/epoch | repeat ratio |
|---|---:|---:|---:|
| Y_top20 | 2,810 | 160.5 | 99.38% |
| Y_top40 | 5,620 | 80.3 | 98.75% |
| Y_top60 | 8,430 | 53.5 | 98.13% |
| Y_top80 | 11,240 | 40.1 | 97.51% |
| Y_top100 | 14,049 | 32.1 | 96.89% |

`Y_top20`では1画像が平均約160.5回/epoch、10 epochでは単純平均で約1,605回drawされる計算になります。`Y_top100`では約32.1回/epochであり、1画像当たりの平均曝露回数には約5倍の差があります。

`Y_top100`のrepeat ratioは、epoch 10 metricsが空ファイルであるためepoch 9の保存値を使用しています。選択画像数14,049件とtotal draws 451,072件は条件設定上固定されており、epoch 9では14,049件すべてが少なくとも1回drawされています。

このため、採用率比較ではreliability rankingの範囲と角度分布に加えて、1サンプル当たりの累積YawPose loss寄与も同時に変化しています。

## 内部development split

内部development splitはVGGHeadsとDAD-3DHeads trainから分離した55,719件で構成されています。YawPoseはdevelopment splitへ含めていません。

### 初期checkpointとY_base

YawPoseを使用しない`Y_base`だけでも、初期checkpointから後方性能が大きく変化しています。

| 指標 | 初期checkpoint | Y_base epoch 10 | 差 |
|---|---:|---:|---:|
| SO(3) overall mean | 22.640° | 21.163° | -1.477° |
| SO(3) rear mean | 35.462° | 26.593° | -8.869° |
| SO(3) rear P90 | 71.487° | 53.863° | -17.624° |
| yaw overall mean | 19.776° | 18.501° | -1.275° |
| yaw rear mean | 28.819° | 22.811° | -6.008° |
| yaw rear P90 | 67.205° | 50.342° | -16.862° |

したがって、初期checkpointから学習後モデルまでの後方改善の大部分はYawPoseを追加する前の既存HPE fine-tuningで発生しています。YawPoseの追加効果は`Y_base`との差として分離して確認する必要があります。

### YawPose採用率

固定epoch 10で確認できる内部development結果は次のとおりです。

| 条件 | SO(3) overall mean | SO(3) rear mean | yaw overall mean | yaw rear mean | yaw rear P90 |
|---|---:|---:|---:|---:|---:|
| Y_base | 21.163° | 26.593° | 18.501° | 22.811° | 50.342° |
| Y_top20 | 21.006° | 26.900° | 18.373° | 23.212° | 50.236° |
| Y_top40 | 20.981° | 26.746° | 18.346° | 23.014° | 50.124° |
| Y_top60 | 20.983° | 26.575° | 18.343° | 22.795° | 49.875° |
| Y_top80 | 20.993° | 26.294° | 18.350° | 22.485° | 49.848° |

`Y_top20`と`Y_top40`ではrear yaw meanが`Y_base`よりそれぞれ0.401°、0.203°高くなっています。`Y_top60`ではほぼ同値となり、`Y_top80`では0.326°低下しています。rear SO(3) meanも同様に、top20・top40では`Y_base`より高く、top80では0.299°低下しています。

`Y_top100`の外部評価は完了していますが、リポジトリに保存された`metrics/epoch_010.json`が空ファイルであり、内部epoch 10値を同じ根拠から再確認できないため、この表から除外しています。

### YawPose training loss

YawPose training lossと内部rear yawの関係は次のとおりです。

| 条件 | epoch 10 YawPose loss | rear yaw mean |
|---|---:|---:|
| Y_top20 | 0.1099 rad | 23.212° |
| Y_top40 | 0.1805 rad | 23.014° |
| Y_top60 | 0.2245 rad | 22.795° |
| Y_top80 | 0.2985 rad | 22.485° |

subsetを狭くするほどYawPose training lossは小さくなっていますが、内部rear yawは同じ順序では改善していません。training lossの差にはサンプル難易度と反復回数の両方が含まれるため、この値だけからmemorizationの有無を確定することはできません。

## 外部benchmark評価

外部評価では固定epoch 10 checkpointをFP32で評価しています。SO(3) geodesic errorとyaw errorは異なる指標です。AGORA-HPE・AFLW2000・300W-LPのyaw groupはmanifestのsource yawを使用し、内部developmentでは回転行列からhead-forward yawを算出しているため、内部値と外部値は同一角度値として直接比較しません。

### SO(3)全体性能

各外部データセットのSO(3) geodesic meanは次のとおりです。

| 条件 | AGORA-HPE | AFLW2000 | 300W-LP | DAD-3DHeads validation |
|---|---:|---:|---:|---:|
| 初期checkpoint | 45.475° | 6.562° | 5.845° | 31.482° |
| Y_base | 44.402° | 6.534° | 5.803° | 29.385° |
| Y_top20 | 44.140° | 7.011° | 5.931° | 29.086° |
| Y_top40 | 44.128° | 7.030° | 5.669° | 29.054° |
| Y_top60 | 44.092° | 6.923° | 5.618° | 29.071° |
| Y_top80 | 44.159° | 6.823° | 5.637° | 29.112° |
| Y_top100 | 44.275° | 6.754° | 5.650° | 29.135° |

AGORA-HPE、300W-LP、DAD-3DHeads validationでは、複数のYawPose条件が`Y_base`より低いSO(3) meanを示しています。一方、AFLW2000ではYawPoseを追加した5条件すべてが`Y_base`の6.534°を上回っています。

### AGORA-HPEの後方yaw

AGORA-HPEには`|source yaw| >= 120°`の後方サンプルが2,644件あります。rear yaw meanは次のとおりです。

| 条件 | rear yaw mean | rear yaw P90 | Y_baseとの差 |
|---|---:|---:|---:|
| 初期checkpoint | 30.800° | 65.862° | +3.080° |
| Y_base | 27.720° | 53.542° | 0.000° |
| Y_top20 | 27.534° | 53.706° | -0.186° |
| Y_top40 | 27.479° | 53.468° | -0.241° |
| Y_top60 | 27.434° | 53.469° | -0.286° |
| Y_top80 | 27.385° | 53.501° | -0.334° |
| Y_top100 | 27.472° | 53.297° | -0.248° |

初期checkpointから`Y_base`までにrear yaw meanは3.080°低下しています。YawPose追加後の`Y_base`に対する追加差は最大でも`Y_top80`の-0.334°です。

### AGORA-HPEの後方角度帯

AGORA-HPEのrear yawを符号と角度帯で分けると、YawPose追加の方向が120–150°と150–180°で異なります。

| 条件 | negative 120–150° | positive 120–150° | negative 150–180° | positive 150–180° |
|---|---:|---:|---:|---:|
| Y_base | 31.155° | 39.005° | 19.309° | 20.763° |
| Y_top20 | 29.270° | 37.075° | 20.800° | 22.392° |
| Y_top40 | 29.203° | 36.960° | 20.790° | 22.365° |
| Y_top60 | 29.227° | 37.132° | 20.578° | 22.192° |
| Y_top80 | 29.484° | 37.372° | 20.235° | 21.836° |
| Y_top100 | 30.363° | 37.846° | 19.923° | 21.142° |

`Y_top20`では`Y_base`に対して120–150°がnegative側で1.885°、positive側で1.930°低下しています。一方、150–180°はnegative側で1.491°、positive側で1.629°増加しています。

採用率を広げると150–180°の増加幅は縮小します。`Y_top100`では`Y_base`との差がnegative側+0.614°、positive側+0.380°まで縮小しています。

この角度帯別結果は、ranking上位ほど120–150°へ偏っていたsubset構成と対応しています。rear全体ではYawPose追加による小幅な改善が見えますが、その内訳では主に120–150°が改善し、150–180°は`Y_base`より高い誤差を示しています。

### DAD-3DHeads validationの後方yaw

DAD-3DHeads validationのrearは47件です。rear yaw meanは初期checkpointの55.602°に対して、`Y_base`が51.564°、`Y_top20`が49.438°、`Y_top40`が49.623°、`Y_top60`が49.652°、`Y_top80`が49.793°、`Y_top100`が49.519°でした。

YawPose条件は`Y_base`より2°前後低い値ですが、rearの標本数が47件であるため、AGORA-HPEの2,644件とは分けて扱います。

## 考察

### Reliability rankingと角度分布

reliability ranking上位ほど120–150°に集中していました。top20では97.6%が120–150°であり、150–180°は68件しかありませんでした。

この分布では、adoption ratioを20%から100%へ広げる操作が、低順位サンプルを追加する操作であると同時に、deep-rearの比率を2.4%から46.1%へ増やす操作になっています。したがって、今回の結果だけからreliability scoreの閾値効果とyaw分布の効果を分離することはできません。

AGORA-HPEではtop20で120–150°が`Y_base`より約1.9°低下する一方、150–180°は約1.5–1.6°増加しています。top100まで広げるとdeep-rearの増加幅が0.4–0.6°まで縮小しています。この挙動は、ranking上位の浅い後方角への偏りと同じ方向です。

### 少数subsetの反復

YawPose streamのdraw数を条件間で固定したため、top20では2,810枚を451,072回/epoch使用し、1枚当たり平均約160.5回の曝露が発生しています。top100の平均は約32.1回/epochです。

top20はYawPose training lossが最も低い一方、内部rear yaw meanは`Y_base`より0.401°高くなっています。採用範囲を広げるにつれてtraining loss自体は高くなりますが、rear yawはtop60で`Y_base`とほぼ同値となり、top80では0.326°低下しています。

これらの観測は、少数の高順位サンプルへYawPose lossを繰り返し適用した条件で、subsetへの適合と内部rearへの転移が一致していないことを示します。ただし、採用率変更では反復回数とyaw分布が同時に変化しているため、反復だけをdeep-rear悪化の因果要因として分離することはできません。

### 外部データへの影響

YawPose条件はAGORA-HPEとDAD-3DHeads validationで`Y_base`から追加の改善を示す条件がありますが、AFLW2000ではすべてのYawPose条件が`Y_base`より高いSO(3) meanとなっています。300W-LPではtop20だけが`Y_base`より高く、top40以降は低くなっています。

したがって、YawPose yaw-only lossによる変化はデータセット間で一様ではありません。後方yawの変化と非後方・別domainの性能変化を同一の効果として扱うことはできません。

### Reliability scoreの解釈

reliability scoreはラベル正解確率ではなく、3種類のteacherとcanonical yawの相対的一致度から作成した順位です。さらに、SemiUHPEは方向較正用後方集合に対して選択符号でもmean yaw errorが122.134°でした。

今回のrankingが浅い後方角を上位へ集めたこと、ならびにteacherごとのYawPose後方帯に対する一致度が異なることから、reliability scoreをラベル品質だけの尺度として解釈することはできません。

## 制約

本実験の解釈には次の制約があります。

- reliability rankingとyaw角度帯が独立ではなく、ranking上位ほど120–150°へ偏っています。
- adoption ratioを変えるとsubset sizeも変わるため、1サンプル当たりのYawPose反復回数も同時に変化しています。
- SemiUHPEはyaw符号較正後も較正集合に対するmean yaw errorが122.134°であり、teacher ensemble内のyaw一致度にはモデル間差があります。
- YawPose条件では同じYawPose loss weight 0.2と同じ451,072 draw/epochを使用しているため、少数subsetほど1サンプル当たりの累積損失寄与が大きくなります。
- 内部developmentと外部benchmarkではyawの導出元が異なるため、絶対値をデータセット間で直接比較しません。
- DAD-3DHeads validationのrearは47件であり、rear単独値は標本数が少ない結果です。
- `Y_top100`の外部評価は保存されていますが、内部`metrics/epoch_010.json`が空ファイルのため、内部epoch 10値は保存成果物から再確認できません。

## 結論

YawPoseを使用しない`Y_base`では、初期checkpointに対して内部rear yaw meanが28.819°から22.811°へ、AGORA-HPE rear yaw meanが30.800°から27.720°へ低下しました。したがって、今回観測された後方改善の主要部分は既存HPEデータによるfine-tuningで発生しています。

YawPoseを追加した条件では、AGORA-HPE rear yaw meanが`Y_base`から最大0.334°低下し、内部developmentでも`Y_top80`が`Y_base`から0.326°低下しました。一方、AFLW2000のSO(3) meanはすべてのYawPose条件で`Y_base`より増加しています。

reliability ranking上位は120–150°へ強く偏っており、top20では97.6%を占めました。AGORA-HPEでもYawPose追加後は120–150°が改善する一方、150–180°は`Y_base`より悪化しました。採用率を広げてdeep-rearサンプルを増やすと、この悪化幅は縮小しました。

また、YawPoseの総draw数を固定したため、top20では1画像当たり平均約160.5回/epoch、top100では約32.1回/epochの曝露となりました。採用率の変更によってreliability範囲、yaw分布、反復回数が同時に変化しているため、今回の比較だけではreliability thresholdの純粋な効果、yaw分布の効果、反復の効果を個別には推定できません。

## 再現情報

本レポートの主要な根拠となる成果物は次のとおりです。

- `docs/experiments/yawpose_rear_yaw_experiment_plan.md`
- `experiments/runs/yawpose_reliability/metrics/summary.json`
- `experiments/runs/yawpose_rear_search/config.json`
- `experiments/runs/yawpose_rear_search/conditions/Y_base/metrics/`
- `experiments/runs/yawpose_rear_search/conditions/Y_top20/metrics/`
- `experiments/runs/yawpose_rear_search/conditions/Y_top40/metrics/`
- `experiments/runs/yawpose_rear_search/conditions/Y_top60/metrics/`
- `experiments/runs/yawpose_rear_search/conditions/Y_top80/metrics/`
- `experiments/runs/yawpose_rear_search/conditions/Y_top100/metrics/`
- `eval/yawpose_rear_search/conditions/`
- `eval/yawpose_rear_search/comparisons/`
- `eval/yawpose_rear_search/yaw_comparisons/`
- `eval/yawpose_rear_search/yaw_summary_baseline_fp32_final.json`
- `eval/yawpose_rear_search/yaw_summary_baseline_dad_fp32_final_dad.json`
