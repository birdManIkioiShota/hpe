# YawPose後方yaw・15度層別single-draw実験結果

## 概要

本実験では、YawPose後方サンプルをyaw-only教師信号として追加する際に、従来実験で同時に変化していたreliability ranking、yaw角度分布、同一サンプルの反復回数を分離しやすくするため、YawPoseの選別方法とsampling方法を変更しました。

YawPose rear candidateをsigned yawの15度単位で8区分へ分け、各区分の内部で独立してreliability rankingを計算しました。各条件では、8区分それぞれから上位20%、40%、60%、80%、100%を選択します。選択されたYawPoseサンプルは各epochで1回だけdrawし、subsetが小さい場合でも同一画像を反復して総draw数を揃える処理は行いません。

既存HPE学習データについては、VGGHeadsを全条件で使用し、DAD-3DHeads trainを加えない系列と加える系列を分けました。これにより、YawPose採用率とDAD-3DHeads利用有無を独立した条件として比較しました。

固定epoch 10 checkpointを用いた結果、YawPose追加による全体性能とrear平均の変化は従来実験より大幅に小さくなりました。一方、AGORA-HPEの15度yaw帯では、120–150°付近で小さな改善、165–180°付近で小さな退行という角度依存の変化が、VGGHeads-only系列とVGGHeads+DAD-3DHeads系列の両方で観測されました。

## 目的

本実験の目的は、次の3点を確認することです。

1. YawPoseのreliability採用率を変更したときの性能変化を、yaw帯の大きな偏りと同一画像の反復集中を抑えた条件で再評価すること。
2. AGORA-HPEの全周yawを15度単位で評価し、rear平均では見えない角度依存の変化を確認すること。
3. VGGHeadsを固定して使用し、DAD-3DHeads trainの有無による性能差を確認すること。

DAD-3DHeadsは非商用条件を含むライセンスであるため、最終構成では学習データから排除することを前提とします。本実験では、DAD-3DHeadsを外した系列で後方性能が成立するかも確認対象としました。

## 対象モデル

対象モデルはSixDRepNet360-ResNet50です。モデル構造は変更せず、ResNet50の`layer4`とrotation regression headを更新します。

| 項目 | 設定 |
|---|---|
| 初期checkpoint | `checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` |
| SHA-256 | `3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6` |
| model | SixDRepNet360-ResNet50 |
| update scope | ResNet50 `layer4` + regression head |
| epochs | 10 |
| batch size | 64 |
| gradient accumulation | 2 |
| precision | BF16 |
| backbone learning rate | `1e-6` |
| head learning rate | `3e-5` |
| weight decay | `1e-4` |
| warmup updates | 100 |
| gradient clip norm | 1.0 |
| retention distillation weight | 1.0 |
| rear flip-consistency weight | 0.2 |
| YawPose yaw loss weight | 0.2 |
| seed | 42 |

主比較には各条件の固定epoch 10 checkpointを使用します。内部best checkpointは保存しますが、条件ごとに異なるcheckpoint選択時期を主比較へ混ぜないため、外部評価には使用しません。

## YawPoseデータ選別

### canonical yaw

YawPoseの公開yawは`[0°, 360°)`から次式でsigned yawへ変換します。

`yaw_signed = ((yaw_deg + 180) mod 360) - 180`

値域は`[-180°, 180°)`です。

`manual_corrections.jsonl`に対象画像の修正値が存在する場合は、その`corrected_yaw`をcanonical yawとして使用します。修正が存在しない場合は、公開`labels_fixed.jsonl`のyawをcanonical yawとして使用します。

今回のrear candidate 14,049件では、保存されたsubset summary上の`human_corrected`はすべて0件です。

teacher予測はcanonical yawの置換には使用しません。teacher ensembleは、固定されたcanonical yawに対してサンプルの相対的なreliabilityを計算するためだけに使用します。

### rear candidate

canonical yawが次の条件を満たすサンプルだけをYawPose学習候補とします。

`|canonical yaw| >= 120°`

今回のrear candidateは14,049件です。

候補はsigned yawで次の8区分へ分けます。

| 区分 | yaw範囲 |
|---|---|
| negative 180–165 | `[-180°, -165°)` |
| negative 165–150 | `[-165°, -150°)` |
| negative 150–135 | `[-150°, -135°)` |
| negative 135–120 | `[-135°, -120°]` |
| positive 120–135 | `[120°, 135°)` |
| positive 135–150 | `[135°, 150°)` |
| positive 150–165 | `[150°, 165°)` |
| positive 165–180 | `[165°, 180°)` |

signed yawでは`180°`が`-180°`へwrapされるため、180°相当はnegative側の最深部区分へ入ります。

### teacher ensemble

reliability事前計算には次の3 teacherを使用します。

| teacher ID | モデル |
|---|---|
| `sixdrepnet360_base` | 学習開始時のSixDRepNet360-ResNet50 |
| `semiuhpe_effnetv2s` | SemiUHPE EfficientNetV2-S |
| `whenet` | WHENet |

SemiUHPEとWHENetは、固定revision、checkpoint SHA-256、rear candidate manifest SHA-256、yaw符号較正結果をsidecar manifestで検証したpredictionだけを使用します。

teacherごとのyawは共通の`[-180°, 180°)`規約へ変換してから比較します。

### reliability指標

各サンプルについて、3 teacherとcanonical yawの円周角度差を求めます。

teacher `i`の予測yawを`t_i`、canonical yawを`y`とすると、teacherとcanonical yawの誤差は円周距離で計算します。

`e_i = d_circ(t_i, y)`

3 teacherの誤差から次の2指標を作ります。

- `gt_median_error_deg = median(e_i)`
- `gt_max_error_deg = max(e_i)`

実装上の`gt`という名称は、この処理ではcanonical yawを指します。teacherをpseudo-GTとして採用する意味ではありません。

さらに、3 teacher同士の全pairについて円周角度差を求め、そのmedianを次の指標とします。

- `teacher_dispersion_deg = median(d_circ(t_i, t_j))`

`agree_10`、`agree_20`、`agree_30`も保存しますが、reliability scoreの計算には使用しません。

### 15度帯内percentile

従来実験ではrear candidate全体に対してpercentile rankを計算していました。今回の実験では、8個の15度yaw区分ごとに独立してpercentile rankを計算します。

各区分の内部で、次の3指標をそれぞれ0から1のpercentile rankへ変換します。

- `gt_median_error_deg`
- `teacher_dispersion_deg`
- `gt_max_error_deg`

値が小さいほどpercentile rankも小さくなります。同値が存在する場合はaverage rankを使用します。

各サンプルのreliability scoreは次式です。

`reliability_score = (r_median + r_dispersion + r_max) / 3`

scoreが小さいほど、同じ15度yaw帯の中でteacher群とcanonical yawの整合度が相対的に高いサンプルとして扱います。

このscoreはラベルが正しい確率ではありません。同じ15度帯に属する候補集合内での相対順位です。

### subset生成

各15度区分の内部でreliability scoreを昇順に並べ、次の比率を個別に採用します。

| subset | 各15度区分からの採用率 |
|---|---:|
| `top020` | 20% |
| `top040` | 40% |
| `top060` | 60% |
| `top080` | 80% |
| `top100` | 100% |

100%以外では各区分について`ceil(N_bin × ratio)`件を採用します。そのため、全体件数は14,049件へ単純に採用率を掛けた値と完全には一致しません。

生成されたsubset件数は次のとおりです。

| subset | 件数 |
|---|---:|
| top020 | 2,812 |
| top040 | 5,623 |
| top060 | 8,433 |
| top080 | 11,244 |
| top100 | 14,049 |

各15度区分の内部でnestedになるため、同一区分について次の関係を維持します。

`top20 ⊂ top40 ⊂ top60 ⊂ top80 ⊂ top100`

各区分から同じ採用率を取るため、従来実験で発生していたreliability threshold変更によるrear-near / rear-deep構成比の大幅な変化を抑えています。

### source構成

subsetの生成source構成は次のとおりです。

| subset | synthetic_001 | synthetic_004 | synthetic_005 |
|---|---:|---:|---:|
| top20 | 845 | 1,232 | 735 |
| top40 | 1,482 | 2,592 | 1,549 |
| top60 | 1,699 | 4,286 | 2,448 |
| top80 | 1,846 | 6,110 | 3,288 |
| top100 | 1,909 | 7,934 | 4,206 |

15度yaw帯の構成は固定比率化されていますが、source構成は採用率によって変化しています。このため、reliability採用率とsource構成の関係は今回も完全には分離されていません。

## YawPose sampling

選択されたYawPose sampleは各epochで1回だけ使用します。

各epochの開始時にsubset全体をseed付きでshuffleし、同一epoch内では各sampleを重複drawしません。YawPose batchはexisting-HPE epoch全体へ決定論的に分散して配置します。

sampling結果では、すべてのYawPose条件について次の関係が成立しています。

`selected_unique_samples = unique_samples_drawn = total_draws`

`repeated_draws = 0`

`repeat_ratio = 0`

例えばtop20では2,812 draw / epoch、top100では14,049 draw / epochです。従来実験のように、subsetが小さい条件を約451,000 drawまで反復する処理は行いません。

最後のYawPose batchが64件未満の場合は、batch meanをそのまま通常batchと同じ重みで加えず、実サンプル数に比例する形でyaw lossの寄与を補正しています。

## Existing-HPE data

### VGGHeads-only

`vgg_only`系列ではVGGHeads trainだけをexisting-HPE学習データとして使用します。

checkpoint selection用development splitもVGGHeads devです。

### VGGHeads+DAD-3DHeads

`vgg_plus_dad`系列ではVGGHeads trainとDAD-3DHeads trainをexisting-HPE学習poolとして使用します。

checkpoint selection用development splitはVGGHeads devへ固定しており、DAD-3DHeads devは使用しません。

existing-HPE側の1 epoch学習量は両系列でVGGHeads train件数をeffective batch境界へ切り下げた値に固定しています。そのため、DAD-3DHeadsを追加した系列ではtraining poolは増えますが、optimizer update数は増加しません。

したがって、DAD有無の比較は「VGGHeadsにDADを単純追加して学習量も増やした比較」ではなく、「固定training budgetの中でVGGHeads-only poolとVGGHeads+DAD poolを比較したもの」です。

## 条件行列

YawPose採用率6条件とexisting-HPE data 2系列を組み合わせ、合計12条件を比較しました。

| 系列 | 条件 |
|---|---|
| VGGHeads-only | `Y_base_vgg`、`Y_top20_vgg`、`Y_top40_vgg`、`Y_top60_vgg`、`Y_top80_vgg`、`Y_top100_vgg` |
| VGGHeads+DAD | `Y_base_vgg_dad`、`Y_top20_vgg_dad`、`Y_top40_vgg_dad`、`Y_top60_vgg_dad`、`Y_top80_vgg_dad`、`Y_top100_vgg_dad` |

`Y_base`ではYawPose streamを使用しません。その他の学習条件は同系列のYawPose条件と揃えます。

## 内部development評価

checkpoint selectionに使用したVGGHeads devの固定epoch 10結果は次のとおりです。

### VGGHeads-only

| 条件 | rear SO(3) mean | rear P90 | front mean | side mean | rear yaw mean | rear yaw P90 | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Y_base | 25.712° | 51.564° | 19.695° | 20.935° | 22.277° | 49.312° | 8 |
| Y_top20 | 25.711° | 51.654° | 19.686° | 20.934° | 22.276° | 49.309° | 8 |
| Y_top40 | 25.696° | 51.545° | 19.689° | 20.924° | 22.258° | 49.288° | 8 |
| Y_top60 | 25.693° | 51.482° | 19.701° | 20.915° | 22.248° | 49.271° | 8 |
| Y_top80 | 25.688° | 51.520° | 19.694° | 20.911° | 22.241° | 49.256° | 8 |
| Y_top100 | 25.673° | 51.482° | 19.700° | 20.918° | 22.234° | 49.274° | 8 |

YawPose採用率を広げるとrear meanはわずかに低下しますが、`Y_base → Y_top100`でもrear SO(3) meanの差は約-0.039°、rear yaw meanの差は約-0.043°です。

### VGGHeads+DAD-3DHeads

| 条件 | rear SO(3) mean | rear P90 | front mean | side mean | rear yaw mean | rear yaw P90 | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Y_base | 26.483° | 53.624° | 19.886° | 21.016° | 22.914° | 50.700° | 10 |
| Y_top20 | 26.477° | 53.613° | 19.883° | 21.016° | 22.906° | 50.342° | 10 |
| Y_top40 | 26.474° | 53.419° | 19.881° | 21.009° | 22.901° | 49.799° | 10 |
| Y_top60 | 26.482° | 53.541° | 19.888° | 21.001° | 22.906° | 50.364° | 10 |
| Y_top80 | 26.487° | 53.596° | 19.884° | 20.995° | 22.912° | 50.410° | 10 |
| Y_top100 | 26.461° | 53.357° | 19.881° | 21.003° | 22.884° | 49.810° | 10 |

VGGHeads+DAD系列でもYawPoseによるrear meanの変化は小さく、`Y_base → Y_top100`のrear SO(3) mean差は約-0.022°です。

内部best epochはVGGHeads-only系列の全条件が8、VGGHeads+DAD系列の全条件が10でした。主比較ではこの差を使用せず、全条件をepoch 10で評価しています。

## 外部benchmark評価

### データセット全体

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

YawPose追加による同系列の`Y_base`からの変化は、多くの条件で0.01–0.05°程度です。

VGGHeads-onlyではAGORA-HPE overallの最大低下は`Y_top80`の約-0.050°です。VGGHeads+DAD系列では`Y_top80`の約-0.038°です。

AFLW2000ではYawPose追加条件が同系列の`Y_base`より約0.001–0.008°高く、300W-LPでは多くの条件で約0.002–0.009°低い値です。いずれも従来YawPose実験で観測された変化より小さい範囲です。

## AGORA-HPE後方yaw

AGORA-HPEのrear `|yaw| >= 120°`は2,644件です。

### VGGHeads-only

| 条件 | rear yaw mean | Y_baseとの差 |
|---|---:|---:|
| Y_base | 27.333° | — |
| Y_top20 | 27.337° | +0.004° |
| Y_top40 | 27.325° | -0.008° |
| Y_top60 | 27.307° | -0.026° |
| Y_top80 | 27.303° | -0.030° |
| Y_top100 | 27.301° | -0.032° |

### VGGHeads+DAD-3DHeads

| 条件 | rear yaw mean | Y_baseとの差 |
|---|---:|---:|
| Y_base | 27.775° | — |
| Y_top20 | 27.771° | -0.004° |
| Y_top40 | 27.772° | -0.003° |
| Y_top60 | 27.758° | -0.017° |
| Y_top80 | 27.758° | -0.018° |
| Y_top100 | 27.765° | -0.010° |

YawPoseによるrear yaw meanの追加変化は両系列とも数百分の一度です。

初期checkpointのAGORA-HPE rear yaw meanは30.800°です。したがって、初期checkpointから各`Y_base`までのfine-tuningによる変化の方が、YawPose追加による変化より大きくなっています。

## AGORA-HPE 15度yaw帯

全rearを平均するとYawPoseの変化は小さい一方、15度単位では角度依存の傾向があります。

`Y_top80 - Y_base`のSO(3) geodesic mean差をrear 8区分で比較すると次のとおりです。負値は同系列の`Y_base`より誤差が低いことを表します。

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

両系列で、120–150°付近では誤差が低下し、150–165°付近ではほぼ同値、165–180°では誤差が増加する方向が共通しています。

同じ傾向はhead-forward yaw errorでも確認されます。したがって、15度帯ごとの変化はEuler yaw指標だけに依存したものではありません。

VGGHeads-onlyの`Y_top80`では、学習対象外の-120〜-105°で約-0.239°、+105〜+120°で約-0.193°のhead-forward yaw mean低下も観測されます。変化はrear境界の外側へ連続していますが、front中央付近の変化は小さい範囲です。

## DAD-3DHeads利用有無

YawPoseを使用しない`Y_base`同士を比較すると、DAD有無によるデータセット全体のSO(3) mean差は次のとおりです。

| 評価 | VGGHeads-only | VGGHeads+DAD | DADあり − VGG-only |
|---|---:|---:|---:|
| AGORA-HPE overall | 44.498° | 44.415° | -0.083° |
| AFLW2000 overall | 6.541° | 6.535° | -0.006° |
| 300W-LP overall | 5.809° | 5.803° | -0.007° |
| DAD validation overall | 29.497° | 29.460° | -0.037° |
| AGORA rear yaw mean | 27.333° | 27.775° | +0.442° |
| VGGHeads dev rear SO(3) | 25.712° | 26.483° | +0.771° |

DADを加えた系列では、外部データセット全体のSO(3) meanがわずかに低い一方、AGORA-HPE rear yawとVGGHeads dev rear SO(3)はVGGHeads-only系列の方が低い値です。

AGORA-HPEの15度SO(3)では、DAD追加の影響はyaw帯と符号に依存します。`Y_base_vgg_dad - Y_base_vgg`では、negative側のside付近で誤差が低下する一方、positive rearでは誤差が増加しています。

例えば-105〜-90°は約-1.078°、-90〜-75°は約-0.993°ですが、+135〜+150°は約+0.609°、+150〜+165°は約+0.895°、+165〜+180°は約+0.960°です。

したがって、AGORA-HPE overallの約-0.083°という差は全yaw帯の一様な改善ではなく、角度帯ごとに異なる変化を平均した結果です。

## 従来YawPose実験との比較

従来のYawPose reliability実験では、YawPose総draw数をexisting-HPE側に合わせて約451,000 draw / epochへ固定していました。

そのため、top20では約2,810件のsubsetを1 epochあたり平均約160.5回/画像、top100でも約32.1回/画像提示していました。また、従来rankingはrear全体で計算していたため、top20の97.6%が120–150°へ集中していました。

今回の実験では、15度yaw帯ごとに採用率を固定し、各sampleを1 epochに1回だけ使用しています。

AGORA-HPE rear yaw meanの`Y_base → Y_top80`変化を比較すると次のとおりです。

| 実験 | Y_base → Y_top80 |
|---|---:|
| 従来YawPose実験 | -0.334° |
| 今回 VGGHeads-only | -0.030° |
| 今回 VGGHeads+DAD | -0.018° |

従来実験ではAFLW2000に0.3–0.5°規模の悪化も観測されましたが、今回はYawPose追加によるAFLW2000 overall SO(3)の変化は約+0.001–0.008°です。

今回の変更では反復回数とyaw帯ranking方式を同時に変更しているため、effect size縮小を反復回数だけに帰属することはできません。ただし、従来観測された大きな改善と大きな外部domain退行の双方が、YawPose sampleを1回提示するだけでは再現しなかったことは確認できます。

## Reliability採用率

今回の15度層別rankingでは、top20が他の採用率より明確に低い誤差を示す結果にはなっていません。

AGORA-HPE overallではVGGHeads-only、VGGHeads+DADの両系列でtop60–80付近まで誤差がわずかに低下し、top100で一部戻る形があります。一方、その差は0.02–0.05°程度です。

この結果から、reliability上位20%だけを選択すれば外部性能が明確に高くなるという関係は確認されていません。

また、adoption ratioを広げるほどYawPose総sample数も増加します。今回の設計では同一sampleの反復回数を固定しましたが、reliability thresholdと1 epochあたりのYawPose教師信号総量は連動しています。

したがって、top20からtop100までの差をreliability scoreだけの効果として扱うことはできません。

## Teacher ensemble

reliability scoreにはSixDRepNet360 base、SemiUHPE、WHENetの3 teacherを使用しています。

yaw符号較正後のYawPose方向QA集合に対する平均誤差は、WHENetが約29.677°、SemiUHPEが約122.134°です。SemiUHPEは反対符号より選択符号の方が低いため符号判定自体は成立していますが、YawPose後方domainに対する絶対誤差は大きい状態です。

reliability scoreには`gt_max_error`も含むため、domain mismatchの大きいteacherはrankingへ影響します。今回の実験ではteacher構成を変更していないため、その寄与を独立には評価していません。

## DAD-3DHeads validation

DAD-3DHeads official validation全体では、初期checkpointのSO(3) mean 31.482°に対して、今回の全条件が約29.45–29.51°です。

一方、rear `|yaw| >= 120°`は47件しかありません。

rear yaw meanはVGGHeads-only系列で約50.20–50.41°、VGGHeads+DAD系列で約51.25–51.40°です。sample数が少ないため、DAD validationのrear細分化は補助的な確認に限定します。

## 制約

本実験の解釈に関係する制約は次のとおりです。

- 単一seed 42で実行しており、seed間分散は測定していません。
- YawPose adoption ratioと1 epochあたりのYawPose教師サンプル総数が連動しています。
- 15度yaw帯内の採用率は固定しましたが、生成source構成はadoption ratioによって変化します。
- reliability scoreはラベル正解確率ではなく、同一15度yaw帯内の相対順位です。
- teacher ensembleにはYawPose後方domainとの整合度が低いSemiUHPEが含まれます。
- YawPose rear candidateの判定はcanonical yawに依存するため、canonical yaw自体が大きく誤っている場合はcandidate集合にも影響します。
- DAD有無の比較ではexisting-HPE側のtraining budgetを固定しているため、DADあり系列ではVGGHeads sampleの一部がDAD sampleに置き換わります。
- AGORA-HPEでは通常evaluationのEuler yaw errorと、YawPose比較用のhead-forward yaw errorの両方を保存しています。両指標は同一ではありません。
- DAD-3DHeads official validationのrear sampleは47件であり、rear 15度帯の評価には十分な件数ではありません。
- 今回のYawPose追加効果は数百分の一度規模であり、単一seedの差から細かなadoption ratioの優劣を確定することはできません。

## 結論

YawPoseを15度yaw帯ごとにreliability rankingし、選択sampleを1 epochに1回だけ使用すると、YawPose追加による平均性能の変化は従来実験より大幅に小さくなりました。

AGORA-HPE rear yaw meanでは、VGGHeads-only系列の`Y_base → Y_top80`が約-0.030°、VGGHeads+DAD系列が約-0.018°です。従来実験の約-0.334°と比べるとeffect sizeは小さくなっています。AFLW2000で従来観測された大きな退行も今回のsingle-draw条件では再現していません。

一方、15度単位では、両existing-HPE系列で120–150°付近の誤差低下と165–180°付近の誤差増加が共通して観測されました。YawPoseの影響は全後方帯を一様に改善するものではなく、rear境界付近と完全後方付近で方向が異なっています。

reliability採用率については、top20が明確に優位になる結果ではありません。今回の1回draw条件では、高reliabilityな少数subsetだけではモデルへの影響が小さく、採用量を増やしたtop60–80付近で外部誤差がわずかに低下する傾向があります。ただし差は小さく、adoption ratioとYawPose教師信号総量も連動しているため、reliability score単独の効果とは扱いません。

DAD-3DHeadsについては、追加した系列で外部データセット全体のSO(3) meanが数百分の一〜0.08°程度低くなる一方、AGORA-HPE rear yawとVGGHeads dev rear SO(3)ではVGGHeads-only系列の方が低い値です。少なくとも今回の後方yaw評価では、DAD-3DHeadsを使用しないことによって後方性能が成立しなくなる結果は観測されていません。

## 保存成果物

本レポートの根拠となる主要成果物は次の場所に保存されています。

- 実験計画は`docs/experiments/yawpose_rear_yaw_stratified_single_draw_plan.md`です。
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
