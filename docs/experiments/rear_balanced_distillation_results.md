# Rear-balanced distillation・後方姿勢診断・外部評価レポート

## 目的

SixDRepNet360-ResNet50のモデル構造を変更せず、後方姿勢のSO(3)回転誤差を改善しながら、既存データセットに対する性能を維持できるかを検証した実験です。

本レポートでは、rear-balanced distillationによる学習結果、base modelのhorizontal-flip equivariance、VGGHeads後方サンプルの監査用export、学習後checkpointの外部評価をまとめます。

評価の中心はVGGHeads・DAD-3DHeadsのdev、および実写系benchmarkであるAFLW2000と300W-LPです。AGORA-HPEはCGであり、評価対象の75.3%でhead sizeが64 px未満となるため、後方姿勢を含む補助評価として扱います。

## 対象モデルとcheckpoint

モデル構造はSixDRepNet360-ResNet50で固定されています。

| 種別 | checkpoint | SHA-256 |
|---|---|---|
| Base | `checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` | `3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6` |
| Rear-balanced distillation best | `experiments/runs/dad_vgg_rear_distill/checkpoints/best.pth` | `b9a38260a57bae6d4e2502d25cbe95096f4aa3222f565d7953308fe0473444eb` |

学習runのbest checkpointはepoch 7です。

## 姿勢区分

VGGHeadsとDAD-3DHeadsの学習・dev評価では、rotation matrixからhead-localの`+Z`軸をhead-forward方向として取り出し、XZ平面へ射影した方位角`azimuth`を使用します。

姿勢区分は次の定義です。

| 区分 | 定義 |
|---|---|
| front | `|azimuth| < 60°` |
| side | `60° <= |azimuth| < 120°` |
| rear-near | `120° <= |azimuth| < 150°` |
| rear-deep | `150° <= |azimuth| <= 180°` |

AGORA-HPE、AFLW2000、300W-LPの外部評価では、manifestに保存されたsource yawを使用しています。したがって、devの`azimuth`の正負と外部評価のsource yawの正負は、同じ物理方向として直接対応付けません。

## Rear-balanced distillationの条件

学習データはVGGHeadsとDAD-3DHeads trainを使用し、devは同じprepared dataのdev splitを使用しています。

| split | VGGHeads | DAD-3DHeads | 合計 |
|---|---:|---:|---:|
| train | 417,055 | 34,035 | 451,090 |
| dev | 51,914 | 3,805 | 55,719 |

主要な学習条件は次のとおりです。

| 項目 | 設定 |
|---|---|
| epochs | 10 |
| samples / epoch | 451,072 |
| rear fraction | 25% |
| update scope | `layer4` + regression head |
| precision | BF16 |
| batch size | 64 |
| gradient accumulation | 2 |
| effective batch | 128 |
| backbone LR | `1e-6` |
| head LR | `3e-5` |
| weight decay | `1e-4` |
| rear supervised weight | 1.0 |
| non-rear supervised weight | 1.0 |
| base-model distillation weight | 1.0 |
| seed | 42 |

各サンプルにはGT rotation matrixに対するSO(3) geodesic lossを適用します。後方4区分に入らないサンプルには、GT lossに加えて、凍結したbase modelのrotation predictionへ近づけるdistillation lossを適用します。

学習前のraw train分布は次のとおりです。

| bucket | samples |
|---|---:|
| 後方以外 | 443,229 |
| negative rear-near | 1,593 |
| negative rear-deep | 2,102 |
| positive rear-near | 1,511 |
| positive rear-deep | 2,655 |

学習時には1 epochの25%を後方サンプルとし、その後方サンプルを4 bucketへ均等配分します。1 epochあたりの後方サンプルは112,768件で、各後方bucketから28,192件ずつサンプリングされます。

## Dev評価

ここでいうdevは、VGGHeadsの51,914件とDAD-3DHeadsの3,805件を結合したdevelopment splitです。checkpoint選択にもこのdevを使用しています。

### Dev全体と後方

base modelとbest checkpointのSO(3) geodesic errorは次の結果でした。

| group | metric | Base | Best | 差 |
|---|---|---:|---:|---:|
| overall | mean | 22.640° | 21.696° | -0.944° |
| overall | P90 | 60.321° | 54.155° | -6.166° |
| rear | mean | 35.462° | 22.517° | **-12.945°** |
| rear | median | 25.037° | 13.379° | **-11.657°** |
| rear | P90 | 71.487° | 47.603° | **-23.884°** |
| rear | >90° rate | 7.060% | 3.675% | **-3.385 pt** |
| 後方以外 | mean | 22.397° | 21.680° | -0.717° |
| 後方以外 | P90 | 59.533° | 54.586° | -4.947° |

rear meanは35.462°から22.517°へ低下し、相対的には36.5%の低下です。90°を超える大誤差の割合も7.060%から3.675%へ低下し、相対的には47.9%減少しています。

後方以外のmeanとP90もbaseより低下しており、dev全体では後方改善に伴う平均性能の退行は確認されていません。

### Devの後方角度帯

既存の後方角度区分ごとの結果は次のとおりです。

| azimuth帯 | metric | Base | Best | 差 |
|---|---|---:|---:|---:|
| 120–150° | mean | 34.737° | 22.635° | **-12.101°** |
| 120–150° | median | 25.985° | 13.111° | -12.874° |
| 120–150° | P90 | 67.221° | 50.125° | **-17.096°** |
| 120–150° | >90° rate | 6.170% | 3.085% | -3.085 pt |
| 150–180° | mean | 35.899° | 22.446° | **-13.453°** |
| 150–180° | median | 25.000° | 13.671° | -11.329° |
| 150–180° | P90 | 78.102° | 46.662° | **-31.440°** |
| 150–180° | >90° rate | 7.597% | 4.031% | -3.566 pt |

rear-nearとrear-deepの両方でmean、median、P90、90°超率が改善しています。特にrear-deepのP90は78.102°から46.662°へ31.440°低下しています。

### Devの符号別後方性能

後方全体を`azimuth`の符号で分けると、base modelの時点で非対称性があります。

| group | metric | Base | Best | 差 |
|---|---|---:|---:|---:|
| negative rear | mean | 29.477° | 19.781° | -9.695° |
| negative rear | P90 | 53.279° | 36.094° | -17.185° |
| positive rear | mean | 40.852° | 24.981° | -15.871° |
| positive rear | P90 | 90.432° | 55.249° | -35.183° |

base modelではpositive rearのmeanがnegative rearより11.376°高く、best checkpointでは差が5.200°です。学習によって符号別の性能差は縮小していますが、消失していません。

保存されたdev metricsは角度帯と符号を別々に集計しているため、`negative rear-near`と`positive rear-near`のGT誤差を直接比較する集計は残っていません。このため、devの符号差がrear-nearとrear-deepのどちらに集中しているかは、このrunの保存済みGT metricsだけでは確定できません。

### データセット別dev

VGGHeadsとDAD-3DHeadsを分離した結果でも、meanは両方で低下しています。

| dataset | metric | Base | Best | 差 |
|---|---|---:|---:|---:|
| VGGHeads | mean | 21.427° | 20.510° | -0.917° |
| VGGHeads | P90 | 51.632° | 46.028° | -5.604° |
| DAD-3DHeads | mean | 39.187° | 37.876° | -1.311° |
| DAD-3DHeads | P90 | 127.390° | 124.819° | -2.570° |

このdev評価はDAD-3DHeads trainから作成したdev splitを含みます。DAD-3DHeadsの公式validation評価ではありません。

### Best checkpointの選択

checkpoint選択では、後方以外のmeanがbaseを悪化させないことを先に要求し、その条件を満たしたcheckpointについてrear P90、90°超率、rear meanの順で比較します。

epoch 7のrear P90は47.603°で、epoch 8から10のP90よりわずかに低かったため、epoch 7がbest checkpointになっています。epoch 10ではrear meanが22.411°、90°超率が3.578%まで低下していますが、P90は47.679°でepoch 7を上回りました。

## Horizontal-flip equivariance

base modelに対して、VGGHeads dev 51,914件を使用してhorizontal-flip equivarianceを測定しています。

この実験では、元画像のpredictionと、左右反転画像を推論してrotationを元の座標系へ戻したpredictionのSO(3)距離を測ります。この値はGT誤差ではなく、同一画像を左右反転したときにモデル自身のpredictionがどの程度一致するかを表します。

姿勢帯ごとのmeanは次の結果です。

| pose band | count | mean flip-equivariance error | P90 |
|---|---:|---:|---:|
| front | 48,260 | 15.551° | 40.127° |
| side | 2,631 | 14.841° | 41.278° |
| rear-near | 383 | 26.507° | 69.930° |
| rear-deep | 640 | 31.532° | 76.685° |

front・sideと比較してrear-near・rear-deepで不一致が大きく、base modelは後方姿勢でhorizontal flipに対するequivarianceが低下しています。

符号と後方角度帯を組み合わせると、rear-nearではmeanに大きな符号差がありませんが、rear-deepではpositive側の不一致が大きくなっています。

| group | count | mean | P90 |
|---|---:|---:|---:|
| negative rear-near | 199 | 26.852° | 74.924° |
| positive rear-near | 184 | 26.133° | 65.779° |
| negative rear-deep | 286 | 28.286° | 62.085° |
| positive rear-deep | 354 | **34.155°** | **92.323°** |

rear-deepではpositive側のP90が92.323°、negative側が62.085°で、30.238°の差があります。

この結果は、GTラベルとの比較を使わなくても、base modelの後方predictionに符号依存の非対称性が存在することを示します。一方、この実験だけでは、その非対称性がbase checkpointの学習データ、画像統計、ラベル規約、crop、最適化過程のどこから生じたかは特定できません。

## 後方データの件数と非対称性

今回のfine-tuning用raw dataでは、positive rear-deepは2,655件、negative rear-deepは2,102件です。VGGHeads devでもpositive rear-deepは354件、negative rear-deepは286件です。

さらにrear-balanced trainingでは、4つの後方bucketを1 epochあたり28,192件ずつ均等にサンプリングしています。

したがって、今回使用したfine-tuning用データについては、positive rear-deepが単純に少ないことだけでは、base modelで観測されたpositive側の大きな誤差やflip-equivariance低下を説明できません。ただし、base checkpoint自体のpretraining data分布はこの実験では監査していないため、base modelに存在する非対称性の起源までは確定できません。

## VGGHeads後方監査サンプル

VGGHeads devから、後方4 bucketについて16件ずつ、合計64件のcropを監査用にexportしています。

| bucket | exported |
|---|---:|
| negative rear-near | 16 |
| negative rear-deep | 16 |
| positive rear-near | 16 |
| positive rear-deep | 16 |

manifestは`experiments/runs/vgg_rear_audit_samples/artifacts/rear_samples.jsonl`に保存されています。

このrunが自動的に保証するのは、4 bucketから指定件数のサンプルを抽出して監査用cropとmanifestを生成したことまでです。画像内容とrotation annotationの目視一致を判定するmetricは含まれていません。

## 外部benchmark評価

best checkpointは、base modelの保存済みFP32評価と同じmanifest、image lock、crop、batch size、metric定義で比較されています。

主要条件は次のとおりです。

| 項目 | 設定 |
|---|---|
| inference precision | FP32 |
| AMP | 無効 |
| batch size | 256 |
| metric dtype | FP64 |
| SO(3) geodesic formula | stable atan2 |
| deterministic algorithms | 有効 |
| CUDA matmul / cuDNN conv FP32 precision | IEEE |
| paired bootstrap | 1,000 repetitions |
| bootstrap seed | 0 |

### 実写系benchmarkの全体性能

AFLW2000と300W-LPの全体SO(3) meanは次の結果です。

| dataset | Base | Best | 差 | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| AFLW2000 | 6.5617° | 6.5360° | **-0.0257°** | [-0.0427°, -0.0072°] |
| 300W-LP | 5.8453° | 5.8323° | **-0.0129°** | [-0.0210°, -0.0018°] |

両データセットとも全体meanはbaseよりわずかに低下しています。今回の学習で、AFLW2000または300W-LPのデータセット全体に大きな性能退行は発生していません。

### 実写系benchmarkのyaw帯

AFLW2000ではfrontがほぼ同値で、sideが改善しています。

| AFLW2000 source yaw帯 | Base | Best | 差 | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| front `|yaw| < 60°` | 6.2408° | 6.2408° | -0.00005° | [-0.0167°, +0.0157°] |
| side `60° <= |yaw| < 120°` | 8.3530° | 8.1842° | **-0.1687°** | [-0.2375°, -0.1029°] |

300W-LPではfrontが改善し、sideが悪化しています。

| 300W-LP source yaw帯 | Base | Best | 差 | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| front `|yaw| < 60°` | 6.0418° | 5.9038° | **-0.1379°** | [-0.1457°, -0.1314°] |
| side `60° <= |yaw| < 120°` | 5.5170° | 5.7128° | **+0.1958°** | [+0.1871°, +0.2159°] |

300W-LPのsideには局所的な退行が残っています。AFLW2000と300W-LPの保存済みyaw-band評価には`|yaw| >= 120°`のrear groupが存在しないため、この2データセットから外部rear性能を直接評価することはできません。

## AGORA-HPE補助評価

AGORA-HPEは補助評価として扱います。

### 全体とyaw帯

AGORA-HPE全体のSO(3) meanは45.475°から44.208°へ1.267°低下しました。

yaw帯ごとの結果は次のとおりです。

| source yaw帯 | Base | Best | 差 | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| front `|yaw| < 60°` | 51.091° | 52.581° | **+1.490°** | [+1.062°, +1.904°] |
| side `60° <= |yaw| < 120°` | 47.754° | 47.221° | -0.534° | [-1.080°, +0.044°] |
| rear `|yaw| >= 120°` | 37.900° | 33.233° | **-4.667°** | [-5.267°, -4.116°] |

rearではmeanだけでなくtail metricも改善しています。

| rear metric | Base | Best | 差 |
|---|---:|---:|---:|
| mean | 37.900° | 33.233° | -4.667° |
| median | 30.502° | 28.082° | -2.420° |
| P90 | 71.800° | 58.548° | **-13.253°** |
| >90° rate | 5.787% | 3.177% | **-2.610 pt** |

補助評価では、devで観測された後方改善と同方向の結果が得られています。一方、frontは1.490°悪化しています。

### 後方の符号付きyaw bin

AGORA-HPEのrearをsource yawの30° binで分解すると、学習の影響は左右対称ではありません。

| source yaw bin | Base | Best | 差 |
|---|---:|---:|---:|
| -180° to -150° | 29.415° | 26.155° | **-3.260°** |
| -150° to -120° | 33.738° | 34.792° | **+1.054°** |
| +120° to +150° | 51.119° | 42.303° | **-8.816°** |
| +150° to +180° | 36.262° | 29.139° | **-7.122°** |

base modelではnear-rearに相当する`-150° to -120°`と`+120° to +150°`のmean差が17.381°あります。best checkpointではこの差が7.511°まで縮小しています。

deep-rearに相当する`-180° to -150°`と`+150° to +180°`のmean差も、baseの6.846°からbestの2.984°へ縮小しています。

ただし、`-150° to -120°`では1.054°悪化しており、後方4 binのすべてが改善したわけではありません。

### Head size

AGORA-HPEのhead sizeは`sqrt(bbox_width * bbox_height)`で集計されています。

| head size | count | 比率 | Base mean | Best mean | 差 |
|---|---:|---:|---:|---:|---:|
| <32 px | 2,976 | 39.7% | 63.655° | 60.866° | -2.788° |
| 32–64 px | 2,672 | 35.6% | 39.532° | 38.907° | -0.625° |
| 64–128 px | 1,718 | 22.9% | 25.037° | 25.166° | +0.129° |
| >=128 px | 139 | 1.9% | 23.080° | 24.781° | +1.701° |

7,505件中5,648件、75.3%が64 px未満です。全体改善は主として64 px未満の2 groupで観測され、64 px以上では同等または悪化しています。

### Occlusion

AGORA-HPEのocclusion binでは、4 groupすべてでSO(3) meanが低下しています。

| occlusion | Base mean | Best mean | 差 |
|---|---:|---:|---:|
| 0–25% | 39.726° | 38.809° | -0.917° |
| 25–50% | 40.141° | 38.430° | -1.711° |
| 50–75% | 54.036° | 52.726° | -1.310° |
| 75–90% | 73.492° | 71.465° | -2.027° |

この集計では、AGORA-HPEの改善は低occlusion sampleだけに限定されていません。

## 符号非対称性について今回の実験から言えること

今回の実験結果から、符号非対称性について確認できる事実は4点あります。

1. rear-balanced trainingを行う前のbase modelから、devの後方GT誤差に符号差があります。positive rearのmeanは40.852°、negative rearは29.477°です。
2. base modelのhorizontal-flip equivarianceでも後方に符号差があり、特にrear-deepではpositive側のP90が92.323°、negative側が62.085°です。
3. fine-tuning用raw dataでは悪い側のrear-deep sampleが不足しているわけではなく、学習時には4つの後方bucketを完全に均等サンプリングしています。
4. 学習後にはdevのpositive/negative mean差が11.376°から5.200°へ縮小し、AGORA-HPEでも対応するsource-yaw bin間の差が縮小しています。

したがって、今回のrear-balanced trainingが符号非対称性を新たに作ったとは考えにくく、base modelが後方姿勢に対して既に持っていた非対称なprediction behaviorを学習が部分的に緩和した、という解釈が実験結果と整合します。

一方、非対称性の発生原因そのものは確定していません。特にbase checkpointのpretraining data分布を本実験では監査していないため、pretraining data、画像分布、annotation convention、crop、モデルの学習過程の寄与を分離できません。

## 制約

本実験の解釈には次の制約があります。

- devのGT metricsは、符号と120–150° / 150–180°を交差させた4 groupでは保存されていません。
- VGGHeads後方監査runは64件のcropをexportしていますが、画像とannotationの目視一致を自動評価していません。
- DAD-3DHeads official validationは今回のcandidate checkpointに対して評価されていません。
- AFLW2000と300W-LPの保存済み評価にはrear groupがなく、実写系外部benchmarkによる`|yaw| >= 120°`の直接検証はありません。
- AGORA-HPEは補助評価として使用しており、devの`azimuth`とAGORA-HPEのsource yawは異なる定義です。
- horizontal-flip equivarianceはモデル自己整合性のmetricであり、GT accuracyそのものではありません。
- 符号非対称性とflip-equivariance低下の共存は確認できますが、両者の因果関係はこの実験だけでは確定できません。

## 結論

rear-balanced distillationは、VGGHeads・DAD-3DHeads devにおいてrear meanを35.462°から22.517°へ、rear P90を71.487°から47.603°へ低下させました。rear-nearとrear-deepの両方で改善し、後方以外のdev meanも22.397°から21.680°へ低下しています。

AFLW2000と300W-LPのデータセット全体のSO(3) meanはbaseと同等以上を維持しています。ただし、300W-LPのsideでは0.196°の局所的な退行があります。

補助評価のAGORA-HPEではrear meanが37.900°から33.233°へ低下し、P90も71.800°から58.548°へ低下しました。一方、frontと`-150° to -120°`には退行があります。

符号非対称性は学習前のbase modelから存在します。devのGT誤差ではpositive rearがnegative rearより大きく、horizontal-flip equivarianceでもrear-deepのpositive側で大きな不一致が確認されています。rear-balanced training後はdevの符号別mean差が11.376°から5.200°へ縮小しています。

今回の結果から、後方精度とtail errorの大幅な改善、既存実写系benchmark全体の性能保持、符号非対称性の縮小を同時に確認できます。一方、300W-LP sideの局所退行、後方の符号差、AGORA-HPEの一部yaw binの退行は残っています。

## 保存された成果物

本レポートの主要な根拠となる成果物は次のパスに保存されています。

- `experiments/runs/dad_vgg_rear_distill/`
- `experiments/runs/base_flip_equivariance/`
- `experiments/runs/vgg_rear_audit_samples/`
- `eval/dad_vgg_rear_distill/`
- `eval/comparisons/baseline_vs_dad_vgg_rear_distill/`
- `eval/baseline_fp32/evaluations/baseline/`
