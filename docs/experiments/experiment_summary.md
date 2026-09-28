各実験について、**学習・検証・テストの使用データ、導入した手法、結果**の順に整理します。

ここでは、学習途中の確認やモデル選択に使うデータを「検証」、学習後の外部評価に使うデータを「テスト」と呼びます。DAD-3DHeadsの公式validationは、この整理ではテストに当たります。

対象モデルは全実験で **6DRepNet360（ResNet-50）** です。初期モデルには、300W-LPとPanopticで事前学習された配布モデルを使用します。各条件は原則として同じ初期モデルから独立に学習しています。

結果を読むために必要な指標は次のとおりです。

| 指標 | 意味 |
|---|---|
| SO(3)誤差 | 推定した3次元姿勢を正解姿勢に合わせるために必要な最小回転角です。姿勢全体の誤差を表します。 |
| 全周yaw誤差 | 頭部の左右方向の角度誤差です。角度の一周を考慮し、179°と−179°の差を2°として扱います。 |
| yaw・roll・pitch誤差 | 予測回転と正解回転を同じEuler角へ変換し、各軸の円周上の絶対差をサンプル間で平均した値（MAE）です。 |
| 3軸平均誤差 | yaw・roll・pitchのMAEを等しく平均した値です。 |
| P90 | 90%のサンプルがこの値以下となる誤差です。大きな失敗が抑えられたかを確認します。 |
| 90°超過率 | 誤差が90°を超えるサンプルの割合です。 |

いずれも小さいほど良い値です。姿勢帯は、おおむね正面を絶対角60°未満、側面を60〜120°、後方を120°以上として区分します。

軸別の表では、**yaw・roll・pitch・3軸平均・SO(3)をすべて度（°）**で示します。各軸は $R=R_z(\mathrm{roll})R_y(\mathrm{yaw})R_x(\mathrm{pitch})$ のEuler角へ変換して評価し、1画像の誤差は $e_a=|((\hat a-a+180)\bmod360)-180|$、各軸のMAEはそのサンプル平均です。3軸平均は $(\mathrm{MAE}_{\mathrm{yaw}}+\mathrm{MAE}_{\mathrm{roll}}+\mathrm{MAE}_{\mathrm{pitch}})/3$ です。表は保存値を小数第3位に丸めているため、表示した3軸から計算した平均と末尾がずれる場合があります。[評価処理](../../src/hpe/evaluation/runner.py)、[角度変換と円周誤差](../../src/hpe/geometry/rotations.py)

**Euler yawと全周yaw**は別の指標です。 Euler yawは−90〜90°の範囲に正規化され、後ろ向きの姿勢はpitch・rollとの組み合わせで表現されます。一方、全周yawは頭部の前方軸の向き $\operatorname{atan2}(R_{13},R_{33})$ を元データのyawラベルと比較します。そのため、本文で用いてきた後方の全周yaw誤差と、軸別表のyaw誤差は直接比較できません。横向きの特異点付近ではpitch・rollの表現が大きく変わるため、3軸平均とSO(3)で改善の順位が一致しない場合もあります。

以下の結果表は、保存済みの外部テスト集計を転記したものです。新たな学習・推論は行っていません。学習損失や内部検証のSO(3)から軸別誤差を推定せず、各テストデータセット全体と、主目的であるAGORA-HPE後方について示します。各行の条件名は保存結果へのリンクです。初期モデルはFP32評価を使用し、3軸平均にはEuler角の平均誤差を使っています。保存結果のVMAE（3本の方向ベクトルの角度誤差の平均）とは異なります。

以下の損失式では、$\hat R$を学習中モデルの予測回転、$R$を正解回転、$d(\hat R,R)$を両者のSO(3)距離とします。学習時の角度損失はラジアンで計算し、結果の表では度へ換算しています。したがって、表にある平均誤差と、学習ログの損失値は単位が異なります。

SO(3)距離は、相対回転 $Q=R^\top\hat R$ から回転角を求めます。実装では $c=(\mathrm{tr}(Q)-1)/2$ を $[-1,1]$ に収め、$s=\frac{1}{2}\sqrt{(Q_{32}-Q_{23})^2+(Q_{13}-Q_{31})^2+(Q_{21}-Q_{12})^2}$ として、$d(\hat R,R)=\operatorname{atan2}(s,c)$ を計算します。行列の添字はここでは1始まりです。この形は0°や180°付近でも角度を安定して計算するために使っています。[回転誤差の実装](../../src/hpe/geometry/rotations.py)

1. **通常の追加学習**では、VGGHeadsだけでモデルを学習しました。

   | 用途 | 使用データ |
   |---|---|
   | 学習 | VGGHeads：417,055件 |
   | 検証 | VGGHeads：51,914件 |
   | テスト | AGORA-HPE：7,505件、AFLW2000：1,988件、300W-LP：122,217件 |

   初期モデルを出発点に、VGGHeadsの正解回転へ予測を近づける通常の教師あり学習を20エポック行いました。検証データの平均SO(3)誤差が小さいモデルを選びました。

   損失は、学習バッチ $B$ に含まれる各画像のSO(3)距離を平均した $L_{\mathrm{sup}}=\frac{1}{|B|}\sum_{i\in B}d(\hat R_i,R_i)$ です。上下・左右・傾きの誤差を、正解回転に対する一つの角度として学習します。各サンプルの係数は同じなので、頻出する姿勢ほど多くの更新に関わります。

   最初の1エポックは回転を出力する層だけを更新し、その後は全層を更新します。AdamWを使い、学習率は特徴抽出部分が $10^{-5}$、出力層が $10^{-4}$ です。64件のバッチを2回分蓄積して更新するため、通常の1更新は128件分です。BatchNormの移動平均・分散は固定し、学習時には50%の左右反転と明度・コントラスト・彩度の変化を加えます。左右反転時には正解回転も対応する向きへ変換します。[基本学習の実装](../../src/training/train.py)

   検証誤差は約21.43°から約2.92°へ大きく改善しました。しかし、テストでは改善と悪化が分かれました。

   | テスト指標 | 初期モデル | 追加学習後 |
   |---|---:|---:|
   | AGORA-HPE全体のSO(3)平均 | 45.475° | 39.554° |
   | AGORA-HPE後方のSO(3)平均 | 37.900° | 44.820° |
   | AFLW2000全体のSO(3)平均 | 6.562° | 9.423° |
   | 300W-LP全体のSO(3)平均 | 5.845° | 4.423° |

   全体平均が改善しても、目的とする後方姿勢や別のデータセットでは悪化しました。

   併せてVGGHeadsの姿勢分布を調べると、学習データの約93%が正面で、後方は約1.86%でした。通常の学習と全体平均によるモデル選択では、後方サンプルの影響が小さい構成でした。ただし、この分布だけで悪化の原因を確定したわけではありません。

   軸別誤差とSO(3)を、初期モデルと比較します。

   AGORA-HPE全体7,505件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 27.046 | 41.840 | 42.858 | 37.248 | 45.475 |
   | [VGGHeads追加学習](../../eval/vgg_ft/summary.json) | 23.976 | 39.074 | 40.539 | 34.529 | 39.554 |

   AFLW2000全体1,988件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 3.495 | 4.092 | 5.997 | 4.528 | 6.562 |
   | [VGGHeads追加学習](../../eval/vgg_ft/summary.json) | 5.942 | 6.990 | 8.896 | 7.276 | 9.423 |

   300W-LP全体122,217件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 4.133 | 4.311 | 5.126 | 4.523 | 5.845 |
   | [VGGHeads追加学習](../../eval/vgg_ft/summary.json) | 2.885 | 3.069 | 3.950 | 3.301 | 4.423 |

   AGORA-HPEの後方2,644件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/datasets/agora_hpe/metrics/yaw_bands.json) | 26.610 | 17.807 | 22.026 | 22.148 | 37.900 |
   | [VGGHeads追加学習](../../eval/vgg_ft/datasets/agora_hpe/metrics/yaw_bands.json) | 25.679 | 31.354 | 35.471 | 30.835 | 44.820 |

2. **パラメータ補間**では、初期モデルと実験1のモデルを混ぜました。

   | 用途 | 使用データ・モデル |
   |---|---|
   | 学習 | 新規学習は行わず、初期モデルと、VGGHeadsで学習した実験1のモデルを使用 |
   | 検証 | 新たな検証によるモデル選択は行わず、補間割合を0.25に固定 |
   | テスト | AGORA-HPE、AFLW2000、300W-LP |

   **パラメータ補間**は、同じ構造を持つ二つのモデルのパラメータを、一定の割合で混ぜる方法です。今回は初期モデル75%、追加学習後モデル25%としました。追加学習で得た改善を残しながら、初期モデルの能力を回復できるかを調べています。

   計算式は $W_{\mathrm{mix}}=0.75W_{\mathrm{initial}}+0.25W_{\mathrm{FT}}$ です。二つのモデルで同じ位置にある浮動小数点のパラメータや保存値を補間し、一つのモデルを作ります。推論時にはこのモデルを1回実行します。この操作に対して損失を定義して最適化する処理はありません。整数型の保存値は、両モデルで一致することを確認して保持します。

   テストの平均SO(3)誤差は、AGORA-HPE全体で37.678°、AFLW2000で6.046°、300W-LPで4.561°となり、すべて初期モデルより改善しました。

   一方、AGORA-HPE後方は38.458°で、初期モデルの37.900°を下回りませんでした。後方の中央値は改善しましたが、P90は71.800°から80.911°へ悪化しました。典型的なサンプルの改善と、大きな失敗の抑制が一致しない結果でした。[実験1・2の結果](vgg_pose_distribution_weight_interpolation.md)

   軸別誤差とSO(3)を、初期モデルと比較します。

   AGORA-HPE全体7,505件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 27.046 | 41.840 | 42.858 | 37.248 | 45.475 |
   | [VGGHeads追加学習](../../eval/vgg_ft/summary.json) | 23.976 | 39.074 | 40.539 | 34.529 | 39.554 |
   | [初期75%＋追加学習25%](../../eval/weight_interp_a025/summary.json) | 23.042 | 35.477 | 37.072 | 31.864 | 37.678 |

   AFLW2000全体1,988件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 3.495 | 4.092 | 5.997 | 4.528 | 6.562 |
   | [VGGHeads追加学習](../../eval/vgg_ft/summary.json) | 5.942 | 6.990 | 8.896 | 7.276 | 9.423 |
   | [初期75%＋追加学習25%](../../eval/weight_interp_a025/summary.json) | 3.123 | 3.785 | 5.564 | 4.157 | 6.046 |

   300W-LP全体122,217件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 4.133 | 4.311 | 5.126 | 4.523 | 5.845 |
   | [VGGHeads追加学習](../../eval/vgg_ft/summary.json) | 2.885 | 3.069 | 3.950 | 3.301 | 4.423 |
   | [初期75%＋追加学習25%](../../eval/weight_interp_a025/summary.json) | 3.000 | 3.391 | 4.153 | 3.515 | 4.561 |

   AGORA-HPEの後方2,644件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/datasets/agora_hpe/metrics/yaw_bands.json) | 26.610 | 17.807 | 22.026 | 22.148 | 37.900 |
   | [VGGHeads追加学習](../../eval/vgg_ft/datasets/agora_hpe/metrics/yaw_bands.json) | 25.679 | 31.354 | 35.471 | 30.835 | 44.820 |
   | [初期75%＋追加学習25%](../../eval/weight_interp_a025/datasets/agora_hpe/metrics/yaw_bands.json) | 24.304 | 22.292 | 26.438 | 24.345 | 38.458 |

3. **後方サンプリングと蒸留**では、DAD-3DHeadsを学習データに追加しました。

   | 用途 | 使用データ |
   |---|---|
   | 学習 | VGGHeads：417,055件、DAD-3DHeads train：34,035件 |
   | 検証 | VGGHeads：51,914件、DAD-3DHeads trainから分離した内部検証：3,805件 |
   | テスト | AGORA-HPE、AFLW2000、300W-LP |

   **後方サンプリング**は、学習中に後方画像を提示する頻度を増やす方法です。今回は学習で提示するサンプルの25%を後方にし、左右と角度帯を組み合わせた後方4区分を均等に提示しました。既存画像を繰り返し使うため、元のデータ件数が増えるわけではありません。

   **蒸留**は、別のモデルの予測に近づけるように学習する方法です。今回は初期モデルを固定し、後方以外の画像では、正解姿勢に加えて初期モデルの予測にも近づけました。既存の予測能力を維持することが狙いです。

   学習中モデルとは別に、初期モデルを蒸留用教師として固定します。同じ入力画像に対する教師の予測を $R_i^{(0)}$、バッチ中の非後方画像の集合を $B_{\mathrm{nonrear}}$ とすると、蒸留損失は $L_{\mathrm{distill}}=\frac{1}{|B_{\mathrm{nonrear}}|}\sum_{i\in B_{\mathrm{nonrear}}}d(\hat R_i,R_i^{(0)})$ です。教師側は更新せず、学習中モデル側だけに勾配を流します。

   総損失は $L=L_{\mathrm{sup}}+1.0L_{\mathrm{distill}}$ です。教師あり損失は後方を含むバッチ全体で平均し、蒸留損失は非後方だけで平均してから加えます。正解へ近づける目的と初期モデルの予測を保持する目的が食い違う場合は、両方を考慮した更新になります。バッチに蒸留対象がない場合、その項は0です。

   後方の提示頻度は増やしますが、教師あり損失の1サンプル当たりの係数は後方・非後方とも1.0です。実験3以降の学習では、更新対象をResNet-50の最後のブロックと出力層に絞り、それぞれの学習率を $10^{-6}$ と $3\times10^{-5}$ に下げています。64件×勾配蓄積2回、AdamW、weight decay $10^{-4}$、勾配ノルム上限1.0を使い、BatchNormの移動平均・分散は固定します。[損失と更新範囲の実装](../../src/experiments/common/training.py)

   検証データの後方SO(3)平均は35.462°から22.517°へ、P90は71.487°から47.603°へ改善しました。テストでも、AGORA-HPE後方が37.900°から33.233°へ改善しました。AFLW2000全体は6.536°、300W-LP全体は5.832°で、初期モデルの全体性能をほぼ維持しています。

   ただし、AGORA-HPE正面や300W-LP側面など、一部の姿勢帯では悪化しました。また、この実験ではDADの追加、後方の提示頻度、蒸留などを同時に導入しているため、改善をDADだけの効果とは言えません。[実験3の結果](rear_balanced_distillation_results.md)

   軸別誤差とSO(3)を、初期モデルと比較します。

   AGORA-HPE全体7,505件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 27.046 | 41.840 | 42.858 | 37.248 | 45.475 |
   | [後方25%＋蒸留](../../eval/dad_vgg_rear_distill/summary.json) | 25.688 | 42.155 | 43.530 | 37.124 | 44.208 |

   AFLW2000全体1,988件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 3.495 | 4.092 | 5.997 | 4.528 | 6.562 |
   | [後方25%＋蒸留](../../eval/dad_vgg_rear_distill/summary.json) | 3.470 | 3.929 | 5.807 | 4.402 | 6.536 |

   300W-LP全体122,217件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 4.133 | 4.311 | 5.126 | 4.523 | 5.845 |
   | [後方25%＋蒸留](../../eval/dad_vgg_rear_distill/summary.json) | 4.156 | 4.114 | 4.911 | 4.394 | 5.832 |

   AGORA-HPEの後方2,644件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/datasets/agora_hpe/metrics/yaw_bands.json) | 26.610 | 17.807 | 22.026 | 22.148 | 37.900 |
   | [後方25%＋蒸留](../../eval/dad_vgg_rear_distill/datasets/agora_hpe/metrics/yaw_bands.json) | 23.626 | 13.401 | 18.539 | 18.522 | 33.233 |

4. **左右反転整合性学習**では、反転した画像への予測を一致させる損失を追加しました。

   | 用途 | 使用データ |
   |---|---|
   | 学習 | VGGHeads：417,055件、DAD-3DHeads train：34,035件 |
   | 検証 | VGGHeads：51,914件、DAD-3DHeads内部検証：3,805件 |
   | テスト | AGORA-HPE、AFLW2000、300W-LP、DAD-3DHeads公式validation：4,312件 |
   | 反転整合性の診断 | VGGHeads検証データ：51,914件 |

   **左右反転整合性学習**では、元画像と左右反転画像をモデルに入力します。反転画像の予測を元の座標系へ戻したときに、元画像の予測と一致するよう学習します。画像を反転しただけで予測が不安定になることを抑える方法です。

   後方25%のサンプリングと蒸留に、この損失を係数1.0で追加しました。検証によるモデル選択も、正面・側面の性能を確認しながら、後方4区分の悪い側を重視する基準に変更しています。

   元画像の予測を $\hat R(x)$、左右反転した画像の予測を $\hat R(Fx)$ とします。反転を元に戻す行列を $S=\operatorname{diag}(-1,1,1)$ とすると、後方画像集合 $B_{\mathrm{rear}}$ に対する損失は $L_{\mathrm{flip}}=\frac{1}{|B_{\mathrm{rear}}|}\sum_{x\in B_{\mathrm{rear}}}d(\hat R(x),S\hat R(Fx)S)$ です。元画像と反転画像の両方を学習中モデルで推論し、両方の予測を通して勾配を流します。

   総損失は $L=L_{\mathrm{sup}}+1.0L_{\mathrm{distill}}+1.0L_{\mathrm{flip}}$ です。反転損失は後方画像だけで平均します。通常の左右反転によるデータ拡張が正解ラベルに対する学習機会を増やすのに対し、この項は二つの入力に対する予測同士の対応を直接学習させます。バッチに後方画像がない場合は0です。

   検証の後方SO(3)平均は35.462°から19.644°へ、AGORA-HPE後方は37.900°から32.961°へ改善しました。反転前後の予測の不一致も、後方平均で29.651°から6.230°へ減りました。

   一方、AGORA-HPE全体は46.856°、AFLW2000は6.752°、300W-LPは5.948°となり、初期モデルより悪化しました。DAD公式validation全体は31.482°から28.837°へ改善しています。後方改善と、他のデータ・姿勢の性能維持が両立しない部分が残りました。[実験4の結果](rear_flip_consistency_results.md)

   軸別誤差とSO(3)を、初期モデルと比較します。

   AGORA-HPE全体7,505件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 27.046 | 41.840 | 42.858 | 37.248 | 45.475 |
   | [後方25%＋蒸留＋反転係数1.0](../../eval/dad_vgg_rear_flip_consistency_bench/summary.json) | 26.773 | 43.806 | 45.582 | 38.720 | 46.856 |

   AFLW2000全体1,988件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 3.495 | 4.092 | 5.997 | 4.528 | 6.562 |
   | [後方25%＋蒸留＋反転係数1.0](../../eval/dad_vgg_rear_flip_consistency_bench/summary.json) | 3.509 | 4.590 | 6.489 | 4.863 | 6.752 |

   300W-LP全体122,217件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 4.133 | 4.311 | 5.126 | 4.523 | 5.845 |
   | [後方25%＋蒸留＋反転係数1.0](../../eval/dad_vgg_rear_flip_consistency_bench/summary.json) | 4.279 | 3.946 | 4.738 | 4.321 | 5.948 |

   DAD-3DHeads公式validation全体4,312件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_dad_fp32/evaluations/baseline/summary.json) | 16.016 | 25.247 | 26.054 | 22.439 | 31.482 |
   | [後方25%＋蒸留＋反転係数1.0](../../eval/dad_vgg_rear_flip_consistency_dad/summary.json) | 13.881 | 23.541 | 24.822 | 20.748 | 28.837 |

   AGORA-HPEの後方2,644件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/datasets/agora_hpe/metrics/yaw_bands.json) | 26.610 | 17.807 | 22.026 | 22.148 | 37.900 |
   | [後方25%＋蒸留＋反転係数1.0](../../eval/dad_vgg_rear_flip_consistency_bench/datasets/agora_hpe/metrics/yaw_bands.json) | 23.528 | 11.919 | 17.824 | 17.757 | 32.961 |

5. **後方の提示割合と反転損失係数の比較**では、二つの要因を9条件で調べました。

   | 用途 | 使用データ |
   |---|---|
   | 学習 | 全9条件でVGGHeadsとDAD-3DHeads train |
   | 検証 | VGGHeadsとDAD-3DHeads内部検証 |
   | テスト | AGORA-HPE、AFLW2000、300W-LP、DAD-3DHeads公式validation |

   後方の提示割合を約1.74%・5%・25%、反転損失係数を0・0.2・1.0とし、全組み合わせを比較しました。損失係数は、その学習目的を全体の損失へどの程度強く反映するかを決める値です。

   この実験では、提示割合と反転損失の強さを別々に変えることで、それぞれの効果を比較しています。各条件を10エポック学習し、テストにはすべて最終エポックのモデルを使いました。

   損失式は $L=L_{\mathrm{sup}}+1.0L_{\mathrm{distill}}+\lambda_{\mathrm{flip}}L_{\mathrm{flip}}$ で、$\lambda_{\mathrm{flip}}$ を0・0.2・1.0と変えています。後方の提示割合は別に設定するため、後方の正解から学ぶ機会と、予測の反転整合性を求める強さを分けて比較できます。

   実験3・4では後方4区分を均等に提示していましたが、この9条件比較では元データの各区分の件数比率を維持します。1エポックの提示数は451,072件に揃えています。また、反転損失は後方だけで平均するため、係数0.2が「全サンプルの損失の20%を反転損失にする」という意味にはなりません。各項の値、対象画像の数、対象が現れる頻度によって、実際の勾配への寄与は変わります。

   自然比率・反転係数0では、確認した主要な姿勢別平均を維持・改善しました。自然比率・係数0.2では後方改善が拡大した一方、300W-LP側面が0.126°悪化しました。後方25%・係数0.2は9条件中でAGORA-HPE後方のSO(3)平均が最小でしたが、AGORA-HPE正面などが悪化しました。全9条件の値を以下の結果表に示します。

   後方画像を多く提示すると後方精度は上がりますが、正面・側面の悪化も増えました。反転損失も、強くすれば一様に良くなるわけではありませんでした。[実験5の結果](rear_sampling_flip_consistency_tradeoff.md)

   以下では全9条件を、同じテストデータセットごとに比較します。

   AGORA-HPE全体7,505件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 27.046 | 41.840 | 42.858 | 37.248 | 45.475 |
   | [自然比率約1.74%・反転係数0](../../eval/rear_flip_search/conditions/rear_natural_flip_000/summary.json) | 26.886 | 41.625 | 42.727 | 37.079 | 45.146 |
   | [自然比率約1.74%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_natural_flip_020/summary.json) | 26.200 | 41.152 | 42.613 | 36.655 | 44.402 |
   | [自然比率約1.74%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_natural_flip_100/summary.json) | 27.588 | 41.615 | 43.426 | 37.543 | 46.259 |
   | [後方5%・反転係数0](../../eval/rear_flip_search/conditions/rear_005_flip_000/summary.json) | 26.668 | 41.534 | 42.731 | 36.978 | 44.886 |
   | [後方5%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_005_flip_020/summary.json) | 26.117 | 41.616 | 43.177 | 36.970 | 44.708 |
   | [後方5%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_005_flip_100/summary.json) | 29.010 | 43.146 | 45.103 | 39.086 | 48.668 |
   | [後方25%・反転係数0](../../eval/rear_flip_search/conditions/rear_025_flip_000/summary.json) | 25.824 | 42.176 | 43.561 | 37.187 | 44.456 |
   | [後方25%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_025_flip_020/summary.json) | 25.649 | 42.702 | 44.321 | 37.557 | 44.886 |
   | [後方25%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_025_flip_100/summary.json) | 27.217 | 43.957 | 45.701 | 38.958 | 47.386 |

   AFLW2000全体1,988件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 3.495 | 4.092 | 5.997 | 4.528 | 6.562 |
   | [自然比率約1.74%・反転係数0](../../eval/rear_flip_search/conditions/rear_natural_flip_000/summary.json) | 3.488 | 4.057 | 5.966 | 4.504 | 6.542 |
   | [自然比率約1.74%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_natural_flip_020/summary.json) | 3.475 | 3.959 | 5.848 | 4.427 | 6.534 |
   | [自然比率約1.74%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_natural_flip_100/summary.json) | 3.408 | 4.232 | 6.185 | 4.609 | 6.570 |
   | [後方5%・反転係数0](../../eval/rear_flip_search/conditions/rear_005_flip_000/summary.json) | 3.482 | 3.997 | 5.896 | 4.458 | 6.535 |
   | [後方5%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_005_flip_020/summary.json) | 3.471 | 4.095 | 5.979 | 4.515 | 6.552 |
   | [後方5%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_005_flip_100/summary.json) | 3.368 | 4.375 | 6.394 | 4.713 | 6.656 |
   | [後方25%・反転係数0](../../eval/rear_flip_search/conditions/rear_025_flip_000/summary.json) | 3.464 | 3.964 | 5.843 | 4.424 | 6.537 |
   | [後方25%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_025_flip_020/summary.json) | 3.462 | 4.145 | 6.018 | 4.542 | 6.592 |
   | [後方25%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_025_flip_100/summary.json) | 3.514 | 4.518 | 6.424 | 4.819 | 6.755 |

   300W-LP全体122,217件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 4.133 | 4.311 | 5.126 | 4.523 | 5.845 |
   | [自然比率約1.74%・反転係数0](../../eval/rear_flip_search/conditions/rear_natural_flip_000/summary.json) | 4.089 | 4.244 | 5.055 | 4.463 | 5.781 |
   | [自然比率約1.74%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_natural_flip_020/summary.json) | 4.135 | 4.080 | 4.888 | 4.368 | 5.803 |
   | [自然比率約1.74%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_natural_flip_100/summary.json) | 4.201 | 3.793 | 4.645 | 4.213 | 5.856 |
   | [後方5%・反転係数0](../../eval/rear_flip_search/conditions/rear_005_flip_000/summary.json) | 4.102 | 4.209 | 5.014 | 4.442 | 5.789 |
   | [後方5%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_005_flip_020/summary.json) | 4.149 | 3.997 | 4.808 | 4.318 | 5.807 |
   | [後方5%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_005_flip_100/summary.json) | 4.200 | 3.718 | 4.585 | 4.167 | 5.852 |
   | [後方25%・反転係数0](../../eval/rear_flip_search/conditions/rear_025_flip_000/summary.json) | 4.163 | 4.121 | 4.912 | 4.398 | 5.836 |
   | [後方25%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_025_flip_020/summary.json) | 4.215 | 4.067 | 4.863 | 4.382 | 5.877 |
   | [後方25%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_025_flip_100/summary.json) | 4.239 | 3.933 | 4.725 | 4.299 | 5.907 |

   DAD-3DHeads公式validation全体4,312件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_dad_fp32/evaluations/baseline/summary.json) | 16.016 | 25.247 | 26.054 | 22.439 | 31.482 |
   | [自然比率約1.74%・反転係数0](../../eval/rear_flip_search/conditions/rear_natural_flip_000_dad/summary.json) | 15.706 | 24.704 | 25.484 | 21.965 | 30.722 |
   | [自然比率約1.74%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_natural_flip_020_dad/summary.json) | 15.083 | 23.530 | 24.440 | 21.018 | 29.385 |
   | [自然比率約1.74%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_natural_flip_100_dad/summary.json) | 13.704 | 20.787 | 21.738 | 18.743 | 26.546 |
   | [後方5%・反転係数0](../../eval/rear_flip_search/conditions/rear_005_flip_000_dad/summary.json) | 15.662 | 24.585 | 25.451 | 21.899 | 30.607 |
   | [後方5%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_005_flip_020_dad/summary.json) | 14.718 | 22.933 | 23.917 | 20.523 | 28.744 |
   | [後方5%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_005_flip_100_dad/summary.json) | 12.650 | 20.132 | 21.247 | 18.009 | 25.608 |
   | [後方25%・反転係数0](../../eval/rear_flip_search/conditions/rear_025_flip_000_dad/summary.json) | 15.438 | 24.749 | 25.868 | 22.018 | 30.566 |
   | [後方25%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_025_flip_020_dad/summary.json) | 15.012 | 24.392 | 25.626 | 21.677 | 30.081 |
   | [後方25%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_025_flip_100_dad/summary.json) | 13.915 | 23.779 | 25.039 | 20.911 | 29.155 |

   AGORA-HPEの後方2,644件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/datasets/agora_hpe/metrics/yaw_bands.json) | 26.610 | 17.807 | 22.026 | 22.148 | 37.900 |
   | [自然比率約1.74%・反転係数0](../../eval/rear_flip_search/conditions/rear_natural_flip_000/datasets/agora_hpe/metrics/yaw_bands.json) | 26.312 | 17.251 | 21.627 | 21.730 | 37.353 |
   | [自然比率約1.74%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_natural_flip_020/datasets/agora_hpe/metrics/yaw_bands.json) | 24.562 | 14.719 | 20.008 | 19.763 | 34.819 |
   | [自然比率約1.74%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_natural_flip_100/datasets/agora_hpe/metrics/yaw_bands.json) | 25.485 | 13.914 | 19.774 | 19.724 | 35.830 |
   | [後方5%・反転係数0](../../eval/rear_flip_search/conditions/rear_005_flip_000/datasets/agora_hpe/metrics/yaw_bands.json) | 25.649 | 16.150 | 20.801 | 20.867 | 36.235 |
   | [後方5%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_005_flip_020/datasets/agora_hpe/metrics/yaw_bands.json) | 23.921 | 13.423 | 18.887 | 18.743 | 33.811 |
   | [後方5%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_005_flip_100/datasets/agora_hpe/metrics/yaw_bands.json) | 26.722 | 13.137 | 19.217 | 19.692 | 36.593 |
   | [後方25%・反転係数0](../../eval/rear_flip_search/conditions/rear_025_flip_000/datasets/agora_hpe/metrics/yaw_bands.json) | 23.634 | 13.271 | 18.380 | 18.428 | 33.181 |
   | [後方25%・反転係数0.2](../../eval/rear_flip_search/conditions/rear_025_flip_020/datasets/agora_hpe/metrics/yaw_bands.json) | 22.919 | 12.387 | 17.960 | 17.755 | 32.317 |
   | [後方25%・反転係数1.0](../../eval/rear_flip_search/conditions/rear_025_flip_100/datasets/agora_hpe/metrics/yaw_bands.json) | 24.031 | 11.860 | 17.717 | 17.869 | 33.339 |

6. **YawPoseの信頼度選別**では、合成後方画像を選んで追加しました。

   | 用途 | 使用データ |
   |---|---|
   | 学習 | 全条件でVGGHeadsとDAD-3DHeads train。追加条件ではYawPose後方候補14,049件の上位20%・40%・60%・80%・100% |
   | 検証 | VGGHeadsとDAD-3DHeads内部検証 |
   | テスト | AGORA-HPE、AFLW2000、300W-LP、DAD-3DHeads公式validation |
   | 学習前の選別 | YawPose後方候補14,049件 |

   YawPoseは合成画像のデータセットです。この実験ではyawだけを教師に使います。

   **信頼度選別**では、初期6DRepNet360、SemiUHPE、WHENetの3モデルの予測を事前に計算しました。YawPoseのyawラベルとの一致度と、モデル同士の一致度から順位を作り、上位から一定割合を採用します。モデルの予測でラベルを書き換える方法ではありません。

   信頼度の順位には、3教師とyawラベルとの円周誤差の中央値、同誤差の最大値、教師同士の3組の円周誤差の中央値を使います。それぞれを候補全体の中で0〜1の順位へ変換し、3順位の平均をスコアにします。小さいほど一致度が高い候補です。採用後は、このスコアによって画像ごとの損失を重み付けせず、選ばれた画像を同じ係数で学習します。

   YawPose画像の予測回転から、頭部の前方軸をXZ平面へ射影した角度 $\hat y=\operatorname{atan2}(\hat R_{13},\hat R_{33})$ を取り出します。正解yawを $y$ とすると、1画像の損失は $\ell_{\mathrm{yaw}}=|\operatorname{atan2}(\sin(\hat y-y),\cos(\hat y-y))|$ です。これは一周を考慮した絶対角度誤差で、179°と−179°が近い向きであることを学習に反映します。YawPoseからpitch・rollの正解を作る処理はありません。ただし、同じネットワークのパラメータを更新するため、他の姿勢成分の予測が変わる可能性はあります。

   この実験では、既存データ64件のバッチごとにYawPoseも64件投入し、その平均損失を $L_{\mathrm{yaw}}$ としました。総損失は $L=L_{\mathrm{sup}}+1.0L_{\mathrm{distill}}+0.2L_{\mathrm{flip}}+0.2L_{\mathrm{yaw}}$ です。前3項はVGGHeads・DAD画像に、最後の項はYawPose画像に適用します。蒸留教師は初期6DRepNet360であり、選別に用いた3教師を学習中に実行する構成ではありません。

   既存データとYawPoseをそれぞれ451,072件／エポック提示し、選別したYawPoseを繰り返して所定の提示数に合わせました。既存データ側の後方比率は自然比率の約1.74%です。YawPoseには明度・コントラスト・彩度の変化を加えますが、左右反転は加えません。[実験6当時の学習実装](https://github.com/birdManIkioiShota/hpe/blob/8d9b00cb137f6e03cd8db304c3a02718b5611641/src/experiments/scripts/train_yawpose_rear.py)

   YawPoseを追加しない対照条件を含め、6条件を比較しました。AGORA-HPE後方の全周yaw誤差は、初期モデルの30.800°から、対照条件だけで27.720°へ改善しました。YawPoseを追加した効果は、そこから最大約0.334°の改善でした。一方、AFLW2000の平均SO(3)誤差は、YawPose追加条件がすべて対照条件より悪化しました。

   選別結果には偏りがあり、上位20%の97.6%が120〜150°に集中していました。また、YawPoseの総提示数を固定したため、上位20%では同じ画像を1エポックに平均約160.5回使用していました。信頼度、角度分布、反復回数が同時に変わる比較になっていました。[実験6の結果](yawpose_rear_yaw_reliability_results.md)

   以下では全6条件を、同じテストデータセットごとに比較します。初期モデルは学習前の比較基準、対照条件はYawPoseを追加せずに追加学習したモデルです。

   AGORA-HPE全体7,505件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 27.046 | 41.840 | 42.858 | 37.248 | 45.475 |
   | [対照条件（YawPose追加なし）](../../eval/yawpose_rear_search/conditions/Y_base/summary.json) | 26.200 | 41.152 | 42.613 | 36.655 | 44.402 |
   | [YawPose上位20%](../../eval/yawpose_rear_search/conditions/Y_top20/summary.json) | 25.854 | 41.559 | 43.028 | 36.814 | 44.140 |
   | [YawPose上位40%](../../eval/yawpose_rear_search/conditions/Y_top40/summary.json) | 25.842 | 41.566 | 43.027 | 36.811 | 44.128 |
   | [YawPose上位60%](../../eval/yawpose_rear_search/conditions/Y_top60/summary.json) | 25.825 | 41.517 | 42.976 | 36.773 | 44.092 |
   | [YawPose上位80%](../../eval/yawpose_rear_search/conditions/Y_top80/summary.json) | 25.879 | 41.504 | 42.966 | 36.783 | 44.159 |
   | [YawPose上位100%](../../eval/yawpose_rear_search/conditions/Y_top100/summary.json) | 25.978 | 41.473 | 42.910 | 36.787 | 44.275 |

   AFLW2000全体1,988件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 3.495 | 4.092 | 5.997 | 4.528 | 6.562 |
   | [対照条件（YawPose追加なし）](../../eval/yawpose_rear_search/conditions/Y_base/summary.json) | 3.475 | 3.959 | 5.848 | 4.427 | 6.534 |
   | [YawPose上位20%](../../eval/yawpose_rear_search/conditions/Y_top20/summary.json) | 3.683 | 7.133 | 8.962 | 6.593 | 7.011 |
   | [YawPose上位40%](../../eval/yawpose_rear_search/conditions/Y_top40/summary.json) | 3.821 | 6.421 | 8.253 | 6.165 | 7.030 |
   | [YawPose上位60%](../../eval/yawpose_rear_search/conditions/Y_top60/summary.json) | 3.770 | 5.862 | 7.687 | 5.773 | 6.923 |
   | [YawPose上位80%](../../eval/yawpose_rear_search/conditions/Y_top80/summary.json) | 3.684 | 5.527 | 7.358 | 5.523 | 6.823 |
   | [YawPose上位100%](../../eval/yawpose_rear_search/conditions/Y_top100/summary.json) | 3.635 | 5.213 | 7.049 | 5.299 | 6.754 |

   300W-LP全体122,217件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 4.133 | 4.311 | 5.126 | 4.523 | 5.845 |
   | [対照条件（YawPose追加なし）](../../eval/yawpose_rear_search/conditions/Y_base/summary.json) | 4.135 | 4.080 | 4.888 | 4.368 | 5.803 |
   | [YawPose上位20%](../../eval/yawpose_rear_search/conditions/Y_top20/summary.json) | 4.050 | 8.682 | 9.415 | 7.382 | 5.931 |
   | [YawPose上位40%](../../eval/yawpose_rear_search/conditions/Y_top40/summary.json) | 3.821 | 7.502 | 8.264 | 6.529 | 5.669 |
   | [YawPose上位60%](../../eval/yawpose_rear_search/conditions/Y_top60/summary.json) | 3.826 | 6.608 | 7.377 | 5.937 | 5.618 |
   | [YawPose上位80%](../../eval/yawpose_rear_search/conditions/Y_top80/summary.json) | 3.884 | 6.070 | 6.833 | 5.596 | 5.637 |
   | [YawPose上位100%](../../eval/yawpose_rear_search/conditions/Y_top100/summary.json) | 3.922 | 5.580 | 6.335 | 5.279 | 5.650 |

   DAD-3DHeads公式validation全体4,312件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_dad_fp32/evaluations/baseline/summary.json) | 16.016 | 25.247 | 26.054 | 22.439 | 31.482 |
   | [対照条件（YawPose追加なし）](../../eval/yawpose_rear_search/conditions/Y_base_dad/summary.json) | 15.083 | 23.530 | 24.440 | 21.018 | 29.385 |
   | [YawPose上位20%](../../eval/yawpose_rear_search/conditions/Y_top20_dad/summary.json) | 15.006 | 23.257 | 24.210 | 20.825 | 29.086 |
   | [YawPose上位40%](../../eval/yawpose_rear_search/conditions/Y_top40_dad/summary.json) | 14.991 | 23.224 | 24.172 | 20.796 | 29.054 |
   | [YawPose上位60%](../../eval/yawpose_rear_search/conditions/Y_top60_dad/summary.json) | 14.987 | 23.246 | 24.198 | 20.810 | 29.071 |
   | [YawPose上位80%](../../eval/yawpose_rear_search/conditions/Y_top80_dad/summary.json) | 14.988 | 23.276 | 24.242 | 20.836 | 29.112 |
   | [YawPose上位100%](../../eval/yawpose_rear_search/conditions/Y_top100_dad/summary.json) | 14.975 | 23.300 | 24.239 | 20.838 | 29.135 |

   AGORA-HPEの後方2,644件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) | 全周yaw |
   |---|---:|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/datasets/agora_hpe/metrics/yaw_bands.json) | 26.610 | 17.807 | 22.026 | 22.148 | 37.900 | 30.800 |
   | [対照条件（YawPose追加なし）](../../eval/yawpose_rear_search/conditions/Y_base/datasets/agora_hpe/metrics/yaw_bands.json) | 24.562 | 14.719 | 20.008 | 19.763 | 34.819 | 27.720 |
   | [YawPose上位20%](../../eval/yawpose_rear_search/conditions/Y_top20/datasets/agora_hpe/metrics/yaw_bands.json) | 24.396 | 14.511 | 19.760 | 19.556 | 34.532 | 27.534 |
   | [YawPose上位40%](../../eval/yawpose_rear_search/conditions/Y_top40/datasets/agora_hpe/metrics/yaw_bands.json) | 24.348 | 14.503 | 19.731 | 19.527 | 34.484 | 27.479 |
   | [YawPose上位60%](../../eval/yawpose_rear_search/conditions/Y_top60/datasets/agora_hpe/metrics/yaw_bands.json) | 24.281 | 14.568 | 19.789 | 19.546 | 34.450 | 27.434 |
   | [YawPose上位80%](../../eval/yawpose_rear_search/conditions/Y_top80/datasets/agora_hpe/metrics/yaw_bands.json) | 24.249 | 14.554 | 19.808 | 19.537 | 34.421 | 27.385 |
   | [YawPose上位100%](../../eval/yawpose_rear_search/conditions/Y_top100/datasets/agora_hpe/metrics/yaw_bands.json) | 24.322 | 14.604 | 19.834 | 19.587 | 34.511 | 27.472 |

   全周yaw列は[保存済みの全周yaw集計](../../eval/yawpose_rear_search/yaw_summary_baseline_fp32_final.json)に基づきます。初期モデルの値は[初期モデルとの比較記録](../../eval/yawpose_rear_search/yaw_summary_baseline_fp32_final.json)で確認しています。

7. **15度ごとの選別と1エポック1回の投入**では、角度の偏りと反復を抑えました。

   | 用途 | VGGHeadsのみの系列：6条件 | VGGHeads＋DADの系列：6条件 |
   |---|---|---|
   | 学習 | VGGHeads。追加条件では選別したYawPose | VGGHeadsとDAD-3DHeads train。追加条件では選別したYawPose |
   | 検証 | VGGHeads | VGGHeads |
   | テスト | AGORA-HPE、AFLW2000、300W-LP、DAD-3DHeads公式validation | AGORA-HPE、AFLW2000、300W-LP、DAD-3DHeads公式validation |

   **層別選別**は、データを区分してから、区分ごとに選ぶ方法です。今回は後方を符号付き15度刻みの8区間に分け、各区間から同じ割合を採用しました。これにより、信頼度上位の画像が特定の角度帯へ集中することを抑えています。

   さらに、採用したYawPose画像は各エポックで1回だけ使うよう変更しました。採用率は20%・40%・60%・80%・100%で、対照条件を含む6条件を2系列、合計12条件で比較しました。

   両系列とも既存データの提示数は417,024件／エポックに固定しました。そのため、DADあり系列では学習量を増やすのではなく、VGGHeadsの一部をDADへ置き換えています。検証も両系列でVGGHeadsに統一しました。

   信頼度スコアに用いる3指標は実験6と同じですが、順位の計算と採用件数の決定を15度区間ごとに行います。各区間の件数に採用率を掛け、小数を切り上げて採用します。上位20%から100%へ採用率を増やすと、それまでの画像を含んだまま集合が広がります。

   損失式と係数は実験6と同じです。ただし、YawPoseのバッチを既存データの学習中に分散して配置し、投入しないバッチではYawPose項を0にします。YawPoseを投入する場合は、実際の件数を $n$ として $L_{\mathrm{yaw}}=\frac{1}{64}\sum_{j=1}^{n}\ell_{\mathrm{yaw},j}$ と計算します。最後のバッチが64件未満でも分母を64に保つことで、端数バッチの画像だけが強く寄与することを防ぎます。

   総損失を2で割って2バッチ分の勾配を蓄積し、既存データ128件につき1回更新します。YawPoseはその更新へ加えるため、追加すること自体で更新回数は増えません。一方、採用率を増やすと投入画像数が増えるので、係数が同じ0.2でも1エポック全体のYawPose教師信号量は増えます。[YawPose損失と選別の実装](../../src/experiments/common/yawpose.py)、[投入位置と損失集約の実装](../../src/experiments/scripts/train_yawpose_rear.py)

   AGORA-HPE後方の全周yaw誤差について、対照条件から上位80%追加への改善は、VGGHeadsのみで約0.030°、DAD併用で約0.018°でした。実験6より追加効果は小さく、AFLW2000の悪化も小さくなりました。

   一方、120〜150°付近では改善し、165〜180°付近では悪化する傾向が残りました。上位20%だけを使う条件が、一貫して最良になる結果でもありませんでした。[実験7の結果](yawpose_rear_yaw_stratified_single_draw_results.md)

   以下では全12条件を、同じテストデータセットごとに比較します。初期モデルは学習前の比較基準、対照条件はYawPoseを追加せずに追加学習したモデルです。 VGGHeads系列とVGGHeads＋DAD系列を併記し、YawPoseの効果は同じ系列の対照条件と比較します。

   AGORA-HPE全体7,505件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 27.046 | 41.840 | 42.858 | 37.248 | 45.475 |
   | [VGGHeads・対照条件](../../eval/yawpose_rear_stratified_search/conditions/Y_base_vgg/summary.json) | 26.186 | 41.294 | 42.762 | 36.747 | 44.498 |
   | [VGGHeads・各15度帯上位20%](../../eval/yawpose_rear_stratified_search/conditions/Y_top20_vgg/summary.json) | 26.184 | 41.298 | 42.766 | 36.749 | 44.494 |
   | [VGGHeads・各15度帯上位40%](../../eval/yawpose_rear_stratified_search/conditions/Y_top40_vgg/summary.json) | 26.168 | 41.301 | 42.768 | 36.746 | 44.482 |
   | [VGGHeads・各15度帯上位60%](../../eval/yawpose_rear_stratified_search/conditions/Y_top60_vgg/summary.json) | 26.147 | 41.297 | 42.761 | 36.735 | 44.460 |
   | [VGGHeads・各15度帯上位80%](../../eval/yawpose_rear_stratified_search/conditions/Y_top80_vgg/summary.json) | 26.136 | 41.297 | 42.760 | 36.731 | 44.448 |
   | [VGGHeads・各15度帯上位100%](../../eval/yawpose_rear_stratified_search/conditions/Y_top100_vgg/summary.json) | 26.153 | 41.310 | 42.776 | 36.746 | 44.471 |
   | [VGGHeads＋DAD・対照条件](../../eval/yawpose_rear_stratified_search/conditions/Y_base_vgg_dad/summary.json) | 26.214 | 41.168 | 42.626 | 36.669 | 44.415 |
   | [VGGHeads＋DAD・各15度帯上位20%](../../eval/yawpose_rear_stratified_search/conditions/Y_top20_vgg_dad/summary.json) | 26.211 | 41.171 | 42.627 | 36.670 | 44.413 |
   | [VGGHeads＋DAD・各15度帯上位40%](../../eval/yawpose_rear_stratified_search/conditions/Y_top40_vgg_dad/summary.json) | 26.202 | 41.173 | 42.630 | 36.668 | 44.404 |
   | [VGGHeads＋DAD・各15度帯上位60%](../../eval/yawpose_rear_stratified_search/conditions/Y_top60_vgg_dad/summary.json) | 26.183 | 41.175 | 42.632 | 36.663 | 44.388 |
   | [VGGHeads＋DAD・各15度帯上位80%](../../eval/yawpose_rear_stratified_search/conditions/Y_top80_vgg_dad/summary.json) | 26.176 | 41.175 | 42.632 | 36.661 | 44.377 |
   | [VGGHeads＋DAD・各15度帯上位100%](../../eval/yawpose_rear_stratified_search/conditions/Y_top100_vgg_dad/summary.json) | 26.191 | 41.176 | 42.631 | 36.666 | 44.392 |

   AFLW2000全体1,988件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 3.495 | 4.092 | 5.997 | 4.528 | 6.562 |
   | [VGGHeads・対照条件](../../eval/yawpose_rear_stratified_search/conditions/Y_base_vgg/summary.json) | 3.469 | 3.948 | 5.850 | 4.422 | 6.541 |
   | [VGGHeads・各15度帯上位20%](../../eval/yawpose_rear_stratified_search/conditions/Y_top20_vgg/summary.json) | 3.470 | 3.958 | 5.860 | 4.429 | 6.543 |
   | [VGGHeads・各15度帯上位40%](../../eval/yawpose_rear_stratified_search/conditions/Y_top40_vgg/summary.json) | 3.473 | 3.970 | 5.870 | 4.438 | 6.547 |
   | [VGGHeads・各15度帯上位60%](../../eval/yawpose_rear_stratified_search/conditions/Y_top60_vgg/summary.json) | 3.475 | 3.975 | 5.875 | 4.441 | 6.548 |
   | [VGGHeads・各15度帯上位80%](../../eval/yawpose_rear_stratified_search/conditions/Y_top80_vgg/summary.json) | 3.474 | 3.976 | 5.875 | 4.442 | 6.547 |
   | [VGGHeads・各15度帯上位100%](../../eval/yawpose_rear_stratified_search/conditions/Y_top100_vgg/summary.json) | 3.476 | 3.977 | 5.878 | 4.444 | 6.549 |
   | [VGGHeads＋DAD・対照条件](../../eval/yawpose_rear_stratified_search/conditions/Y_base_vgg_dad/summary.json) | 3.473 | 3.966 | 5.855 | 4.431 | 6.535 |
   | [VGGHeads＋DAD・各15度帯上位20%](../../eval/yawpose_rear_stratified_search/conditions/Y_top20_vgg_dad/summary.json) | 3.473 | 3.979 | 5.868 | 4.440 | 6.536 |
   | [VGGHeads＋DAD・各15度帯上位40%](../../eval/yawpose_rear_stratified_search/conditions/Y_top40_vgg_dad/summary.json) | 3.476 | 3.989 | 5.878 | 4.447 | 6.539 |
   | [VGGHeads＋DAD・各15度帯上位60%](../../eval/yawpose_rear_stratified_search/conditions/Y_top60_vgg_dad/summary.json) | 3.479 | 3.994 | 5.883 | 4.452 | 6.541 |
   | [VGGHeads＋DAD・各15度帯上位80%](../../eval/yawpose_rear_stratified_search/conditions/Y_top80_vgg_dad/summary.json) | 3.479 | 3.996 | 5.883 | 4.453 | 6.541 |
   | [VGGHeads＋DAD・各15度帯上位100%](../../eval/yawpose_rear_stratified_search/conditions/Y_top100_vgg_dad/summary.json) | 3.479 | 3.993 | 5.880 | 4.451 | 6.541 |

   300W-LP全体122,217件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 4.133 | 4.311 | 5.126 | 4.523 | 5.845 |
   | [VGGHeads・対照条件](../../eval/yawpose_rear_stratified_search/conditions/Y_base_vgg/summary.json) | 4.150 | 4.044 | 4.851 | 4.348 | 5.809 |
   | [VGGHeads・各15度帯上位20%](../../eval/yawpose_rear_stratified_search/conditions/Y_top20_vgg/summary.json) | 4.147 | 4.051 | 4.856 | 4.351 | 5.807 |
   | [VGGHeads・各15度帯上位40%](../../eval/yawpose_rear_stratified_search/conditions/Y_top40_vgg/summary.json) | 4.143 | 4.056 | 4.860 | 4.353 | 5.804 |
   | [VGGHeads・各15度帯上位60%](../../eval/yawpose_rear_stratified_search/conditions/Y_top60_vgg/summary.json) | 4.141 | 4.057 | 4.862 | 4.353 | 5.802 |
   | [VGGHeads・各15度帯上位80%](../../eval/yawpose_rear_stratified_search/conditions/Y_top80_vgg/summary.json) | 4.146 | 4.058 | 4.864 | 4.356 | 5.806 |
   | [VGGHeads・各15度帯上位100%](../../eval/yawpose_rear_stratified_search/conditions/Y_top100_vgg/summary.json) | 4.140 | 4.057 | 4.861 | 4.353 | 5.800 |
   | [VGGHeads＋DAD・対照条件](../../eval/yawpose_rear_stratified_search/conditions/Y_base_vgg_dad/summary.json) | 4.132 | 4.082 | 4.894 | 4.369 | 5.803 |
   | [VGGHeads＋DAD・各15度帯上位20%](../../eval/yawpose_rear_stratified_search/conditions/Y_top20_vgg_dad/summary.json) | 4.131 | 4.094 | 4.906 | 4.377 | 5.804 |
   | [VGGHeads＋DAD・各15度帯上位40%](../../eval/yawpose_rear_stratified_search/conditions/Y_top40_vgg_dad/summary.json) | 4.125 | 4.099 | 4.908 | 4.378 | 5.799 |
   | [VGGHeads＋DAD・各15度帯上位60%](../../eval/yawpose_rear_stratified_search/conditions/Y_top60_vgg_dad/summary.json) | 4.120 | 4.103 | 4.912 | 4.378 | 5.794 |
   | [VGGHeads＋DAD・各15度帯上位80%](../../eval/yawpose_rear_stratified_search/conditions/Y_top80_vgg_dad/summary.json) | 4.121 | 4.100 | 4.911 | 4.377 | 5.795 |
   | [VGGHeads＋DAD・各15度帯上位100%](../../eval/yawpose_rear_stratified_search/conditions/Y_top100_vgg_dad/summary.json) | 4.124 | 4.102 | 4.913 | 4.379 | 5.798 |

   DAD-3DHeads公式validation全体4,312件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_dad_fp32/evaluations/baseline/summary.json) | 16.016 | 25.247 | 26.054 | 22.439 | 31.482 |
   | [VGGHeads・対照条件](../../eval/yawpose_rear_stratified_search/conditions/Y_base_vgg__dad/summary.json) | 14.970 | 23.780 | 24.769 | 21.173 | 29.497 |
   | [VGGHeads・各15度帯上位20%](../../eval/yawpose_rear_stratified_search/conditions/Y_top20_vgg__dad/summary.json) | 14.966 | 23.771 | 24.762 | 21.167 | 29.487 |
   | [VGGHeads・各15度帯上位40%](../../eval/yawpose_rear_stratified_search/conditions/Y_top40_vgg__dad/summary.json) | 14.958 | 23.773 | 24.763 | 21.165 | 29.481 |
   | [VGGHeads・各15度帯上位60%](../../eval/yawpose_rear_stratified_search/conditions/Y_top60_vgg__dad/summary.json) | 14.973 | 23.781 | 24.775 | 21.176 | 29.498 |
   | [VGGHeads・各15度帯上位80%](../../eval/yawpose_rear_stratified_search/conditions/Y_top80_vgg__dad/summary.json) | 14.967 | 23.777 | 24.770 | 21.171 | 29.491 |
   | [VGGHeads・各15度帯上位100%](../../eval/yawpose_rear_stratified_search/conditions/Y_top100_vgg__dad/summary.json) | 14.973 | 23.794 | 24.787 | 21.185 | 29.506 |
   | [VGGHeads＋DAD・対照条件](../../eval/yawpose_rear_stratified_search/conditions/Y_base_vgg_dad__dad/summary.json) | 15.125 | 23.599 | 24.513 | 21.079 | 29.460 |
   | [VGGHeads＋DAD・各15度帯上位20%](../../eval/yawpose_rear_stratified_search/conditions/Y_top20_vgg_dad__dad/summary.json) | 15.122 | 23.593 | 24.508 | 21.074 | 29.454 |
   | [VGGHeads＋DAD・各15度帯上位40%](../../eval/yawpose_rear_stratified_search/conditions/Y_top40_vgg_dad__dad/summary.json) | 15.120 | 23.591 | 24.505 | 21.072 | 29.452 |
   | [VGGHeads＋DAD・各15度帯上位60%](../../eval/yawpose_rear_stratified_search/conditions/Y_top60_vgg_dad__dad/summary.json) | 15.118 | 23.595 | 24.508 | 21.074 | 29.454 |
   | [VGGHeads＋DAD・各15度帯上位80%](../../eval/yawpose_rear_stratified_search/conditions/Y_top80_vgg_dad__dad/summary.json) | 15.124 | 23.592 | 24.504 | 21.073 | 29.453 |
   | [VGGHeads＋DAD・各15度帯上位100%](../../eval/yawpose_rear_stratified_search/conditions/Y_top100_vgg_dad__dad/summary.json) | 15.115 | 23.593 | 24.503 | 21.070 | 29.450 |

   AGORA-HPEの後方2,644件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) | 全周yaw |
   |---|---:|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/datasets/agora_hpe/metrics/yaw_bands.json) | 26.610 | 17.807 | 22.026 | 22.148 | 37.900 | 30.800 |
   | [VGGHeads・対照条件](../../eval/yawpose_rear_stratified_search/conditions/Y_base_vgg/datasets/agora_hpe/metrics/yaw_bands.json) | 24.397 | 14.169 | 19.479 | 19.348 | 34.479 | 27.333 |
   | [VGGHeads・各15度帯上位20%](../../eval/yawpose_rear_stratified_search/conditions/Y_top20_vgg/datasets/agora_hpe/metrics/yaw_bands.json) | 24.394 | 14.178 | 19.485 | 19.352 | 34.481 | 27.337 |
   | [VGGHeads・各15度帯上位40%](../../eval/yawpose_rear_stratified_search/conditions/Y_top40_vgg/datasets/agora_hpe/metrics/yaw_bands.json) | 24.377 | 14.178 | 19.483 | 19.346 | 34.466 | 27.325 |
   | [VGGHeads・各15度帯上位60%](../../eval/yawpose_rear_stratified_search/conditions/Y_top60_vgg/datasets/agora_hpe/metrics/yaw_bands.json) | 24.356 | 14.172 | 19.473 | 19.334 | 34.442 | 27.307 |
   | [VGGHeads・各15度帯上位80%](../../eval/yawpose_rear_stratified_search/conditions/Y_top80_vgg/datasets/agora_hpe/metrics/yaw_bands.json) | 24.348 | 14.177 | 19.476 | 19.334 | 34.433 | 27.303 |
   | [VGGHeads・各15度帯上位100%](../../eval/yawpose_rear_stratified_search/conditions/Y_top100_vgg/datasets/agora_hpe/metrics/yaw_bands.json) | 24.360 | 14.161 | 19.465 | 19.329 | 34.438 | 27.301 |
   | [VGGHeads＋DAD・対照条件](../../eval/yawpose_rear_stratified_search/conditions/Y_base_vgg_dad/datasets/agora_hpe/metrics/yaw_bands.json) | 24.596 | 14.804 | 20.077 | 19.825 | 34.869 | 27.775 |
   | [VGGHeads＋DAD・各15度帯上位20%](../../eval/yawpose_rear_stratified_search/conditions/Y_top20_vgg_dad/datasets/agora_hpe/metrics/yaw_bands.json) | 24.592 | 14.798 | 20.070 | 19.820 | 34.865 | 27.771 |
   | [VGGHeads＋DAD・各15度帯上位40%](../../eval/yawpose_rear_stratified_search/conditions/Y_top40_vgg_dad/datasets/agora_hpe/metrics/yaw_bands.json) | 24.586 | 14.809 | 20.078 | 19.824 | 34.862 | 27.772 |
   | [VGGHeads＋DAD・各15度帯上位60%](../../eval/yawpose_rear_stratified_search/conditions/Y_top60_vgg_dad/datasets/agora_hpe/metrics/yaw_bands.json) | 24.569 | 14.799 | 20.063 | 19.810 | 34.840 | 27.758 |
   | [VGGHeads＋DAD・各15度帯上位80%](../../eval/yawpose_rear_stratified_search/conditions/Y_top80_vgg_dad/datasets/agora_hpe/metrics/yaw_bands.json) | 24.565 | 14.807 | 20.069 | 19.814 | 34.835 | 27.758 |
   | [VGGHeads＋DAD・各15度帯上位100%](../../eval/yawpose_rear_stratified_search/conditions/Y_top100_vgg_dad/datasets/agora_hpe/metrics/yaw_bands.json) | 24.575 | 14.817 | 20.080 | 19.824 | 34.851 | 27.765 |

   全周yaw列は[保存済みの全周yaw集計](../../eval/yawpose_rear_stratified_search/yaw_summary_baseline_fp32_final.json)に基づきます。初期モデルの値は[初期モデルとの比較記録](../../eval/yawpose_rear_search/yaw_summary_baseline_fp32_final.json)で確認しています。

8. **YawPose損失係数の比較**では、使用画像を固定して教師信号の強さを変えました。

   | 用途 | 使用データ |
   |---|---|
   | 学習 | 全6条件でVGGHeadsとDAD-3DHeads train。YawPose追加条件では、各15度帯の信頼度上位60%に当たる同一の8,433件 |
   | 検証 | VGGHeads |
   | テスト | AGORA-HPE、AFLW2000、300W-LP、DAD-3DHeads公式validation |

   最新の実験です。YawPoseを追加する5条件では画像集合を同じにし、yaw損失へ掛ける係数だけを0.2・0.4・0.6・0.8・1.0と変えました。これにより、画像の品質や角度分布を変えずに、YawPoseの教師信号を強めた効果を比較しています。

   各YawPose画像は1エポックに1回使用し、各条件を10エポック学習しました。テストには、検証で選ばれた途中のモデルではなく、すべて最終エポックのモデルを使っています。

   総損失は $L=L_{\mathrm{sup}}+1.0L_{\mathrm{distill}}+0.2L_{\mathrm{flip}}+\lambda_{\mathrm{yaw}}L_{\mathrm{yaw}}$ です。YawPose追加条件では、$\lambda_{\mathrm{yaw}}$ だけを0.2・0.4・0.6・0.8・1.0と変えます。蒸留と反転整合性の係数は固定します。

   YawPoseの投入方法と端数バッチの扱いは実験7と同じです。8,433件を64件ずつに分けるため、1エポックに132バッチを投入し、最後は49件です。追加条件間では選択画像、各エポックの提示順、提示回数を揃えています。このため、実験7の採用率比較に含まれていた画像集合と投入量の違いを固定して、係数の影響を比較できます。同じ予測値に対して係数を0.2から1.0へ変えるとYawPose項とその勾配は5倍になりますが、学習後の誤差改善量が5倍になることを意味するものではありません。

   | YawPose係数 | AGORA全体 SO(3) | AGORA後方 全周yaw | AFLW2000 SO(3) | 300W-LP SO(3) | DAD公式validation SO(3) |
   |---|---:|---:|---:|---:|---:|
   | 対照条件 | 44.415° | 27.775° | 6.535° | 5.803° | 29.460° |
   | 0.2 | 44.388° | 27.758° | 6.541° | 5.794° | 29.454° |
   | 0.4 | 44.357° | 27.754° | 6.547° | 5.790° | 29.450° |
   | 0.6 | 44.336° | 27.745° | 6.552° | 5.781° | 29.453° |
   | 0.8 | 44.323° | 27.737° | 6.555° | 5.781° | 29.445° |
   | 1.0 | 44.308° | 27.730° | 6.560° | 5.777° | 29.437° |

   係数を強めると、AGORA-HPEと300W-LPの平均SO(3)誤差は低下する一方、AFLW2000では増加しました。係数1.0でのAGORA-HPE後方の全周yaw改善は、対照条件に対して約0.045°でした。

   一方、300W-LPの3軸平均は、対照条件の4.369°から係数1.0の4.414°へ悪化しました。yawの改善と同時にroll・pitchの誤差が増えており、SO(3)の改善を、各軸が一様に改善した結果と解釈することはできません。

   角度別では、係数1.0によって120〜150°の全周yaw誤差が左右それぞれ約0.37〜0.42°改善し、150〜180°では約0.30〜0.31°悪化しました。後方全体の小さな改善は、すべての後方角度が少しずつ改善した結果ではありません。[実験8の保存結果](../../eval/yawpose_rear_weight_search/weight_summary.json)

   以下では全6条件を、同じテストデータセットごとに比較します。初期モデルは学習前の比較基準、対照条件はYawPoseを追加せずに追加学習したモデルです。

   AGORA-HPE全体7,505件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 27.046 | 41.840 | 42.858 | 37.248 | 45.475 |
   | [対照条件（YawPose追加なし）](../../eval/yawpose_rear_weight_search/conditions/Y_base_vgg_dad/summary.json) | 26.214 | 41.168 | 42.626 | 36.669 | 44.415 |
   | [YawPose係数0.2](../../eval/yawpose_rear_weight_search/conditions/Y_w020_vgg_dad/summary.json) | 26.183 | 41.175 | 42.632 | 36.663 | 44.388 |
   | [YawPose係数0.4](../../eval/yawpose_rear_weight_search/conditions/Y_w040_vgg_dad/summary.json) | 26.158 | 41.182 | 42.638 | 36.659 | 44.357 |
   | [YawPose係数0.6](../../eval/yawpose_rear_weight_search/conditions/Y_w060_vgg_dad/summary.json) | 26.137 | 41.192 | 42.644 | 36.658 | 44.336 |
   | [YawPose係数0.8](../../eval/yawpose_rear_weight_search/conditions/Y_w080_vgg_dad/summary.json) | 26.122 | 41.202 | 42.653 | 36.659 | 44.323 |
   | [YawPose係数1.0](../../eval/yawpose_rear_weight_search/conditions/Y_w100_vgg_dad/summary.json) | 26.106 | 41.213 | 42.665 | 36.662 | 44.308 |

   AFLW2000全体1,988件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 3.495 | 4.092 | 5.997 | 4.528 | 6.562 |
   | [対照条件（YawPose追加なし）](../../eval/yawpose_rear_weight_search/conditions/Y_base_vgg_dad/summary.json) | 3.473 | 3.966 | 5.855 | 4.431 | 6.535 |
   | [YawPose係数0.2](../../eval/yawpose_rear_weight_search/conditions/Y_w020_vgg_dad/summary.json) | 3.479 | 3.994 | 5.883 | 4.452 | 6.541 |
   | [YawPose係数0.4](../../eval/yawpose_rear_weight_search/conditions/Y_w040_vgg_dad/summary.json) | 3.484 | 4.020 | 5.907 | 4.470 | 6.547 |
   | [YawPose係数0.6](../../eval/yawpose_rear_weight_search/conditions/Y_w060_vgg_dad/summary.json) | 3.489 | 4.046 | 5.933 | 4.489 | 6.552 |
   | [YawPose係数0.8](../../eval/yawpose_rear_weight_search/conditions/Y_w080_vgg_dad/summary.json) | 3.491 | 4.066 | 5.951 | 4.503 | 6.555 |
   | [YawPose係数1.0](../../eval/yawpose_rear_weight_search/conditions/Y_w100_vgg_dad/summary.json) | 3.495 | 4.089 | 5.974 | 4.519 | 6.560 |

   300W-LP全体122,217件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/summary.json) | 4.133 | 4.311 | 5.126 | 4.523 | 5.845 |
   | [対照条件（YawPose追加なし）](../../eval/yawpose_rear_weight_search/conditions/Y_base_vgg_dad/summary.json) | 4.132 | 4.082 | 4.894 | 4.369 | 5.803 |
   | [YawPose係数0.2](../../eval/yawpose_rear_weight_search/conditions/Y_w020_vgg_dad/summary.json) | 4.120 | 4.103 | 4.912 | 4.378 | 5.794 |
   | [YawPose係数0.4](../../eval/yawpose_rear_weight_search/conditions/Y_w040_vgg_dad/summary.json) | 4.114 | 4.117 | 4.926 | 4.386 | 5.790 |
   | [YawPose係数0.6](../../eval/yawpose_rear_weight_search/conditions/Y_w060_vgg_dad/summary.json) | 4.103 | 4.135 | 4.940 | 4.393 | 5.781 |
   | [YawPose係数0.8](../../eval/yawpose_rear_weight_search/conditions/Y_w080_vgg_dad/summary.json) | 4.102 | 4.155 | 4.957 | 4.405 | 5.781 |
   | [YawPose係数1.0](../../eval/yawpose_rear_weight_search/conditions/Y_w100_vgg_dad/summary.json) | 4.097 | 4.172 | 4.972 | 4.414 | 5.777 |

   DAD-3DHeads公式validation全体4,312件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) |
   |---|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_dad_fp32/evaluations/baseline/summary.json) | 16.016 | 25.247 | 26.054 | 22.439 | 31.482 |
   | [対照条件（YawPose追加なし）](../../eval/yawpose_rear_weight_search/conditions/Y_base_vgg_dad__dad/summary.json) | 15.125 | 23.599 | 24.513 | 21.079 | 29.460 |
   | [YawPose係数0.2](../../eval/yawpose_rear_weight_search/conditions/Y_w020_vgg_dad__dad/summary.json) | 15.118 | 23.595 | 24.508 | 21.074 | 29.454 |
   | [YawPose係数0.4](../../eval/yawpose_rear_weight_search/conditions/Y_w040_vgg_dad__dad/summary.json) | 15.119 | 23.592 | 24.506 | 21.072 | 29.450 |
   | [YawPose係数0.6](../../eval/yawpose_rear_weight_search/conditions/Y_w060_vgg_dad__dad/summary.json) | 15.122 | 23.595 | 24.509 | 21.075 | 29.453 |
   | [YawPose係数0.8](../../eval/yawpose_rear_weight_search/conditions/Y_w080_vgg_dad__dad/summary.json) | 15.118 | 23.589 | 24.502 | 21.070 | 29.445 |
   | [YawPose係数1.0](../../eval/yawpose_rear_weight_search/conditions/Y_w100_vgg_dad__dad/summary.json) | 15.114 | 23.585 | 24.499 | 21.066 | 29.437 |

   AGORA-HPEの後方2,644件の平均誤差（°）は次のとおりです。

   | 条件 | yaw（Euler） | roll | pitch | 3軸平均 | SO(3) | 全周yaw |
   |---|---:|---:|---:|---:|---:|---:|
   | [初期モデル](../../eval/baseline_fp32/evaluations/baseline/datasets/agora_hpe/metrics/yaw_bands.json) | 26.610 | 17.807 | 22.026 | 22.148 | 37.900 | 30.800 |
   | [対照条件（YawPose追加なし）](../../eval/yawpose_rear_weight_search/conditions/Y_base_vgg_dad/datasets/agora_hpe/metrics/yaw_bands.json) | 24.596 | 14.804 | 20.077 | 19.825 | 34.869 | 27.775 |
   | [YawPose係数0.2](../../eval/yawpose_rear_weight_search/conditions/Y_w020_vgg_dad/datasets/agora_hpe/metrics/yaw_bands.json) | 24.569 | 14.799 | 20.063 | 19.810 | 34.840 | 27.758 |
   | [YawPose係数0.4](../../eval/yawpose_rear_weight_search/conditions/Y_w040_vgg_dad/datasets/agora_hpe/metrics/yaw_bands.json) | 24.553 | 14.815 | 20.071 | 19.813 | 34.823 | 27.754 |
   | [YawPose係数0.6](../../eval/yawpose_rear_weight_search/conditions/Y_w060_vgg_dad/datasets/agora_hpe/metrics/yaw_bands.json) | 24.539 | 14.817 | 20.067 | 19.808 | 34.804 | 27.745 |
   | [YawPose係数0.8](../../eval/yawpose_rear_weight_search/conditions/Y_w080_vgg_dad/datasets/agora_hpe/metrics/yaw_bands.json) | 24.530 | 14.815 | 20.060 | 19.801 | 34.789 | 27.737 |
   | [YawPose係数1.0](../../eval/yawpose_rear_weight_search/conditions/Y_w100_vgg_dad/datasets/agora_hpe/metrics/yaw_bands.json) | 24.518 | 14.814 | 20.057 | 19.796 | 34.775 | 27.730 |

   全周yaw列は[保存済みの全周yaw集計](../../eval/yawpose_rear_weight_search/weight_summary.json)に基づきます。初期モデルの値は[初期モデルとの比較記録](../../eval/yawpose_rear_search/yaw_summary_baseline_fp32_final.json)で確認しています。

これらの結果には、共通して次の評価上の制約があります。

- 学習は主に同じ乱数設定で1回ずつ実行されており、小さな差について学習のばらつきを確認していません。
- 300W-LPは初期モデルの事前学習にも使われているため、未知データへの汎化とは別に解釈する必要があります。
- DAD公式validationの後方は47件しかなく、後方だけの改善を判断するには不確実性があります。
- 実験6のYawPose上位100%条件は、最終エポックの内部検証ファイルが空です。テスト結果は保存されていますが、その内部検証値は再確認できません。
