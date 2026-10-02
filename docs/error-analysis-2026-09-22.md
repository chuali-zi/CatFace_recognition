# Test 陌生猫误认诊断（2026-09-22）

这是 `experiments/runs/edge160/baseline/test/` 的图片级诊断，不是现场来访误认率。固定 test 分区共有 121 张陌生查询，阈值为 `0.9027086322821208`；其中 15 张达到阈值，FPIR 为 `15/121 = 0.1240`。陌生身份和 gallery 身份在 protocol 中是隔离的。

误认按真实陌生身份（分母为该身份在 test unknown 中的全部图片数）集中如下：

| 真实身份 | 该身份图片数 | 达阈值误认 | 误认 gallery 身份 | 分数（score） |
| --- | ---: | ---: | --- | --- |
| `individuals:0267` | 7 | 6 | `individuals:0028` | 0.9097–0.9182 |
| `individuals:0348` | 10 | 4 | `individuals:0208` | 0.9046–0.9125 |
| `individuals:0405` | 4 | 2 | `individuals:0391` | 0.9200–0.9323 |
| `individuals:0430` | 8 | 1 | `individuals:0028` | 0.9030 |
| `individuals:0409` | 3 | 1 | `individuals:0482` | 0.9070 |
| `individuals:0071` | 11 | 1 | `individuals:0391` | 0.9031 |

前三个真实身份贡献 `12/15` 个接受误认；这只是该固定图片分区中的集中现象，不能外推到新的场次或真实门口事件。其余 106 张陌生图片低于当前阈值。每张陌生图片的最高分图库身份都与真实身份不同，但只有达到阈值的接受误认才计入 FPIR。

接受误认的真实样本路径如下：

- `individuals:0267` → `individuals:0028`：
  `clean/faces/individuals/0267/0267_000.png`, `clean/faces/individuals/0267/0267_001.png`, `clean/faces/individuals/0267/0267_002.png`, `clean/faces/individuals/0267/0267_003.png`, `clean/faces/individuals/0267/0267_005.png`, `clean/faces/individuals/0267/0267_006.png`。
- `individuals:0348` → `individuals:0208`：
  `clean/faces/individuals/0348/0348_000.png`, `clean/faces/individuals/0348/0348_001.png`, `clean/faces/individuals/0348/0348_003.png`, `clean/faces/individuals/0348/0348_005.png`。
- `individuals:0405` → `individuals:0391`：
  `clean/faces/individuals/0405/0405_003.png`, `clean/faces/individuals/0405/0405_015.png`。
- `individuals:0430` → `individuals:0028`：
  `clean/faces/individuals/0430/0430_011.png`。
- `individuals:0409` → `individuals:0482`：
  `clean/faces/individuals/0409/0409_006.png`。
- `individuals:0071` → `individuals:0391`：
  `clean/faces/individuals/0071/0071_009.png`。

对应 gallery 注册图片（每只猫三张）为：

- `individuals:0028`: `clean/faces/individuals/0028/0028_004.png`, `clean/faces/individuals/0028/0028_019.png`, `clean/faces/individuals/0028/0028_022.png`。
- `individuals:0208`: `clean/faces/individuals/0208/0208_022.png`, `clean/faces/individuals/0208/0208_020.png`, `clean/faces/individuals/0208/0208_019.png`。
- `individuals:0391`: `clean/faces/individuals/0391/0391_008.png`, `clean/faces/individuals/0391/0391_003.png`, `clean/faces/individuals/0391/0391_013.png`。
- `individuals:0482`: `clean/faces/individuals/0482/0482_018.png`, `clean/faces/individuals/0482/0482_007.png`, `clean/faces/individuals/0482/0482_048.png`。

分数距离阈值很近的边界样本包括 `0430/0430_011.png`（`+0.000299`）和 `0071/0071_009.png`（`+0.000411`）；`0348` 的四张误认距离为 `+0.001900` 至 `+0.009835`。`0267` 的六张误认距离为 `+0.007025` 至 `+0.015509`。这里的“距离”仅指 `score - threshold`，不是未保存的第二名排序 margin；本分析没有重新调阈值。

## 少量视觉核对

仅核对两组代表性配对，没有把视觉印象当作标签证据：

- `0267/0267_000.png` 对 `0028/0028_004.png`：两张都是正面近距离脸部，具有相近的棕/灰虎斑和浅色口鼻区域；光照、裁剪和眼睛状态不同。更像是花色/脸部纹理相似导致的原型混淆，不能据此判断身份标注错误。
- `0348/0348_000.png` 对 `0208/0208_022.png`：两张都是橙色虎斑、正面猫脸，眼睛和鼻口区域相似，但毛色深浅和条纹细节仍有差异。该组支持“同花色、相近正面视角”的谨慎解释，不支持修改标签。

样本由固定 test protocol 和模型阈值筛出，视觉配对也有选择偏差；原图和 clean face 均保留，未修改标签、未重划 test、未删除图片。
