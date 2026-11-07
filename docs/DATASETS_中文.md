# 开发图像获取与跨设备迁移

`data/manifests/` 保存固定选图和 SHA-256。运行命令时从仓库根目录开始。工具默认打印计划，显式传入 `--apply` 才从本地导入，显式传入 `--download` 才允许向 Nutrition5k 官方存储发送图像下载请求。已有目标只在字节校验通过时复用。目标均在 `data/images/`，不随代码提交。工具仅处理图像。

| 清单 | 图像数 | 样本数 | 用途 |
| --- | ---: | ---: | --- |
| `historical` | 663 | 843 | 原混合 C 开发范围。ASA24 有 349 张、529 条关联，Nutrition5k 有 314 张。 |
| `development` | 3000 | 3000 | 后续十轮的固定开发池。每轮 ASA24、Nutrition5k 各 150 张，批次编号 1 至 10。 |
| `external50` | 50 | 50 | NutritionVerse-Real v2 Val 的历史第三方复测子集。 |

```bash
python tools/fetch_data.py historical
python tools/fetch_data.py development --round 8
python tools/fetch_data.py external50
```

## ASA24

使用 [NCI 份量图库](https://epi.grants.cancer.gov/asa24/resources/portionsize.html) 的 [2011–2022 官方 ZIP](https://asa24.nci.nih.gov/Asa24PortionImages.zip)。原实验 ZIP 是 670,269,206 字节，SHA-256 为 `2c0c68caaba894f43a87277a8d2484eded03c630d65fb7a308a590b2457e1c38`，随附 `ASA24_ImageMetadata.xlsx`。NCI 说明图像可自由使用。图像由 Baylor College of Medicine 的 Children's Nutrition Research Center、Westat 与 NCI 开发。ASA24 是 HHS 注册商标。

工具读取已选文件名对应的 ZIP 成员，按图像 SHA-256 检查原始字节。不同版本有可能缺少成员或使用不同字节，导入会逐项报告。后续池 ASA24 的 1500 张都有归档文件名。历史 ASA24 有 343 张确认了归档文件名，剩余 6 张需从既有历史开发图迁移。

```bash
python tools/fetch_data.py development --dataset asa24 --round 8 --apply --archive /path/to/Asa24PortionImages.zip
python tools/fetch_data.py historical --dataset asa24 --apply --archive /path/to/Asa24PortionImages.zip --source-dir /path/to/historical/images/dev
```

历史本地迁移时，`--source-dir` 指向原项目 `optimization_sol_20261102/mae_v2/round11/prepared/cohort/images/dev/` 的副本，工具按清单中的 `historical_source_path` 取图。也可先把已知同字节文件放到清单指定的目标位置，再执行校验。旧完整图片源包请留在本机数据目录。

## Nutrition5k

[Google 官方下载说明](https://github.com/google-research-datasets/Nutrition5k#download-data) · [官方对象存储](https://console.cloud.google.com/storage/browser/nutrition5k_dataset) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。署名 Quin Thames 等，CVPR 2021。每张图都指向官方 `realsense_overhead/<dish_id>/rgb.png` 对象。后续 1500 张可按对象下载，总计 611,018,069 字节。历史 314 张中 249 张能用现有官方对象回执确认原图字节，另外 65 张的 URL 按公开 dish ID 定位，是否与原输入一致由 SHA-256 校验。若官方对象不可取或字节不匹配，从旧开发图 `images/dev` 迁移。

```bash
python tools/fetch_data.py development --dataset nutrition5k --round 8 --download
python tools/fetch_data.py historical --dataset nutrition5k --download --source-dir /path/to/historical/images/dev
```

评分资料另行由数据使用者获取。官方 [cafe1 元数据](https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv) 与 [cafe2 元数据](https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv) 含整盘目标。评分只应解析固定开发清单匹配的行，且在预测保存后进行。

## NutritionVerse-Real

从 [作者的 Kaggle 数据集 v2](https://www.kaggle.com/datasets/nutritionverse/nutritionverse-real/versions/2) 获取历史版本，解包到本地，保留原相对文件路径。若已解包目录从更深的层级开始，请选择能让清单内 `source_path` 成立的目录。清单保存 50 个原 JPEG 成员名、字节数及 SHA-256。原项目来源记录采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，取得该版时保留发布者的许可条款。作者和数据集出处见 [NutritionVerse-Real 论文](https://arxiv.org/abs/2401.08598)。

```bash
python tools/fetch_data.py external50 --apply --source-dir /path/to/unpacked-kaggle-v2
```

外部 50 张为历史开发复测，按旧样本选择逐字节恢复。评分应使用 v2 对应的整盘参考字段，重量与宏量营养单位为 g，热量为 kcal。

## 评分和数据边界

清单给出图像身份、来源、固定选择、样本对应和哈希。营养标签需要自行从对应官方版本恢复，或按已有权限在设备间单独迁移。原混合 C 的 843 条旧报告还依赖独立保存的历史开发标签与阶段产物。完整历史复算需使用这些资产及原检索索引。封存独立验收使用另外的保护流程。

食品描述、标准份量和营养值来源为 [USDA FNDDS 2019–2020](https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fndds-download-databases/)。营养值使用该版 XLSX。份量编码来源为详细 SAS 发布版中的 `foodweights.sas7bdat` 和 `foodportiondesc.sas7bdat`。

历史关联与组ID已做稳定匿名化，避免旧ASA关联名直接暴露参考食品码。若按已有权限迁移旧标签，关联键的本地转换公式为 `historical-` 加原 sample_id UTF-8 字节的完整SHA-256；图片ID则保持图片哈希。此转换不改变图片、分组关系或折号。
