# 来源和第三方数据条款

本仓库基于 [Runz96/DietAI24](https://github.com/Runz96/DietAI24) 及 THwo0t 的复现。原作者 README 中的 MIT 声明保留于 `UPSTREAM_README.md`；上游没有单独的 LICENSE 文件。本交付不为第三方资料虚构额外许可。

- 论文：Communications Medicine 5, 458 (2025)，[DOI](https://doi.org/10.1038/s43856-025-01159-0)。不捆绑出版商PDF。
- ASA24：NCI [官方下载页](https://epi.grants.cancer.gov/asa24/resources/portionsize.html)。该页允许使用图像，制作方为 Baylor College of Medicine 儿童营养研究中心、Westat 与 NCI。ASA24 是 HHS 注册商标。本仓库只提供开发图选择清单和下载/导入工具，不直接上传原图。
- Nutrition5k：Quin Thames 等，CVPR 2021。[Google原项目](https://github.com/google-research-datasets/Nutrition5k)，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本仓库提供官方对象下载地址和固定开发图哈希，署名不表示作者或Google背书。
- NutritionVerse-Real：[作者Kaggle数据集v2](https://www.kaggle.com/datasets/nutritionverse/nutritionverse-real/versions/2)，来源记录为CC BY-NC-SA 4.0。请保留下载版本附带条款。图像和逐图营养标签未上传。
- FNDDS 2019–2020：USDA Agricultural Research Service, Food Surveys Research Group，[官方下载](https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fndds-download-databases/)。运行包包含原工作簿、标准单位资料和派生目录，用于恢复相同计算依据。
- 保存的5624份向量由FNDDS描述生成，没有模型权重、密钥或图片参考库。运行包保留原Chroma图，避免重新建图导致候选变化；未纳入刚提出的检索修复。
- uv、Python及全部第三方依赖保留各自许可，由安装器按版本下载，不打包本机虚拟环境。

运行包逐文件列表、大小和SHA-256见 `runtime/release.json`。独立验收数据、旧模型原回答、样本真实营养标签与凭据不在Git或运行包中。
