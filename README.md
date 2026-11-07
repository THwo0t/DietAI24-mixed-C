# DietAI24 原版混合 C

这是可独立拉取的原版混合 C 开发基线。保留原有模型分工、多查询检索 FNDDS、1–3 候选码平均和估重方法；不加入后续审核、标准份量新法、修正系数或检索修复。

| 环节 | Nutrition5k / 无标签第三方图片 | ASA24 |
|---|---|---|
| 识别、选码 | Gemini 3.8 Flash | Gemini 3.8 Flash |
| 查询改写 | GPT 6.1 Sol | GPT 6.1 Sol |
| 检索 | 原 Chroma 索引，每条查询取4项 | 相同 |
| 估重 | Gemini：整盘重量，再分配给各食物 | Sol：连续克数，再匹配合法份量 |
| 营养 | 同一食物1–3码营养密度平均×该项重量，再加总 | 条件评测使用命中参考码的密度×重量 |

聊天模型走 `https://api.zicc.cc/v1`，检索文字向量走 `https://api.openai.com/v1`。模型名及可用性取决于账号；失败不会自动换模型。此前的具体付费授权不随仓库转移。

## 新设备开始

```bash
git clone https://github.com/THwo0t/DietAI24-mixed-C.git
cd DietAI24-mixed-C
./setup.sh
.venv/bin/python -B tools/test_offline.py
```

`setup.sh` 下载固定 uv、Python 3.10.13、完整锁定依赖和约165.5 MB运行数据包，全部放在项目内。运行包包含原 FNDDS 表、目录、5624份保存的向量和原 Chroma 索引，不需要付费重建向量。只装代码环境可运行 `./setup.sh code`。安装不调用模型。

适配 Linux x86_64、Apple Silicon / Intel Mac 的安装入口；Windows 使用 WSL2。实际已完成的验证见 [VALIDATION.md](VALIDATION.md)，不要把安装入口存在当成所有平台已实测。

**继续开发先读 [HANDOFF.md](HANDOFF.md)。数据下载地址、固定图片清单和恢复限制见 [数据说明](docs/DATASETS_中文.md)。**

## 图片和免费检查

后续开发池有 ASA24、N5 各1500张，分十轮，每轮各150张。原项目已经用过第1–7轮；接续时从第8轮开始，不把新 clone 当作未使用数据。

```bash
# 只列计划，不下载、不调用模型
.venv/bin/python tools/fetch_data.py development --round 8
# N5 按固定清单从 Google 官方存储下载
.venv/bin/python tools/fetch_data.py development --dataset nutrition5k --round 8 --download
# ASA24 先从文档里的 NCI 官方地址取得压缩包，再按清单导入
.venv/bin/python tools/fetch_data.py development --dataset asa24 --round 8 --apply --archive /path/to/Asa24PortionImages.zip
# 检查本地图片、方法和运行范围，不付费
.venv/bin/python tools/run_mixed_c.py --dataset nutrition5k --manifest data/manifests/development.json --round 8
```

历史843条记录实际对应663张独特图；后续3000张和第三方50张另有独立清单。三个集合有历史交集，不能相加称为3713张未见图。图像及营养参考标签不直接放进 Git；公开来源和哈希在清单里。6张旧 ASA 图需旧开发目录迁移，65张旧 N5 图需要进一步核对官方对象字节；这些限制不影响后续3000张的完整来源清单。

## 获得本轮付费授权后运行

```bash
cp .env.example .env
chmod 600 .env
# 在本机编辑 .env 填入三种用途的 key，勿发送或提交其内容
.venv/bin/python tools/run_mixed_c.py \
  --dataset nutrition5k --manifest data/manifests/development.json --round 8 \
  --output runs/round08-n5 --workers 32 --allow-paid
```

自己的无标签图片或 NutritionVerse-Real 使用相同 N5 路径：

```bash
.venv/bin/python tools/run_mixed_c.py --image /absolute/path/to/meal.jpg
# 经明确授权后才在上述命令增加 --allow-paid；可用多个 --image 或盲清单
```

ASA24 无参考码时输出独立的端到端估计；论文式条件评测另传 `--reference-codes data/local/asa_codes.json`。文件只有 `association_id`、`sample_id`、`code`，格式见 `examples/asa_reference_codes.json`。所有图片先完成并保存盲识别选码，之后才读取此文件；只有最终选择的1–3码中存在参考码时才估条件份量。样本正确份量、真实重量及营养值不得出现在该文件。这个条件协议不等于普通无标签产品推理。

每个请求最多累计3次传输；成功请求复用缓存，合同失败也保留，不为获得更好的答案重采样。结果在指定 `runs/`，可用完全相同命令续传；已完成的运行不会再次付费。接口价格未核实，仓库不承诺具体费用。

## 离线评分

预测完成后，把对应版本的参考标签另存于本机 `data/local/labels.json`，键必须与输出样本ID完全对应，值包含五项。字段形式见 `examples/labels_format.json`，其中数值仅是格式示例。

```bash
.venv/bin/python tools/score.py --predictions runs/round08-n5/predictions.json \
  --labels data/local/labels.json --output runs/round08-n5/metrics.json
```

ASA条件结果加 `--protocol conditional`，标签键改为关联ID。工具报告失败数、成功子集MAE和同子集perunit；存在失败时不把成功子集冒称固定全量成绩。perunit为绝对误差总和除以同一批参考量总和，不是逐图百分比的平均。

## 源码与历史结果

- `optimization_sol_20261102/mae_v2/c_stage_split_20261105/`：原混合C代码和原测试。
- `provenance/source_manifest.json`：28份冻结源码及额外依赖的哈希，全部保持原字节。
- `tools/`：新增跨设备安装、运行和评分外壳，调用原方法工厂。
- `runtime/release.json`：固定运行包下载地址、大小、逐文件校验。
- `results/baseline_summary.json`：原始未修正开发成绩摘要，不是本次新推理。

历史 `run.py`、`score.py` 是已完成实验的归档，有旧路径、旧结果导入、跨折标签中位数回退及校正代码。**新输入统一走 `tools/run_mixed_c.py`；失败保持空值，不用真实标签补预测。** 历史完整重放需要另行迁移旧私有缓存和开发标签，单纯重新调用模型不能保证重现原数值。

来源：[作者原仓库](https://github.com/Runz96/DietAI24)、[原混合C公开备份](https://github.com/THwo0t/DietAI24-reproduction/tree/e81f1f69c6a7d11bc9418ce92cce99689ddf1ccb/research_backups/c_mixed_20261107)。原作者 README 的许可说明保留于 `UPSTREAM_README.md`；数据各自条款见数据文档和 `THIRD_PARTY.md`，不把数据许可替换成代码许可。

本分支额外保存[第 9 批全 Argon（16/20）源码及回退说明](docs/ARGON_ROUND09_中文.md)。
