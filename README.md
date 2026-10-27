# tiny-random · MoE Architecture Atlas

九个指定 MoE 模型的架构目录、正向数据流与可交互对比。基模是解释对象；Collection 中的小模型作为独立参考，不把 tiny 尺寸当作基模参数。

站点采用 **Python 标准库构建 + 原生 HTML/CSS/JS**：没有前端框架、数据库或线上模型 API。仓库保存配置、审核后的结构说明和少量执行记录，不保存权重。

## 页面组织

部署目标为 `https://nideyongbao.github.io/tiny-random/`。这只是预期地址，须在仓库权限与 Pages 配置就绪、workflow 成功后才会上线。

| 路径 | 内容 |
| --- | --- |
| `/tiny-random/` | 九模型目录；按家族、注意力类型和关键词筛选 |
| `/tiny-random/families/qwen/` 等 | 7 个家族子页面 |
| `/tiny-random/models/<model-id>/` | 单模型概览、逐层正向、注意力/专家、状态、原始配置与来源 |
| `/tiny-random/compare/` | 任选 2–4 个基模，按功能阶段比较；只看不同、prefill/decode、可分享链接 |
| `/tiny-random/execution.html` | GLM tiny 的真实 CPU/FP32 执行回放与数值诊断 |
| `/tiny-random/architecture.html` | 保留的早期 GLM 配置级概览 |

所有模型页都是实际生成的目录与 `index.html`，不依赖 SPA 重写或服务端路由。静态资源与配置使用相对路径，部署在项目子路径也能访问。

## 收录范围与分类

| 模型 | 正向结构要点 | tiny 关联 |
| --- | --- | --- |
| Qwen3 30B-A3B | 全 GQA；softmax MoE；标准残差 | 同家族235B来源，非目标精确缩小版 |
| Qwen3.5 35B-A3B | GDN×3 + gated GQA；gated shared expert | 同家族397B来源；expert/top-k不同 |
| Gemma 4 26B-A4B | sliding/global GQA；Dense与MoE并行；logit softcap | 同规格-it来源，tiny改变层周期 |
| DeepSeek V4-Pro | HCA/CSA；压缩历史；前三层HashMoE；mHC | Pro来源tiny，明确不混入bf16变体 |
| MiMo-V2.5-Pro | 纯文本；SWA/global；不等宽QK/V；局部sink | 本次集合快照未找到对应tiny |
| GLM-5.3-Flash | KDA/DSA；mHC；最终mean collapse | 有tiny真实轨迹，保留checkpoint数值异常 |
| MiniMax M3 | MSA按GQA group选历史块；top4+shared | 未找到精确M3，不替换为H3或M3.1 |
| Kimi K3 | KDA/gated MLA；AttnRes；Stable LatentMoE | 同名tiny含兼容patch；latent在tiny中实际扩维 |
| Qwen3.8-Flash-Next | GDN/QSA；gated HC；hashed PLE | 同名tiny；官方model_type=qwen4_exp |

资料核对日期：2026-10-27。每个模型档案包含固定的官方配置revision、实现引用、SHA256与tiny来源说明。集合成员快照在 `sources/collection.json`。

## GitHub Pages 发布

工作流文件：`.github/workflows/pages.yaml`。

参考了用户指定的 [InfraTech Pages workflow](https://github.com/CalvinXKY/InfraTech/blob/main/.github/workflows/pages.yml)，采用 checkout → 构建校验 → upload-pages-artifact → deploy-pages 的官方流程；构建与部署分成两个 job。

1. 将本目录作为仓库根目录推送到 `nideyongbao/tiny-random` 的 `main`。`.github/` 需要位于 Git 仓库根，不能额外套一层 `prototype/`。
2. 在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。此设置需要仓库管理权限；工作流没有使用 PAT 或自定义 secrets。
3. 推送相关文件到 `main`，或在 Actions 手动运行 **Build and deploy tiny-random architecture atlas**。
4. 等待 build/deploy 成功；以 deployment 输出的 `page_url` 为实际地址。

权限仅在部署 job 增加 `pages: write` 与 `id-token: write`。PR只构建和校验，不部署；同一ref的新任务可取消旧任务。构建时不下载上游模型、权重或Python推理依赖。

当前已配置本地 `origin`，但本次环境访问目标仓库仍返回认证错误，GitHub连接器也返回404，因此不能把本地完成等同于线上发布。仓库可见性、写权限与Pages启用状态仍须在GitHub侧就绪。

官方参考：[GitHub Pages 自定义 workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

## 本地构建与预览

需要 Python 3.10+；不需要安装任何Python包。Node仅用于JavaScript语法检查，可选浏览器QA需要Playwright。

```sh
python3 build_site.py
python3 scripts/check_site.py
python3 check_trace.py
node --check dist/assets/atlas.js
python3 -m http.server 8765 --directory dist
```

打开 `http://localhost:8765/`。本次还额外在 `/tiny-random/` 挂载路径做过浏览器验证，防止根路径可用但GitHub项目站点404。

可选浏览器检查：

```sh
# Playwright 已安装且浏览器可用时
ATLAS_URL=http://localhost:8765/ node scripts/browser_check.cjs
```

若Playwright安装在其他目录，可用 `PLAYWRIGHT_MODULE=/absolute/path/to/playwright` 指定。此浏览器检查不进入Pages发布依赖。

## 如何更新或增加模型

```text
catalog/models/<id>.json       唯一维护的结构档案
sources/models/<id>/           pinned base/tiny原始配置
site/atlas.js                 所有模型共用的界面与交互
site/atlas.css                共用样式
site/shell.html               共用页面外壳
build_site.py                 生成首页、家族、模型、对比与资源
scripts/check_site.py         来源/层序列/链接/核心语义校验
.github/workflows/pages.yaml  CI与Pages部署
```

更新顺序：

1. 用户指定关注的基模或想比较的关键问题。
2. Agent读取Collection、官方模型卡、配置和实现，固定revision。只下载配置；不把权重加入仓库。
3. 更新原始配置快照与SHA256，修改对应声明式JSON：层序列、模块说明、状态、功能对比签名、证据与限制。
4. 构建并检查。目录中的 `TRACE` 与结构页的源码推演不能混淆；缺少某型号tiny时如实记录，不替换为同家族其他型号。
5. 审阅Git diff后推送。Pages使用已经核对的本地资料构建，不在CI中自动抓取“latest”并猜结构。

新增一个模型通常只需新增数据档案与配置快照；家族目录由构建脚本生成。当前校验刻意锁定用户要求的9个模型与核心易错事实，扩展清单时同步更新覆盖断言。具体字段说明见 [catalog/README.md](catalog/README.md)。

`dist/` 是可直接托管的生成目录，保留旧版GLM回放资源。模型JSON和配置在生成时各拷贝一次到站点供浏览器读取，避免为每个页面复制一套模型代码。不要手工修改生成的模型页。

## 正确性与限制

- 新的九模型目录是配置与源码级说明，不声称执行过九个基模或捕获所有激活。长度P只用于张量形状推演。
- 层序列来自实际配置：MiMo的全局层不是简单七周期，Kimi的MLA列表是1-based，DeepSeek的末尾ratio0是MTP，GLM最后一层是KDA。
- 对比页按功能阶段对齐，不能把“相同功能签名”理解成参数、数值或性能等价。
- KV可见窗口、稀疏选择预算、物理缓存分配是不同概念；页面注明具体参考实现和后端差异。
- tiny参考可以改变专家数、共享分支、层序列、维度与实现。基模和tiny各自的尺寸及局限分开展示。
- 原有GLM回放保存21,207条ATen元信息和边界数值采样；专家FP8尺度异常导致cached/full检查失败，仍如实显示。复现和诊断详情见 [docs/GLM_TRACE.md](docs/GLM_TRACE.md)。

## 已验证

构建、来源哈希、9模型/7家族路由、真实层序列、tiny关联、15个功能对比维度、相对资源路径、GLM轨迹衔接、JavaScript语法。浏览器覆盖桌面与390px手机、九模型全部标签和节点、配置搜索、2–4模型选择与链接恢复、`/tiny-random/`项目路径及旧GLM回放。

GitHub云端workflow尚未运行；这与本地检查结果分开记录。
