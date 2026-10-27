# tiny-random · Execution Atlas

用真实执行记录解释一次生成的静态原型。入口是 GLM-5.3-Flash 的 tiny checkpoint，包含输入模板、token、8 层前向、KDA / DSA / mHC / FFN、缓存与专家路由、逐算子元信息，以及数值验证结果。右上角保留上一版配置级架构对比。

## 浏览

```sh
python3 -m http.server 8765 --directory dist
```

打开 `http://localhost:8765`。纯 HTML / CSS / JS，无构建步骤；资源为相对路径，`dist/` 可部署到 GitHub Pages。当前预览与远程源码由 Sites 管理，本仓库尚未配置 GitHub Pages 自动发布。

建议先查看输入模板，再依次点击 Prefill 和 Decode；在数据流中选层、选视图、点节点查看 tensor，再进入算子显微镜追溯输入来源。状态页可切换每个 query token 的 MoE 路由与 DSA 索引。

## 记录是真实的，但有明确边界

- checkpoint revision：`aeb642e89f4bb6b55b186ca440edff0acb9ee9f5`。
- 执行条件：CPU、FP8 checkpoint 反量化后全参数转 FP32、eager attention、PyTorch KDA/conv fallback；不是原生 FP8 GPU 后端。
- 固定 prompt `请用一句话介绍你自己。` 经原生模板得到 18 个 token；greedy 生成 4 个 token：`Recovered 分析 użyiles`。随机输出保持原样。
- 4 次前向，每次 216 个模块记录；prefill 9,306 个、每次 decode 3,967 个 ATen events，共 21,207 个。不同实现版本可能改变这些计数。
- 模块边界保留前两个 tensor 输入、前三个 tensor 输出的数值采样，以及所有 tensor ID。函数边界保留采样与 keyword inputs。参数值被排除。
- 数值采样为前 24 个标量、leading 8×12 tile、末行前 24 个标量。统计量覆盖全张量；不是完整 tensor dump。MoE 路由、DSA indices 和 head 0 attention 对本例完整保存。
- 所有被捕获的 ATen events 有输入输出 tensor ID、形状、dtype、常量、参数名和模块 scope。它们不是 GPU kernel / 硬件指令；一个 ATen event 可封装多个底层计算。
- `v` 是记录器对同一 tensor 对象再次返回的逻辑版本，不是 PyTorch storage version。跨对象共享 storage 的别名写入没有完整建模；不能将该视图当作内存别名分析器。
- Vision / MTP / backward 未执行。MTP 辅助 layer 8 的 26 个键被加载器忽略，实际主干层为 0–7。
- 固定示例离线回放；浏览器不加载权重，也不接受任意 prompt 推理。

## 上游 checkpoint 的数值问题

带记录器和不带记录器的相同 cached 路径，4 步 logits 完全一致。但 cached decode 与完整序列重算在后三步没有通过 `rtol=1e-3, atol=1e-4`，最大差分别约 `0.038385 / 0.035868 / 0.006497`；argmax 相同。

审计发现 routed experts 的 576 个尺度 tensor（9,216 元素，包含 MTP）全部为 1.0，而其他量化尺度约为 `0.000119–0.000225`。上游模型卡先 pack 拆分后的专家权重，再 pack 所有二维 `.weight`，覆盖第一次尺度。重复 pack 的最小实验也能复现。第一层 MoE 放大了此前极小的 FP32 差异。没有修改权重或猜测丢失的原尺度；页面以明确诊断标记保留实际执行。

证据：`dist/trace/numerical-audit.json`；复现尺度检查：`audit_checkpoint.py`；原始生成脚本见 [pinned 模型卡](https://huggingface.co/tiny-random/glm-5.3-flash/blob/aeb642e89f4bb6b55b186ca440edff0acb9ee9f5/README.md#L203)。

## 复现采集

将上述 revision 的 checkpoint 与 tokenizer 下载到仓库外的临时目录，然后在隔离 Python 环境安装 `requirements-trace.txt`。不要将权重放入仓库。

```sh
python capture_trace.py --model-dir /path/to/pinned/checkpoint
python audit_checkpoint.py --model-dir /path/to/pinned/checkpoint
python check_trace.py
```

`capture_trace.py` 使用上游模型实现，通用 module hooks + TorchDispatch 采集，另对四个非模块函数加只读边界记录；不重写模型计算。脚本目前针对这个已固定的示例，不是跨所有架构的自动适配器。更换模型时需要复核解释图及模型特有状态的采集。实现文件 SHA256、依赖版本、后端条件与加载报告一并记录。

## 文件与更新

- `dist/index.html` / `replay.css` / `replay.js`：通用回放界面、交互、该模型的解释图。
- `dist/trace/session.json`：阶段、模块边界、缓存、路由和输出记录。
- `dist/trace/ops-*.json`：按阶段懒加载的算子元信息。
- `dist/architecture.html` / `app.js` / `style.css`：上一版配置级结构对比。
- `sources/`：原始配置、Safetensors header 与来源清单；无权重。
- `capture_trace.py` / `audit_checkpoint.py` / `check_trace.py`：采集、诊断和数据一致性检查。

交互页面的数值始终来自记录文件；`TRACE` 节点关联真实模块／函数，`CODE` 节点只解释源码，不伪造中间值。增加模型时可以复用界面与数据格式，但需由 agent 审核模型特有图和状态语义。

## 已做验证

真实模型前向与独立 cached replay；完整序列数值诊断；数据记录的一致性断言；Chromium 验证所有图视图／节点、token 查看、算子筛选和来源追溯、专家选择、缓存阶段切换、诊断展示与 390px 手机布局。数值诊断失败项如实保留，不能记作全项通过。
