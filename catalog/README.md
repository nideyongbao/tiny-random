# 声明式模型档案

每个 `models/<id>.json` 是人工或agent阅读上游资料后形成的结构，不是任意config的自动适配规则。

| 字段 | 含义 |
| --- | --- |
| `id`, `order`, `name`, `family` | 稳定URL、展示顺序、名称与家族分类 |
| `category`, `modalities`, `summary`, `highlights` | 浏览目录所用的可读概括 |
| `sources` | source ID、上游固定链接、revision、原配置相对路径、SHA256 |
| `variants.base/tiny.dimensions` | 已核对的语义参数；不把缺失值推测成真实数值 |
| `variants.*.attention_schedule` | 每个0-based主干层使用的注意力模块键 |
| `variants.*.ffn_schedule` | Dense/MoE/HashMoE/LatentMoE/Parallel Dense+MoE |
| `attention` | 按模块键保存标题、语义、步骤、公式、注意事项和来源 |
| `moe`, `residual` | 同样的可读模块结构，供共用界面展开 |
| `states` | 跨步或本轮状态；基模/tiny形状分别保存，并说明增长与布局边界 |
| `input`, `norm`, `position`, `output`, `prefill`, `decode`, `extra`, `mtp` | 模型特有正向语义 |
| `comparison` | 功能阶段对比：可读标题、说明、来源与语义signature |
| `tiny_status`, `tiny_note`, `limitations`, `has_trace` | 证据覆盖与不能混用的边界 |

比较签名不是对内容的自动hash：只有经过审核、可认为在该行功能上相同的模型才使用相同signature。专家规模行应包含E/top-k/intermediate，避免“同为SwiGLU”掩盖规模差异。

`base` 始终对应用户指定的基模。`tiny` 可以是明确标注的同家族参考，但不能无说明地替换目标。缺少精确tiny时不创建虚构配置。

所有正向模块的数值如果不是运行记录，就必须保持公式、符号或“配置推演”的标签。不能从config推出真实token ID、路由选择、注意力值或性能结论。
