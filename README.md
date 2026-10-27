# tiny-random · Architecture Lab

独立、无构建依赖的架构可视化原型。以 GLM-5.3-Flash 的 tiny 实例为主，附带 GLM-5.3 对照。

## 运行

```sh
python3 -m http.server 8765 --directory dist
```

打开 `http://localhost:8765`。`dist/` 可以作为静态网站部署，所有资源使用相对路径。

## 文件

- `dist/index.html`：页面布局。
- `dist/style.css`：响应式样式。
- `dist/app.js`：模型说明、SVG 绘图和交互。
- `sources/`：原始配置、Safetensors 文件头和来源清单。未下载或提交权重。
- `.openai/hosting.json`：本次独立私有预览的部署标识。

## 原型范围

- 模块选择与详情、主干层展开、尺寸切换、文本/图文路径高亮。
- GLM-5.3 与 GLM-5.3-Flash 的结构对照。
- 手机端架构画布支持横向滑动；模块详情显示在下方。
- 动画仅表示结构路径，不运行模型，不生成激活或专家选择结果。
- 内部模块图经过简化，不等同于完整运行时计算图。
- GLM-5.3-Flash 的 MTP 权重目录已检查；GLM-5.3 的 MTP 只依据配置标注。
- 本原型不包含自动同步和 GitHub Pages 工作流。当前远程源仓库由 Sites 管理。

## 验证

已使用 Chromium 检查 HTTP 响应、模块详情、尺寸切换、8 层展开、路径高亮、演示开关、对比、模型切换，以及 390px 手机布局。没有执行模型推理。
