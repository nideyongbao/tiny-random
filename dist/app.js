(() => {
  'use strict';
  const tinyRevision = 'aeb642e89f4bb6b55b186ca440edff0acb9ee9f5';
  const sources = {
    flash: `https://huggingface.co/tiny-random/glm-5.3-flash/blob/${tinyRevision}/config.json`,
    glm: 'https://huggingface.co/tiny-random/glm-5.3/blob/main/config.json',
    card: `https://huggingface.co/tiny-random/glm-5.3-flash/blob/${tinyRevision}/README.md`,
    code: 'https://github.com/huggingface/transformers/blob/a672eb5f474b8b30173e9f96eab66adf292da635/src/transformers/models/glm5_next/modeling_glm5_next.py'
  };
  const models = {
    flash: {name:'GLM-5.3-Flash',subtitle:'视觉分支 × 混合注意力 × 专家路由',layers:8,vision:true,attn:['KDA','KDA','KDA','DSA','KDA','KDA','KDA','DSA'],groups:[{label:'0–2',type:'KDA',ffn:'Dense',count:3},{label:'3',type:'DSA',ffn:'MoE',count:1},{label:'4–6',type:'KDA',ffn:'MoE',count:3},{label:'7',type:'DSA',ffn:'MoE',count:1}]},
    glm: {name:'GLM-5.3',subtitle:'语言主干 × 稀疏注意力 × 索引共享',layers:11,vision:false,attn:Array(11).fill('DSA'),groups:[{label:'0–2',type:'DSA',ffn:'Dense',count:3},{label:'3–10',type:'DSA',ffn:'MoE',count:8}]}
  };
  const $ = id => document.getElementById(id);
  const state = {model:'flash',selected:'backbone',dimensions:false,expanded:false,path:'all',comparing:false,compareSize:false,tracing:false};
  let traceTimer;
  const escape = value => String(value).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const mini = (rows,caption) => `<div class="module-diagram">${rows.map((row,i)=>`${i?'<div class="mini-arrow">↓</div>':''}<div class="mini-stage">${row.map(item=>`<span class="mini-box ${item[1]||''}">${item[0]}</span>`).join('')}</div>`).join('')}<div class="mini-caption">${caption}</div></div>`;
  function details(id) {
    const m=models[state.model];
    const entries={
      input:{type:'INPUT',name:'输入序列',overline:'TEXT / TOKENS',description:'文本经过 tokenizer 转换为 token 序列。图文输入还包含用于填入视觉特征的占位位置。',rows:[['词表行数','154,880'],['输出','token IDs']],diagram:mini([[['Text']],[['Tokenizer']],[['Token IDs']]],'序列示意，不执行 tokenizer')},
      image:{type:'MULTIMODAL INPUT',name:'图像与视频',overline:'IMAGE / VIDEO',description:'图像与视频输入经处理器组织为视觉 patch。视觉特征由同一视觉编码分支提取，随后填入输入序列中的对应位置。',rows:[['输入通道','3'],['空间 patch','14 × 14'],['时间 patch','2']],diagram:mini([[['Image'],['Video']],[['Processor']],[['Visual patches','']]],'输入预处理示意')},
      embedding:{type:'EMBEDDING',name:'Token embedding',overline:'LOOKUP',description:'将 token IDs 映射到 tiny 的 128 维表示。此实例的输入 embedding 与输出 LM head 不共享权重。',rows:[['权重形状','[154880, 128]'],['存储精度','BF16'],['权重共享','false']],diagram:mini([[['Token IDs']],[['Embedding lookup']],[['D = 128']]],'以 tiny 实例的尺寸显示')},
      vision:{type:'VISION ENCODER',name:'视觉编码分支',overline:'2 VISION BLOCKS',description:'从时空 patch 提取特征。视觉编码后先归一化并下采样，再进入 merger；这里展示的是 forward 的执行顺序。',rows:[['编码层','2'],['hidden / heads','128 / 4'],['空间合并','2 × 2'],['输出宽度','128']],diagram:mini([[['Patch embed']],[['Vision block × 2','blue']],[['RMSNorm']],[['Downsample 2×2']],[['Merger']]],'实际执行顺序；重复层已折叠')},
      fusion:{type:'FEATURE INSERTION',name:'视觉特征填入',overline:'FUSION',description:'用视觉特征替换 token embedding 序列中对应的占位位置，组成语言主干的输入。它不是额外的 cross-attention 层。',rows:[['连接方式','placeholder replacement'],['输入主干宽度','128']],diagram:mini([[['Text embedding'],['Visual features','blue']],[['masked_scatter']],[['Input embeddings']]],'依据多模态 forward')},
      backbone:{type:'TEXT BACKBONE',name:'语言主干',overline:`${m.layers} DECODER LAYERS`,description:m.vision?'两组「3 个 KDA + 1 个 DSA」。前 3 层使用 Dense FFN，后 5 层使用 MoE。mHC 在注意力和前馈子层周围连接四条状态流。':'11 个 DSA 主干层。前 3 层使用 Dense FFN，其余层使用 MoE。索引器包含 full 与 shared 配置，需与注意力层本身区分。',rows:[['主干层数',m.layers],['残差宽度','128'],['前馈分布',m.vision?'3 Dense + 5 MoE':'3 Dense + 8 MoE'],['附加预测配置','1 MTP']],diagram:m.vision?mini([[['KDA × 3','purple'],['DSA','blue']],[['KDA × 3','purple'],['DSA','blue']]],'注意力模式；FFN 分布另行标注'):mini([[['DSA × 11','blue']],[['Full / shared indexers']]],'索引计算与共享由配置定义')},
      kda:{type:'LINEAR ATTENTION',name:'KDA · 线性注意力',overline:'LAYERS 0–2, 4–6',description:'Q/K/V 投影后进入短卷积与门控递归计算。递归状态与普通 attention KV cache 不同；图中折叠了状态更新内部运算。',rows:[['heads × head dim','64 × 128'],['Q/K/V 投影','128 → 8192'],['短卷积 kernel','4'],['权重存储','BF16']],diagram:mini([[['Input · 128']],[['Q','purple'],['K','purple'],['V','purple']],[['Short conv · k=4']],[['Gated recurrence','purple']],[['Gated norm → O proj']]],'机制示意，不展示实际状态数值')},
      dsa:{type:'SPARSE ATTENTION',name:'DSA · 稀疏注意力',overline:m.vision?'LAYERS 3, 7':'LAYERS 0–10',description:'索引器产生位置选择，注意力主路径通过压缩表示与投影计算输出。索引器的旋转位置处理与主注意力的位置配置需分别看待。',rows:[['attention heads','64'],['KV latent rank','256'],['index heads × dim','32 × 128'],['index top-k','128'],['QK RoPE width',m.vision?'0':'32']],diagram:mini([[['Hidden states']],[['Q / KV projections','blue'],['Indexer','blue']],[['Selected positions']],[['Sparse attention','blue']],[['Output projection']]],'选择路径与注意力路径的简化示意')},
      moe:{type:'FEED FORWARD',name:'MoE · 专家路由',overline:'ROUTED + SHARED',description:'路由器为每个 token 选择专家，选中专家的输出与共享专家分支组合。这里展示的是配置拓扑，不演示虚构的路由结果。',rows:[['路由专家数','32'],['每 token 选择','8'],['共享专家数','1'],['expert intermediate','128']],diagram:'<div class="module-diagram"><div class="mini-stage"><span class="mini-box">Hidden states</span></div><div class="mini-arrow">↙ &nbsp; ↘</div><div class="mini-branches"><div><span class="mini-box amber">Router</span><span class="mini-arrow">↓</span><span class="mini-box amber">8 of 32 experts</span></div><div><span class="mini-box amber">Shared expert</span><span class="mini-arrow">↓</span></div></div><div class="mini-arrow">↘ &nbsp; ↙</div><div class="mini-stage"><span class="mini-box">Combine outputs</span></div><div class="mini-caption">共享分支独立于路由选择</div></div>'},
      output:{type:'OUTPUT',name:'归一化与输出',overline:'TOKEN LOGITS',description:m.vision?'先由 HyperHead 汇合 mHC 状态流，再执行 RMSNorm 与 LM head，得到词表上的 logits。随机权重输出不代表语言能力。':'语言主干输出经最终归一化与 LM head 投影为词表 logits。随机权重输出不代表语言能力。',rows:[['LM head weight','[154880, 128]'],['词表输出','154,880'],['与 embedding 共享','false']],diagram:mini(m.vision?[[['mHC stream merge']],[['RMSNorm']],[['LM head']],[['Logits']]]:[[['RMSNorm']],[['LM head']],[['Logits']]],'输出张量路径，不进行模型推理')},
      mtp:{type:'AUXILIARY PREDICTION',name:'MTP · 辅助预测',overline:'OUTSIDE THE MAIN STACK',description:m.vision?'checkpoint 在 layers.8 下保存附加的 DSA/MoE 预测权重。它不计入 8 个主干层；是否执行以及怎样调用取决于运行框架与预测配置。':'配置声明一个 next-token prediction 层。本原型仅标注该配置，不将其视为主干的第 12 层，也未核验此模型的 MTP 权重目录。',rows:[['主干之外','1 层'],['证据范围',m.vision?'config + tensor header':'config only'],['数据流演示','不包含 MTP']],diagram:mini([[['Hidden / token features']],[['Auxiliary predictor','amber']],[['Draft token logits']]],'辅助预测概念图；非主干串行层')}
    };
    return entries[id]||entries.backbone;
  }
  function renderInspector() {
    const d=details(state.selected);
    $('selection-type').textContent=d.type;
    $('inspector-content').innerHTML=`<div class="inspector-body"><div class="module-overline">${d.overline}</div><h2>${d.name}</h2><p class="module-description">${d.description}</p>${d.diagram}<div class="detail-label">TINY CONFIGURATION</div>${d.rows.map(([k,v])=>`<div class="property"><span>${k}</span><span>${escape(v)}</span></div>`).join('')}<div class="detail-label">SOURCE EVIDENCE</div><a class="source-link" href="${sources[state.model]}" target="_blank" rel="noreferrer"><span>config.json</span><span>↗</span></a>${state.model==='flash'?`<a class="source-link" href="${sources.code}" target="_blank" rel="noreferrer"><span>模型实现 · Transformers</span><span>↗</span></a><a class="source-link" href="${sources.card}" target="_blank" rel="noreferrer"><span>构建说明与权重结构</span><span>↗</span></a>`:''}<p class="module-footnote">结构来自配置与实现阅读。路径高亮表示计算顺序，不运行权重，也不展示实际激活。</p></div>`;
  }
  function node(id,x,y,w,h,title,sub,cls='',path='shared') {
    const faded=(state.path==='text'&&path==='vision')?' faded':'';
    return `<g class="node ${cls}${state.selected===id?' selected':''}${faded}" data-node="${id}" role="button" tabindex="0" aria-label="查看${escape(title)}" aria-pressed="${state.selected===id}"><rect class="node-box" x="${x}" y="${y}" width="${w}" height="${h}" rx="9"/><text class="node-title" x="${x+w/2}" y="${y+h/2-3}" text-anchor="middle">${escape(title)}</text><text class="node-sub" x="${x+w/2}" y="${y+h/2+17}" text-anchor="middle">${escape(sub)}</text></g>`;
  }
  function edge(path,kind='shared') {
    const faded=state.path==='text'&&kind==='vision';
    const highlighted=state.path!=='all'&&!faded;
    return `<path class="edge${faded?' faded':''}${highlighted?' active-path':''}${state.tracing&&!faded?' running':''}" d="${path}"/>`;
  }
  function renderGraph() {
    const m=models[state.model];
    const groups=state.expanded?m.attn.map((type,i)=>({label:String(i),type,ffn:i<3?'Dense':'MoE',count:1})):m.groups;
    const start=m.vision?358:235,rowStart=start+63,rowStep=51,groupEnd=rowStart+groups.length*rowStep+24,outY=groupEnd+49;
    const dims=state.dimensions;
    let svg=`<svg viewBox="0 0 720 ${outY+114}" role="img" aria-labelledby="architecture-title architecture-description"><title id="architecture-title">${m.name} tiny 架构图</title><desc id="architecture-description">${m.vision?'视觉分支与文本 embedding 汇合，经过 8 个混合注意力层后输出。MTP 单独展示。':'文本经过 embedding、11 个 DSA 层和输出头。'}点击模块或使用键盘 Enter 查看详情。</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 1 1 L 9 5 L 1 9 z" fill="#a9bbb8"/></marker></defs>`;
    if(m.vision) {
      svg+=edge('M 180 86 V 122','text')+edge('M 540 86 V 122','vision')+edge('M 180 188 V 231 Q 180 248 197 248 H 330 Q 346 248 346 264 V 281','text')+edge('M 540 204 V 231 Q 540 248 523 248 H 390 Q 374 248 374 264 V 281','vision');
      svg+=node('input',80,30,200,56,'文本输入','TEXT TOKENS','','text');
      svg+=node('image',440,30,200,56,'图像 / 视频','VISUAL INPUT','vision','vision');
      svg+=node('embedding',80,122,200,66,'Token embedding',dims?'154880 × 128':'TOKEN → REPRESENTATION','','text');
      svg+=node('vision',440,122,200,82,'Vision encoder × 2',dims?'D=128 · heads=4':'PATCH → BLOCKS → MERGER','vision','vision');
      svg+=node('fusion',244,281,232,58,state.path==='text'?'文本输入表示':'视觉特征填入',dims?'input width = 128':state.path==='text'?'TEXT-ONLY SEQUENCE':'PLACEHOLDER REPLACEMENT');
      svg+=edge(`M 360 339 V ${start+10}`);
    } else {
      svg+=node('input',250,28,220,56,'文本输入','TEXT TOKENS');
      svg+=edge('M 360 84 V 122');
      svg+=node('embedding',250,122,220,66,'Token embedding',dims?'154880 × 128':'TOKEN → REPRESENTATION');
      svg+=edge(`M 360 188 V ${start+10}`);
    }
    svg+=`<rect class="group-bg" x="116" y="${start}" width="488" height="${groupEnd-start}" rx="14"/><g class="node${state.selected==='backbone'?' selected':''}" data-node="backbone" role="button" tabindex="0" aria-label="查看语言主干详情"><text class="section-label" x="138" y="${start+28}">TEXT BACKBONE</text><text class="node-tag" x="582" y="${start+28}" text-anchor="end">${m.layers} LAYERS${m.vision?' · mHC × 4':''}</text></g>`;
    groups.forEach((g,i)=>{
      const y=rowStart+i*rowStep;
      const type=g.type.toLowerCase();
      svg+=`<text class="row-number" x="139" y="${y+24}">${g.label}</text>`;
      if(i)svg+=edge(`M 580 ${y-rowStep+19.5} H 595 V ${y-6} H 173 V ${y+19.5} H 184`);
      svg+=edge(`M 431 ${y+19.5} H 441`);
      svg+=`<g class="node ${type}${state.selected===type?' selected':''}" data-node="${type}" role="button" tabindex="0" aria-label="层 ${g.label} ${g.type}，${g.ffn}"><rect class="node-box" x="184" y="${y}" width="247" height="39" rx="7"/><text class="node-title" x="198" y="${y+25}">${g.type} ${g.type==='KDA'?'linear':'sparse'} attention</text><text class="node-sub" x="418" y="${y+25}" text-anchor="end">${g.count>1?'× '+g.count:''}</text></g>`;
      svg+=`<g class="node ${g.ffn==='MoE'?'moe':''}${state.selected==='moe'&&g.ffn==='MoE'?' selected':''}" data-node="${g.ffn==='MoE'?'moe':'backbone'}" role="button" tabindex="0" aria-label="${g.ffn} 前馈模块"><rect class="node-box" x="441" y="${y}" width="139" height="39" rx="7"/><text class="badge-text" x="510" y="${y+25}" text-anchor="middle">${g.ffn}${dims&&g.ffn==='MoE'?' · 8 / 32':''}</text></g>`;
    });
    svg+=`<text class="node-tag" x="360" y="${groupEnd-10}" text-anchor="middle">${dims?'residual width = 128':m.vision?'ATTENTION + FFN / mHC CONNECTIONS':'ATTENTION + FFN / INDEXER SHARING'}</text>`;
    svg+=edge(`M 360 ${groupEnd} V ${outY}`);
    svg+=node('output',250,outY,220,66,'Norm → LM head',dims?'128 → 154880':'TOKEN LOGITS');
    const mtpY=start+65;
    svg+=node('mtp',618,mtpY,92,94,'MTP',m.vision?'layers.8':'config: 1','mtp');
    svg+=`<text class="node-tag" x="664" y="${mtpY+114}" text-anchor="middle">主干之外</text></svg>`;
    $('canvas').innerHTML=svg;
    $('canvas').querySelectorAll('[data-node]').forEach(el=>{
      const select=()=>{state.selected=el.dataset.node;renderGraph();renderInspector();};
      el.addEventListener('click',select);
      el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();const id=el.dataset.node;select();$('canvas').querySelector(`[data-node="${id}"]`)?.focus();}});
    });
    $('trace-button').textContent=state.tracing?'□ 停止演示':'▷ 路径演示';
  }
  function renderComparison() {
    $('compare-grid').innerHTML=['glm','flash'].map(id=>{
      const m=models[id],s=state.compareSize;
      return `<article class="compare-card"><div class="compare-card-head"><h3>${m.name}</h3><span>TINY INSTANCE</span></div><div class="compare-section"><div class="compare-label"><span>01 / INPUT & VISION</span><span>${m.vision?'新增分支':'语言输入'}</span></div><div class="compare-visual"><span class="compare-block">Text</span><span class="compare-block ${m.vision?'vision':'empty'}">${m.vision?'Vision encoder × 2':'无对应视觉分支'}</span></div>${s?`<div class="compare-params">${m.vision?'vision D=128 · 4 heads':'embedding D=128'}</div>`:''}</div><div class="compare-section"><div class="compare-label"><span>02 / ATTENTION</span><span>${m.layers} LAYERS</span></div><div class="layer-grid">${m.attn.map((type,i)=>`<div class="layer-cell ${type==='KDA'?'kda':''}" aria-label="层 ${i}: ${type}">${type==='KDA'?'K':'D'}<small>${i}</small></div>`).join('')}</div><div class="layer-subtitle">${m.vision?'K = KDA · D = DSA；每 4 层包含 3K + 1D':'D = DSA；索引器另有 full / shared 安排'}</div>${s?'<div class="compare-params">hidden=128 · attention heads=64</div>':''}</div><div class="compare-section"><div class="compare-label"><span>03 / FEED FORWARD</span><span>共同的专家配置</span></div><div class="ffn-bar"><span>Dense × 3</span><span>MoE × ${m.layers-3}</span></div><div class="layer-subtitle">32 routed · top-8 · 1 shared</div>${s?'<div class="compare-params">expert intermediate=128</div>':''}</div><div class="compare-section"><div class="compare-label"><span>04 / EVIDENCE</span></div><a class="source-link" href="${sources[id]}" target="_blank" rel="noreferrer"><span>查看原始配置</span><span>↗</span></a></div></article>`;
    }).join('');
  }
  function stopTrace(){clearTimeout(traceTimer);state.tracing=false;}
  function render() {
    const m=models[state.model];
    $('model-title').textContent=state.comparing?'GLM · 结构对照':m.name;
    $('model-subtitle').textContent=state.comparing?'共同布局，独立层序列。先看机制，再看尺寸。':m.subtitle;
    $('config-link').href=sources[state.model];
    $('revision-label').textContent=state.model==='flash'?'tiny revision · aeb642e8':'GLM-5.3 · 配置观察快照';
    $('meta-strip').innerHTML=`<div class="meta-item">主干 <strong>${m.layers} layers</strong></div><div class="meta-item">专家 <strong>32 / top-8</strong></div><div class="meta-item">hidden <strong>128</strong></div><div class="meta-item">${m.vision?'输入 <strong>text · image · video</strong>':'模型 <strong>glm_moe_dsa</strong>'}</div>`;
    $('meta-strip').hidden=state.comparing;
    $('explorer').hidden=state.comparing;
    $('comparison').hidden=!state.comparing;
    $('compare-button').classList.toggle('is-on',state.comparing);
    $('compare-button').innerHTML=state.comparing?'← <span>返回架构</span>':`⇄ <span>对比 ${state.model==='flash'?'GLM-5.3':'Flash'}</span>`;
    $('expand-button').textContent=state.expanded?'折叠重复层':`展开 ${m.layers} 层`;
    $('expand-button').setAttribute('aria-pressed',state.expanded);
    document.querySelectorAll('.model-choice').forEach(b=>{b.classList.toggle('selected',b.dataset.model===state.model);b.setAttribute('aria-pressed',b.dataset.model===state.model);});
    document.querySelectorAll('[data-path]').forEach(b=>{b.disabled=b.dataset.path==='vision'&&!m.vision;b.classList.toggle('active',b.dataset.path===state.path);b.setAttribute('aria-pressed',b.dataset.path===state.path);});
    document.querySelector('.sidebar-source').href=`https://huggingface.co/tiny-random/${state.model==='flash'?'glm-5.3-flash':'glm-5.3'}`;
    renderGraph();renderInspector();renderComparison();
  }
  document.querySelectorAll('.model-choice').forEach(b=>b.addEventListener('click',()=>{stopTrace();state.model=b.dataset.model;state.selected='backbone';state.path='all';state.expanded=false;state.comparing=false;render();}));
  document.querySelectorAll('[data-path]').forEach(b=>b.addEventListener('click',()=>{state.path=b.dataset.path;render();}));
  $('dimension-toggle').addEventListener('change',e=>{state.dimensions=e.target.checked;renderGraph();});
  $('expand-button').addEventListener('click',()=>{state.expanded=!state.expanded;render();});
  $('compare-button').addEventListener('click',()=>{stopTrace();state.comparing=!state.comparing;render();});
  $('compare-size').addEventListener('change',e=>{state.compareSize=e.target.checked;renderComparison();});
  $('trace-button').addEventListener('click',()=>{if(state.tracing)stopTrace();else{state.tracing=true;traceTimer=setTimeout(()=>{stopTrace();renderGraph();},10000);}renderGraph();});
  window.addEventListener('pagehide',stopTrace);
  render();
})();
