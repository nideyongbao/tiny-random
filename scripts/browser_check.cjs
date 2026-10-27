/* Optional browser QA: PLAYWRIGHT_MODULE=/absolute/path/to/playwright node scripts/browser_check.cjs */
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const base=process.env.ATLAS_URL||'http://127.0.0.1:8765/tiny-random/';
const shotDir=process.env.ATLAS_SCREENSHOTS||'/tmp/tiny-random-atlas-screenshots';
fs.mkdirSync(shotDir,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1536,height:1150}}),errors=[],failed=[];
 page.on('pageerror',e=>errors.push(String(e)));
 page.on('response',r=>{if(r.url().startsWith(base)&&r.status()>=400)failed.push([r.status(),r.url()]);});
 await page.goto(base,{waitUntil:'networkidle'});
 await page.waitForSelector('.model-card');
 assert.equal(await page.locator('.model-card').count(),9);
 await page.screenshot({path:shotDir+'/home.png',fullPage:true});
 await page.fill('#model-search','KDA');assert.equal(await page.locator('.model-card').count(),2);
 await page.fill('#model-search','');
 await page.locator('[data-filter="窗口混合"]').click();assert.equal(await page.locator('.model-card').count(),2);
 await page.locator('[data-filter="all"]').click();
 await page.locator('[data-pick="qwen3-30b-a3b"]').click();await page.locator('[data-pick="qwen3-5-35b-a3b"]').click();assert.equal(await page.locator('.comparison-dock').count(),1);
 const catalog=await (await page.request.get(base+'assets/catalog.json')).json();
 for(const m of catalog.models){
  await page.goto(base+'models/'+m.id+'/',{waitUntil:'networkidle'});
  assert((await page.locator('h1').textContent()).includes(m.name));
  assert.equal(await page.locator('.layer-cell').count(),m.variants.base.dimensions.layers);
  await page.locator('[data-tab="forward"]').click();
  await page.locator('[data-layer]').last().click();
  await page.locator('[data-phase="decode"]').click();
  assert((await page.locator('#model-view').textContent()).includes('本轮输入 T=1'));
  for(const n of await page.locator('[data-node]').all()) await n.click();
  if(m.id==='kimi-k3')await page.screenshot({path:shotDir+'/kimi-flow.png',fullPage:true});
  await page.locator('[data-tab="mechanisms"]').click();assert((await page.locator('.mechanism-chain').count())>=2);
  await page.locator('[data-tab="state"]').click();assert((await page.locator('.state-row').count())>=2);
  if(m.variants.tiny){
   await page.locator('[data-variant="tiny"]').click();
   await page.locator('[data-tab="overview"]').click();assert.equal(await page.locator('.layer-cell').count(),m.variants.tiny.dimensions.layers);
  }else assert(await page.locator('[data-variant="tiny"]').isDisabled());
  await page.locator('[data-tab="sources"]').click();await page.waitForSelector('.config-row');
  await page.fill('#config-search','num_hidden_layers');assert((await page.locator('.config-row').count())>=1);
  if(m.variants.tiny){await page.selectOption('#config-variant','tiny');await page.waitForTimeout(100);assert((await page.locator('.config-row').count())>=1);}
 }
 await page.goto(base+'families/qwen/',{waitUntil:'networkidle'});assert.equal(await page.locator('.model-card').count(),3);
 await page.goto(base+'compare/?models=qwen3-5-35b-a3b,glm-5.3-flash,kimi-k3',{waitUntil:'networkidle'});
 assert.equal(await page.locator('.compare-table thead th').count(),4);
 assert.equal(await page.locator('.compare-table tbody tr').count(),15);
 await page.check('#only-differences');assert.equal(await page.locator('.compare-table tbody tr:not(.different)').count(),0);
 await page.selectOption('#compare-phase','decode');assert.equal(await page.locator('#compare-prefill').count(),0);assert.equal(await page.locator('#compare-decode').count(),1);
 await page.screenshot({path:shotDir+'/compare.png',fullPage:true});
 await page.reload({waitUntil:'networkidle'});assert(await page.locator('#only-differences').isChecked());assert.equal(await page.locator('#compare-phase').inputValue(),'decode');
 await page.check('[data-compare-model="deepseek-v4-pro"]');await page.click('[data-compare-model="minimax-m3"]');assert.equal(await page.locator('[data-compare-model]:checked').count(),4);
 await page.goto(base+'execution.html',{waitUntil:'networkidle'});assert.equal(await page.locator('[data-token]').count(),18);
 await page.setViewportSize({width:390,height:844});
 for(const path of ['', 'models/qwen3-8-flash-next/', 'families/qwen/','compare/?models=qwen3-30b-a3b,gemma-4-26b-a4b']){
  await page.goto(base+path,{waitUntil:'networkidle'});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'mobile overflow '+path);
  if(path.startsWith('models/')){for(const tab of ['forward','mechanisms','state','sources']){await page.locator(`[data-tab="${tab}"]`).click();await page.waitForTimeout(100);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'mobile tab overflow '+tab);}}
 }
 await page.screenshot({path:shotDir+'/mobile-compare.png',fullPage:true});
 assert.deepEqual(errors,[]);assert.deepEqual(failed,[]);
 await browser.close();console.log('PASS: all 9 model routes, 7 tiny variants, family filter, all flow nodes/tabs, config search, 2–4 model comparison/deep links, preserved GLM trace, mobile layout, no JS/HTTP errors.');
})().catch(e=>{console.error(e);process.exit(1)});
