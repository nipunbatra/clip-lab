const fs=require('fs');const puppeteer=require(process.env.CLIP_PUPPETEER_PATH||'puppeteer');
(async()=>{const browser=await puppeteer.launch({headless:true});const page=await browser.newPage();await page.setViewport({width:1440,height:1000});
page.on('console',m=>console.log('CONSOLE',m.type(),m.text().slice(0,200)));page.on('pageerror',e=>console.log('ERROR',e.message));
await page.goto('http://127.0.0.1:8770/',{waitUntil:'networkidle0'});await page.screenshot({path:'output/verification/app-desktop.png',fullPage:true});
await page.click('#run');await page.waitForFunction(()=>window.lastClipResult?.runtime.startsWith('Live'),{timeout:300000});console.log('LIVE',JSON.stringify(await page.evaluate(()=>window.lastClipResult)));
await page.screenshot({path:'output/verification/app-live.png',fullPage:true});
fs.writeFileSync('output/verification/browser-live.json',JSON.stringify(await page.evaluate(()=>window.lastClipResult),null,2));await browser.close();})();
