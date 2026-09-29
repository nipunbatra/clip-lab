async page=>{
 await page.setViewportSize({width:1440,height:1000});await page.goto('http://127.0.0.1:8770/#color-change',{waitUntil:'networkidle'});
 await page.click('#run');await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Live inference complete'),{},{timeout:180000,polling:100});
 const before=await page.evaluate(()=>window.lastClipResult);
 await page.click('#swap');await page.click('#run');await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Live inference complete'),{},{timeout:60000,polling:100});
 const after=await page.evaluate(()=>window.lastClipResult);
 const error=Math.max(...before.results.map(r=>Math.abs(r.cosine+after.results.find(x=>x.label===r.label).cosine)));
 return {before,after,maxSignReversalError:error,passed:error<1e-10};
}
