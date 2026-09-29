async page => {
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const failures=[];const check=(v,s)=>{if(!v)failures.push(s)};
 await page.setViewportSize({width:1440,height:1050});
 await page.goto('http://127.0.0.1:8770/#hat',{waitUntil:'networkidle'});
 check(await page.locator('#title').innerText()==='What if we change?','hat title');
 check(!(await page.locator('#question').isVisible()),'hat subtitle hidden');
 await page.locator('.usage summary').click();
 check(await page.locator('#guided-route a').count()===8,'curated eight');
 const ids=await page.evaluate(()=>window.clipLab.experiments.map(x=>x.id));
 for(const id of ids){
  await page.evaluate(id=>location.hash=id,id);await page.waitForFunction(id=>window.clipLab.current.id===id,id);
  check(await page.locator('#usage-steps li').count()===3,id+' usage steps');
  check((await page.locator('#mechanism').textContent()).length>100,id+' mechanism');
  const difference=await page.evaluate(()=>window.clipLab.current.mode==='difference');
  check(await page.locator('#change-guide').isVisible()===difference,id+' change description');
  await page.click('#recorded');await page.waitForFunction(id=>window.lastClipResult?.experiment.id===id,id);
  if(difference){const original=await page.evaluate(()=>[window.clipLab.current.image,window.clipLab.current.second]);await page.click('#swap');const swapped=await page.evaluate(()=>[window.clipLab.current.image,window.clipLab.current.second]);check(swapped[0]===original[1]&&swapped[1]===original[0],id+' swap');check(await page.locator('.pair-edited').isVisible(),id+' edited pair notice');check((await page.locator('#results').textContent())==='',id+' cleared results');await page.click('#recorded');await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('original example'));await page.click('#reset');}
 }
 await page.evaluate(()=>location.hash='color-change');await page.waitForFunction(()=>window.clipLab.current.id==='color-change');
 check(await page.locator('#question').isVisible(),'subtitle restored');
 await page.locator('#under-hood summary').click();await page.click('#recorded');await page.screenshot({path:'output/verification/guidance-color-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});
 for(const id of ids){await page.evaluate(id=>location.hash=id,id);await page.waitForFunction(id=>window.clipLab.current.id===id,id);check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),id+' mobile width');}
 await page.screenshot({path:'output/verification/guidance-style-phone.png',fullPage:true});
 return {examples:ids.length,curated:8,failures,errors};
}
