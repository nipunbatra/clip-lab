async page=>{
 await page.route('**/*',route=>route.continue());
 await page.setViewportSize({width:1440,height:1000});
 await page.goto('https://nipunbatra.github.io/clip-lab/?revision=guidance#color-change',{waitUntil:'networkidle'});
 await page.waitForFunction(()=>window.clipLab?.current.id==='color-change');
 const result={title:await page.locator('#title').innerText(),change:await page.locator('#change-guide').innerText(),steps:await page.locator('#usage-steps li').count(),route:await page.locator('#guided-route a').count()};
 await page.click('#swap');result.editedNotice=await page.locator('.pair-edited').isVisible();await page.click('#reset');
 await page.evaluate(()=>location.hash='hat');await page.waitForFunction(()=>window.clipLab.current.id==='hat');result.hatTitle=await page.locator('#title').innerText();result.hatSubtitleHidden=!(await page.locator('#question').isVisible());
 await page.evaluate(()=>location.hash='color-change');await page.waitForFunction(()=>window.clipLab.current.id==='color-change');
 await page.screenshot({path:'output/verification/public-guidance.png',fullPage:true});
 await page.unroute('**/*');
 return result;
}
