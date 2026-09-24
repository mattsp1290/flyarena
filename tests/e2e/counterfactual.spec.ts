import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import type { ExportDocument } from '../../src/lib/counterfactual/types';

test('real graph probe, replay, export, same-runtime repetition and CLI comparison and responsive layout', async ({page}, info) => {
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/#counterfactual');
  await expect(page.getByRole('status')).toHaveText('ready');
  await expect(page.getByLabel('Silence group')).toContainText('Bridge neurons · 800 neurons');
  await page.getByRole('button',{name:'Use quick probe settings'}).click();
  await page.getByRole('button',{name:'Fork & compare →'}).click();
  await expect(page.getByRole('status')).toHaveText('completed',{timeout:30_000});
  const downloadEvent=page.waitForEvent('download');
  await page.getByRole('button',{name:/Export counterfactual evidence/}).click();
  const download=await downloadEvent;const path=info.outputPath('counterfactual-browser.json');await download.saveAs(path);
  const doc:ExportDocument=JSON.parse(await readFile(path,'utf8'));
  expect(doc.evidence.results).toHaveLength(4);
  expect(doc.evidence.graph.neuronCount).toBe(1008);
  for(const result of doc.evidence.results) expect(result.branches.sham).toEqual(result.branches.baseline);
  const compared=JSON.parse(execFileSync(process.execPath,['--import','tsx','scripts/experiments/counterfactual.ts','--compare-numerical',path],{encoding:'utf8'}));
  expect(compared.matches).toBe(true);
  expect(compared.inexactLeaves).toBe(0); // Measured browser/Node differences stay beneath the declared noise floor.
  expect(compared.maxAbsoluteError).toBeLessThan(1e-13);
  console.log('Browser/Node comparison:',compared.status,compared.maxAbsoluteError);
  // Independent browser rerun must reproduce every deterministic field exactly.
  await page.getByRole('button',{name:'Fork & compare →'}).click();
  await expect(page.getByRole('status')).toHaveText('completed');
  const repeatEvent=page.waitForEvent('download');await page.getByRole('button',{name:/Export counterfactual evidence/}).click();
  const repeated=JSON.parse(await readFile((await (await repeatEvent).path())!,'utf8')) as ExportDocument;
  expect(repeated.evidence).toEqual(doc.evidence);
  await page.getByRole('button',{name:`Inspect seed ${doc.evidence.seeds[1]}`}).click();
  await page.getByLabel('Paired replay timeline').fill('15');
  const sample=doc.evidence.results[1];const frame=sample.branches.baseline.frames[15];
  const score=frame.scores.left.movementScore-sample.checkpoint.scores.left.movementScore;
  await expect(page.getByTestId('replay-score-0')).toHaveText(`Post-fork score ${score.toFixed(4)}`);
  await expect(page.getByRole('img',{name:`baseline world at tick ${frame.snapshot.tick}`})).toBeVisible();
  await page.getByLabel('Compare baseline with').selectOption('sham');
  await expect(page.getByTestId('replay-score-1')).toHaveText(`Post-fork score ${score.toFixed(4)}`);
  await page.getByRole('button',{name:'Back to fork'}).click();
  await expect(page.getByTestId('replay-score-0')).toHaveText('Post-fork score 0.0000');
  await page.getByLabel('Compare baseline with').selectOption('lesion');
  await page.getByLabel('Paired replay timeline').fill('30');
  await page.screenshot({path:info.outputPath('workbench-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.screenshot({path:info.outputPath('workbench-mobile.png'),fullPage:true});
  expect(errors).toEqual([]);
});

test('copies arena setup, cancels a probe on navigation, handles back/forward and reruns',async({page})=>{
  await page.goto('/');await expect(page.getByRole('status')).toHaveText('ready');
  await page.getByLabel('Seed',{exact:true}).fill('4242');
  await page.getByLabel('Seed',{exact:true}).press('Tab');
  await page.getByLabel('Left arm topology').selectOption('rewired');
  await expect(page.getByRole('button',{name:'Probe this setup (authored decoder)'})).toBeEnabled();
  await page.getByRole('button',{name:'Probe this setup (authored decoder)'}).click();
  await expect(page.getByRole('status')).toHaveText('ready');
  await expect(page.getByLabel('Base seed')).toHaveValue('4242');
  await expect(page.getByLabel('Graph topology')).toHaveValue('rewired');
  await page.getByLabel('Paired seeds').fill('16');await page.getByLabel('Future ticks').fill('300');
  await page.getByRole('button',{name:'Fork & compare →'}).click();
  await expect(page.getByRole('button',{name:'Cancel probe'})).toBeVisible();
  await page.getByRole('link',{name:'01 Arena',exact:true}).click();
  await expect(page.getByRole('status')).toHaveText('ready');
  await page.goBack();await expect(page.getByRole('status')).toHaveText('ready');
  await page.goForward();await expect(page.getByRole('status')).toHaveText('ready');
  await page.getByRole('link',{name:'02 Counterfactual workbench',exact:true}).click();
  await expect(page.getByRole('status')).toHaveText('ready');
  await page.getByRole('button',{name:'Fork & compare →'}).click();
  await page.getByRole('button',{name:'Cancel probe'}).click();
  await expect(page.getByRole('status')).toHaveText('cancelled');
  await page.getByRole('button',{name:'Use quick probe settings'}).click();
  await page.getByRole('button',{name:'Fork & compare →'}).click();
  await expect(page.getByRole('status')).toHaveText('completed');
});

test('graph corruption fails closed and retry restores the real worker',async({page})=>{
  await page.route('**/data/malecns-arena-v1.bin.gz',r=>r.fulfill({status:200,contentType:'application/octet-stream',body:Buffer.from([1,2,3])}));
  await page.goto('/#counterfactual');
  await expect(page.getByRole('alert')).toContainText(/length|sha256/);
  await expect(page.getByRole('button',{name:'Fork & compare →'})).toBeDisabled();
  await page.unroute('**/data/malecns-arena-v1.bin.gz');
  await page.getByRole('button',{name:'Retry graph loading'}).click();
  await expect(page.getByRole('status')).toHaveText('ready');
});

test('counterfactual computation creates no main-thread stall of 200ms',async({page})=>{
  await page.goto('/#counterfactual');await expect(page.getByRole('status')).toHaveText('ready');
  await page.evaluate(()=>{
    const durations:number[]=[];
    const observer=new PerformanceObserver(list=>durations.push(...list.getEntries().map(e=>e.duration)));
    observer.observe({entryTypes:['longtask']});
    Object.assign(window,{probeDurations:durations,probeObserver:observer});
  });
  await page.getByRole('button',{name:'Fork & compare →'}).click();
  await expect(page.getByRole('status')).toHaveText('completed',{timeout:30_000});
  const durations=await page.evaluate(()=>{
    const w=window as unknown as {probeDurations:number[];probeObserver:PerformanceObserver};
    const result=[...w.probeDurations,...w.probeObserver.takeRecords().map(e=>e.duration)];w.probeObserver.disconnect();return result;
  });
  console.log('Counterfactual main-thread long tasks:',durations);
  for(const duration of durations) expect(duration).toBeLessThan(200);
});
