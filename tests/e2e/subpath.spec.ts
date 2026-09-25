import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
test('all views work beneath /fly/ with root asset routes deliberately unavailable',async({page,request})=>{
  expect((await request.get('/data/malecns-arena-v1.manifest.json')).status()).toBe(404);
  const failed:string[]=[];const errors:string[]=[];
  page.on('response',r=>{if(r.status()>=400)failed.push(r.url());});page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/fly/');await expect(page.getByRole('status')).toHaveText('ready');
  const manifest=page.getByRole('link',{name:'Compiled artifact manifest (JSON)'});
  await expect(manifest).toHaveAttribute('href','/fly/data/malecns-arena-v1.manifest.json');
  // WP4 of `.agents/plans/null-explanation`: `loadNullExplanation` (chained
  // after `loadRewiringNull`, both fired from `initialize()` under this same
  // /fly/-prefixed `dataBaseUrl`) must resolve to its real "ok" note here,
  // not silently 404 against the deliberately-unavailable root /data/ path —
  // a base-path bug here would either leave the note absent forever or show
  // up in the `failed` 4xx/5xx log asserted at the end of this test.
  await expect(page.getByRole('heading',{name:/what biological's low score is associated with/i})).toBeVisible({timeout:20000});
  await expect(page.locator('.null-explanation-detail')).toContainText(/linear signal gain from right clearance input to thrust output/i);
  // WP3: `loadPositions` fetches the positions sidecar under this same
  // /fly/ base path — if it fell back to the (deliberately 404ing) root
  // /data/ path instead, the toggle would stay disabled forever and
  // `failed` below would catch the 404s.
  await expect(page.locator('section.activity button')).toBeEnabled({timeout:20000});
  // WP3: `loadLesionAtlas` also fetches under this same /fly/ base path —
  // the same deliberately-404ing root-/data/ tripwire as the positions
  // check above (a base-path bug here would either leave the lesion radio
  // disabled forever, or show up in the `failed` 4xx/5xx log asserted at
  // the end of this test).
  await page.locator('section.activity button').click();
  await expect(page.getByLabel('Neural activity at soma positions')).toBeVisible();
  await page.getByRole('radio',{name:/lesion effect \(offline\)/i}).click();
  // Scoped to the activity panel: "Computed (offline)" alone also matches
  // the always-visible ledger rows regardless of whether lesion mode ever
  // actually activated (round-2 dual review) — the scoped `.legend-bar`
  // below (the lesion-effect color mode's own extracted `LesionColorMode.svelte`
  // legend — see that component's doc comment) is what actually proves
  // activation either way, but scoping this check too keeps it honest on
  // its own.
  await expect(page.locator('section.activity').getByText(/Computed \(offline\)/).first()).toBeVisible({timeout:20000});
  await expect(page.locator('section.activity .legend-bar')).toBeVisible();
  await page.getByRole('button',{name:/^collapse$/i}).click();
  await page.getByRole('link',{name:'02 Counterfactual workbench',exact:true}).click();
  await expect(page.getByRole('status')).toHaveText('ready');
  await page.getByRole('button',{name:'Use quick probe settings'}).click();
  await page.getByRole('button',{name:'Fork & compare →'}).click();
  await expect(page.getByRole('status')).toHaveText('completed');
  await page.getByRole('link',{name:'03 Behavior atlas',exact:true}).click();
  await expect(page.getByRole('group',{name:'Select a discovered behavior'})).toBeVisible();
  await page.getByRole('button',{name:'Use quick probe settings'}).click();
  await page.getByRole('button',{name:'Fork & compare →'}).click();
  await expect(page.getByRole('status')).toHaveText('completed');
  await page.getByRole('link',{name:/04 DGX sandbox/}).click();
  await expect(page.getByLabel('Access token')).toBeVisible();
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.getByRole('link',{name:'01 Arena',exact:true}).click();
  await expect(page.getByRole('status')).toHaveText('ready');
  expect(errors).toEqual([]);expect(failed).toEqual([]);
});

test('real HTTP origin verifies graph hashes in both the arena and experiment Worker', async ({page}) => {
  await page.goto('http://flyarena.test:4174/fly/');
  expect(await page.evaluate(() => isSecureContext)).toBe(false);
  expect(await page.evaluate(() => crypto.subtle === undefined)).toBe(true);
  await expect(page.getByRole('status')).toHaveText('ready');
  await page.getByRole('link',{name:'02 Counterfactual workbench',exact:true}).click();
  await expect(page.getByRole('status')).toHaveText('ready');
  await page.getByRole('button',{name:'Use quick probe settings'}).click();
  await page.getByRole('button',{name:'Fork & compare →'}).click();
  await expect(page.getByRole('status')).toHaveText('completed');
  // Same-length corruption must still fail the hash check on this HTTP origin.
  await page.route('**/data/malecns-arena-v1.bin.gz', async route => {
    const bytes = await readFile('public/data/malecns-arena-v1.bin.gz');
    bytes[24] ^= 1;
    await route.fulfill({contentType:'application/octet-stream', body:bytes});
  });
  await page.reload();
  await expect(page.getByRole('alert')).toContainText('sha256');
  await expect(page.getByRole('button',{name:'Fork & compare →'})).toBeDisabled();
});
