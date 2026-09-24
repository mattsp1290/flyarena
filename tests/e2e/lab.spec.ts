import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";

test("live lab: complete, inspect, export and cancel", async ({
  page,
}, testInfo) => {
  test.skip(
    !process.env.LAB_TEST_TOKEN,
    "Launch scripts/lab.sh and set LAB_TEST_TOKEN for live backend integration",
  );
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/#dgx");
  await page
    .getByLabel("Backend URL")
    .fill(process.env.LAB_TEST_URL ?? "http://127.0.0.1:8765");
  await page.getByLabel("Access token").fill(process.env.LAB_TEST_TOKEN!);
  await page.getByLabel("Device", { exact: true }).selectOption(process.env.LAB_TEST_DEVICE ?? "cpu");
  await page
    .getByRole("button", { name: "Use quick validation settings" })
    .click();
  await page.getByRole("button", { name: /Train & probe/ }).click();
  await expect(
    page.getByRole("heading", { name: "Circuit lesion atlas" }),
  ).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "Inspect G4", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Inspect G4", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.getByLabel("Replay timeline").fill("30");
  await expect(
    page.getByRole("img", { name: "Replay of G4 at tick 30" }),
  ).toBeVisible();
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("button", { name: /Export evidence/ }).click();
  const download = await downloadEvent;
  const path = await download.path();
  const result = JSON.parse(await readFile(path!, "utf8"));
  expect(result.arms).toHaveLength(11);
  expect(result.arms[0].scores).toEqual(result.arms[1].scores);
  expect(JSON.stringify(result)).not.toContain(process.env.LAB_TEST_TOKEN!);
  await page.screenshot({
    path: testInfo.outputPath("lab-desktop.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: testInfo.outputPath("lab-mobile.png"),
    fullPage: true,
  });
  await page.getByLabel("Generations", { exact: true }).fill("40");
  await page.getByLabel("Episode ticks").fill("600");
  await page.getByRole("button", { name: /Train & probe/ }).click();
  await expect(
    page.getByRole("button", { name: "Cancel experiment" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Cancel experiment" }).click();
  await expect(page.getByRole("status")).toContainText("cancelled", {
    timeout: 30_000,
  });
  expect(errors).toEqual([]);
});

test('sandbox navigation retains the same GPU job and exports a result completed while hidden', async ({page}) => {
  test.skip(!process.env.LAB_TEST_TOKEN, 'Needs a separately started sandbox backend');
  test.setTimeout(120_000);
  let submissions=0;
  page.on('request',request=>{if(request.method()==='POST'&&request.url().endsWith('/api/v1/jobs'))submissions++;});
  await page.goto('/#dgx');
  await page.getByLabel('Backend URL').fill(process.env.LAB_TEST_URL ?? 'http://127.0.0.1:8765');
  await page.getByLabel('Access token').fill(process.env.LAB_TEST_TOKEN!);
  await page.getByLabel('Device',{exact:true}).selectOption(process.env.LAB_TEST_DEVICE ?? 'cpu');
  await page.getByLabel('Generations',{exact:true}).fill('40');
  await page.getByLabel('Episode ticks').fill('600');
  await page.getByRole('button',{name:/Train & probe/}).click();
  await expect(page.getByTestId('lab-job-id')).toBeVisible();
  const identifier=await page.getByTestId('lab-job-id').innerText();
  await expect(page.getByRole('button',{name:'Cancel experiment'})).toBeVisible();
  await page.getByRole('link',{name:'01 Arena',exact:true}).click();
  await expect(page.getByRole('status')).toHaveText('ready');
  await page.getByRole('link',{name:/04 DGX sandbox/}).click();
  await expect(page.getByTestId('lab-job-id')).toHaveText(identifier);
  expect(submissions).toBe(1);
  await page.getByRole('button',{name:'Cancel experiment'}).click();
  await expect(page.getByRole('status')).toContainText('cancelled');
  await page.getByRole('button',{name:'Use quick validation settings'}).click();
  await page.getByRole('button',{name:/Train & probe/}).click();
  await expect(page.getByTestId('lab-job-id')).not.toHaveText(identifier);
  const second=await page.getByTestId('lab-job-id').innerText();
  await page.getByRole('link',{name:'02 Counterfactual workbench',exact:true}).click();
  await expect(page.getByRole('status')).toHaveText('ready');
  // The hidden instance keeps serial polling, so completion does not discard evidence.
  await expect(page.locator('.dgx-sandbox [role="status"]')).toContainText('completed',{timeout:60_000});
  await page.getByRole('link',{name:/04 DGX sandbox/}).click();
  await expect(page.getByTestId('lab-job-id')).toHaveText(second);
  expect(submissions).toBe(2);
  const event=page.waitForEvent('download');await page.getByRole('button',{name:/Export evidence/}).click();
  const downloaded=JSON.parse(await readFile((await (await event).path())!,'utf8'));
  expect(downloaded.arms).toHaveLength(11);
  expect(JSON.stringify(downloaded)).not.toContain(process.env.LAB_TEST_TOKEN!);
});
