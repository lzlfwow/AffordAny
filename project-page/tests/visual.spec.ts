import { expect, test } from "@playwright/test";

test("renders the complete project page without cropping key figures", async ({ page }, testInfo) => {
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "AffordAny", exact: true })).toBeVisible();
  await expect(page.locator(".hero-gallery .case-media img")).toHaveCount(6);
  await expect(page.locator('.publication-links a[href="https://arxiv.org/abs/2608.20720"]')).toBeVisible();

  const explorer = page.locator("#explorer");
  const explorerImage = explorer.locator(".sample-image-wrap img");

  await explorer.getByLabel("Evaluation split").selectOption("Unseen instruction");
  await expect(explorer.locator(".sample-selector button")).toHaveCount(3);
  for (const sampleName of ["Umbrella", "Fork"]) {
    await explorer.getByRole("button", { name: new RegExp(sampleName) }).click();
    await expect(explorer.locator(".sample-meta h3")).toHaveText(sampleName);
    for (const mode of ["Source image", "3D reconstruction", "Part annotation", "AffordAny"]) {
      await explorer.getByRole("button", { name: mode, exact: true }).click();
      await expect.poll(() => explorerImage.evaluate((image: HTMLImageElement) => image.naturalWidth)).toBeGreaterThan(0);
    }
  }

  await explorer.getByLabel("Evaluation split").selectOption("Unseen category");
  await expect(explorer.locator(".sample-selector button")).toHaveCount(2);
  await explorer.getByRole("button", { name: /Drinking glass/ }).click();
  await expect(explorer.locator(".sample-meta")).toContainText("holding it around its center.");

  await expect.poll(() => explorerImage.evaluate((image: HTMLImageElement) => image.naturalWidth)).toBeGreaterThan(0);
  const imageFit = await explorerImage.evaluate((image: HTMLImageElement) => {
    const renderedRatio = image.getBoundingClientRect().width / image.getBoundingClientRect().height;
    const naturalRatio = image.naturalWidth / image.naturalHeight;
    return Math.abs(renderedRatio - naturalRatio);
  });
  expect(imageFit).toBeLessThan(0.02);
  for (const selector of ['img[src$="teaser.svg"]', 'img[src$="comparison.svg"]']) {
    const image = page.locator(selector);
    await image.scrollIntoViewIfNeeded();
    await expect.poll(() => image.evaluate((element: HTMLImageElement) => element.naturalWidth)).toBeGreaterThan(0);
  }

  const citation = page.locator("#citation");
  await expect(citation.locator("code")).toContainText("@misc{wu2026affordanyopenworld3daffordance,");
  await expect(citation.locator("code")).toContainText("eprint={2608.20720}");
  await expect(citation.locator("code")).toContainText("primaryClass={cs.CV}");

  expect(browserErrors).toEqual([]);

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);

  await page.screenshot({ path: testInfo.outputPath("full-page.png"), fullPage: true });
});
