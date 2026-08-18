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

  const explorer = page.locator("#explorer");
  await explorer.getByLabel("Evaluation split").selectOption("Unseen category");
  await expect(explorer.locator(".sample-selector button")).toHaveCount(2);
  await explorer.getByRole("button", { name: /Toilet/ }).click();
  await expect(explorer.locator(".sample-meta")).toContainText("Close the toilet lid.");

  const explorerImage = explorer.locator(".sample-image-wrap img");
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

  expect(browserErrors).toEqual([]);

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);

  await page.screenshot({ path: testInfo.outputPath("full-page.png"), fullPage: true });
});
