import { expect, test } from "@playwright/test";

test("renders the complete project page and a nonblank WebGL scene", async ({ page }, testInfo) => {
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "AffordAny", exact: true })).toBeVisible();
  const canvas = page.locator("canvas");
  await expect(canvas).toBeVisible();
  await expect(page.locator(".viewer-loading")).toHaveCount(0, { timeout: 15_000 });
  await page.waitForTimeout(2_000);

  const canvasStats = await canvas.evaluate((element: HTMLCanvasElement) => {
    const gl = (element.getContext("webgl2") || element.getContext("webgl")) as
      | WebGL2RenderingContext
      | WebGLRenderingContext
      | null;
    if (!gl) return { width: element.width, height: element.height, coloredPixels: 0, colors: 0 };

    const pixels = new Uint8Array(element.width * element.height * 4);
    gl.readPixels(0, 0, element.width, element.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const seen = new Set<string>();
    let coloredPixels = 0;
    for (let offset = 0; offset < pixels.length; offset += 16) {
      const r = pixels[offset];
      const g = pixels[offset + 1];
      const b = pixels[offset + 2];
      if (Math.abs(r - 17) + Math.abs(g - 19) + Math.abs(b - 17) > 18) coloredPixels += 1;
      if (seen.size < 256) seen.add(`${r},${g},${b}`);
    }
    return { width: element.width, height: element.height, coloredPixels, colors: seen.size };
  });

  expect(browserErrors, JSON.stringify(canvasStats)).toEqual([]);
  expect(canvasStats.width).toBeGreaterThan(300);
  expect(canvasStats.height).toBeGreaterThan(300);
  expect(canvasStats.coloredPixels).toBeGreaterThan(100);
  expect(canvasStats.colors).toBeGreaterThan(8);

  const microwaveLoaded = page.waitForResponse(
    (response) => response.url().endsWith("/assets/pointclouds/microwave.ply") && response.ok(),
  );
  await page.getByRole("button", { name: "Microwave", exact: true }).click();
  await microwaveLoaded;
  await expect(page.locator(".hero-controls")).toContainText("Pull the front panel to open the oven.");
  if (testInfo.project.name === "desktop") {
    await page.getByRole("button", { name: "Parts", exact: true }).click();
    await expect(page.getByRole("button", { name: "Parts", exact: true })).toHaveClass(/active/);
  } else {
    await expect(page.locator(".mode-control")).toBeHidden();
  }

  const explorer = page.locator("#dataset");
  await explorer.getByLabel("Evaluation split").selectOption("Unseen category");
  await expect(explorer.locator(".sample-item")).toHaveCount(2);
  await explorer.getByRole("button", { name: /Toilet/ }).click();
  await expect(explorer.locator(".sample-meta")).toContainText("Close the toilet lid.");

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);

  await page.screenshot({ path: testInfo.outputPath("full-page.png"), fullPage: true });
});
