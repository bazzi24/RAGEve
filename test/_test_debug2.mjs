import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.setViewportSize({ width: 1400, height: 900 });

await page.goto('http://localhost:3000/huggingface', { waitUntil: 'networkidle', timeout: 30000 });
await page.waitForTimeout(5000);

// Find all elements that contain "quannguyen"
const quannguyenEls = await page.locator('*').filter({ hasText: /quannguyen/i }).all();
console.log('Elements with quannguyen:', quannguyenEls.length);
for (const el of quannguyenEls.slice(0, 10)) {
  const tag = await el.evaluate(e => e.tagName);
  const cls = await el.evaluate(e => (e.className || '').substring(0, 120));
  const rect = await el.boundingBox();
  const txt = await el.textContent();
  console.log(`\n  <${tag}> class="${cls}"`);
  console.log(`  bbox: ${JSON.stringify(rect)}`);
  console.log(`  text: "${txt?.substring(0, 120)}"`);
}

// Get the parent of the quannguyen text element
const firstQuannguyen = page.locator('*').filter({ hasText: /quannguyen204\/combined_medical_qa_dataset/i }).first();
const parent = await firstQuannguyen.locator('xpath=..').evaluate(e => ({
  tag: e.tagName,
  cls: e.className,
  id: e.id,
  children: Array.from(e.children).map(c => ({ tag: c.tagName, cls: c.className?.substring(0, 80) }))
}));
console.log('\nParent element:', JSON.stringify(parent, null, 2));

await page.screenshot({ path: '/tmp/hf_debug2.png', fullPage: true });

await browser.close();
