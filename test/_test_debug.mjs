import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.setViewportSize({ width: 1400, height: 900 });

console.log('Opening http://localhost:3000/huggingface ...');
await page.goto('http://localhost:3000/huggingface', { waitUntil: 'networkidle', timeout: 30000 });
await page.waitForTimeout(5000);

// Get page title
const title = await page.title();
console.log('Page title:', title);

// Check for any local datasets
const bodyText = await page.locator('body').textContent();
console.log('\nPage text (first 2000 chars):\n', bodyText?.substring(0, 2000));

// Look for the quannguyen dataset
const hasQuannguyen = bodyText?.includes('quannguyen');
console.log('\n"quannguyen" found on page:', hasQuannguyen);

// Get all elements with text containing 'quannguyen'
const quannguyenEls = await page.locator('*').filter({ hasText: /quannguyen/i }).all();
console.log('Elements containing quannguyen:', quannguyenEls.length);
for (const el of quannguyenEls.slice(0, 5)) {
  const tag = await el.evaluate(e => e.tagName);
  const cls = await el.evaluate(e => e.className?.substring(0, 80));
  const txt = await el.textContent();
  console.log(`  <${tag}> class="${cls}" text="${txt?.substring(0, 80)}"`);
}

// Get all sections
const sections = await page.locator('section, [class*="section"]').all();
console.log('\nSections found:', sections.length);

// Look for Local Datasets
const localSection = page.locator('*').filter({ hasText: /^Local Datasets$/ }).first();
const localSectionExists = await localSection.count();
console.log('Local Datasets section count:', localSectionExists);

// Try to find any dataset cards
const cards = await page.locator('[class*="card"]').all();
console.log('\nCard elements found:', cards.length);
for (const c of cards.slice(0, 10)) {
  const txt = await c.textContent().catch(() => 'error');
  if (txt) console.log('  Card:', txt.substring(0, 100).replace(/\s+/g, ' '));
}

await page.screenshot({ path: '/tmp/hf_page_debug.png', fullPage: true });
console.log('\nScreenshot saved to /tmp/hf_page_debug.png');

await browser.close();
