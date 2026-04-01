import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.setViewportSize({ width: 1400, height: 900 });

await page.goto('http://localhost:3000/huggingface', { waitUntil: 'networkidle', timeout: 30000 });
await page.waitForTimeout(5000);

console.log('✅ Page loaded');

// Find and click the quannguyen dataset row header
const rowHeader = page.locator('[class*="datasetRowHeader"]').filter({ hasText: /quannguyen204/ }).first();
const headerVisible = await rowHeader.isVisible();
console.log('Dataset row header visible:', headerVisible);

if (headerVisible) {
  await rowHeader.click();
  await page.waitForTimeout(2000);
  console.log('✅ Clicked row header');
}

// Look for "Columns to Embed" label
const colsLabel = page.locator('text=Columns to Embed');
const labelVisible = await colsLabel.isVisible().catch(() => false);
console.log('"Columns to Embed" label visible:', labelVisible);

if (labelVisible) {
  // Find and click the MultiSelect dropdown
  const multiselect = page.locator('[class*="multiSelectWrapper"]').or(page.locator('[class*="multiSelectField"]')).first();
  const mselVisible = await multiselect.isVisible().catch(() => false);
  console.log('MultiSelect wrapper visible:', mselVisible);
  
  if (mselVisible) {
    await multiselect.click();
  } else {
    // Click the label itself
    await colsLabel.click();
  }
  await page.waitForTimeout(1500);
  console.log('✅ Clicked dropdown');
} else {
  // Try to find the dropdown directly
  const dropdown = page.locator('[class*="dropdown"], [class*="selectPill"]').filter({ hasText: /column|question|answer/i }).first();
  const dropVisible = await dropdown.isVisible().catch(() => false);
  console.log('Alternate dropdown visible:', dropVisible);
  if (dropVisible) {
    await dropdown.click();
    await page.waitForTimeout(1500);
  }
}

// Now count options
await page.waitForTimeout(2000);

// Try multiple selectors for dropdown options
const optionSelectors = [
  '[class*="option"]',
  '[class*="menuItem"]',
  '[class*="dropdownOption"]',
  'li[role="option"]',
  '[role="option"]',
  '[class*="item"]',
];

let allOptions = [];
for (const sel of optionSelectors) {
  const opts = await page.locator(sel).all();
  for (const opt of opts) {
    const isVisible = await opt.isVisible().catch(() => false);
    if (isVisible) {
      const text = (await opt.textContent() || '').trim();
      const bbox = await opt.boundingBox();
      if (text) {
        allOptions.push({ text, bbox, sel });
      }
    }
  }
}

// Deduplicate by text
const seen = new Set();
const uniqueOptions = allOptions.filter(o => {
  if (seen.has(o.text)) return false;
  seen.add(o.text);
  return true;
});

console.log('\n=== DROPDOWN OPTIONS ===');
console.log('Total visible options:', uniqueOptions.length);
for (const opt of uniqueOptions) {
  console.log(`  [${opt.bbox?.y?.toFixed(0)}] "${opt.text}"`);
}

// Check for question and answer
const questionOpt = uniqueOptions.find(o => o.text.toLowerCase() === 'question');
const answerOpt = uniqueOptions.find(o => o.text.toLowerCase() === 'answer');
console.log('\n"question" found:', !!questionOpt, questionOpt ? `at y=${questionOpt.bbox?.y?.toFixed(0)}` : '');
console.log('"answer" found:', !!answerOpt, answerOpt ? `at y=${answerOpt.bbox?.y?.toFixed(0)}` : '');

// Check viewport bounds
const viewportHeight = 900;
const viewportMid = viewportHeight / 2;
const questionInView = questionOpt && questionOpt.bbox && questionOpt.bbox.y >= 0 && questionOpt.bbox.y <= viewportHeight;
const answerInView = answerOpt && answerOpt.bbox && answerOpt.bbox.y >= 0 && answerOpt.bbox.y <= viewportHeight;
console.log('"question" in viewport:', questionInView);
console.log('"answer" in viewport:', answerInView);

// Also look at what the expanded panel contains
const panelText = await page.locator('[class*="panel"], [class*="expanded"], [class*="ingest"]').filter({ hasText: /column|question|answer/i }).first().textContent().catch(() => 'not found');
console.log('\nPanel content snippet:', panelText?.substring(0, 300));

await page.screenshot({ path: '/tmp/hf_multiselect.png', fullPage: true });
console.log('\nScreenshot: /tmp/hf_multiselect.png');

await browser.close();

console.log('\n=== FINAL REPORT ===');
if (questionOpt && answerOpt) {
  console.log('✅ PASS: Both "question" AND "answer" are visible in the dropdown');
} else if (questionOpt) {
  console.log('⚠️  PARTIAL: "question" is visible — "answer" is NOT visible (clipping bug)');
} else {
  console.log('❌ FAIL: Could not find dropdown or options');
}
