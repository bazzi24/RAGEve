import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

console.log('Opening http://localhost:3000/huggingface ...');
await page.goto('http://localhost:3000/huggingface', { waitUntil: 'networkidle', timeout: 30000 });

// Wait for the page to load
await page.waitForTimeout(3000);

// Look for "Local Datasets" section
const localSection = await page.locator('text=Local Datasets').first();
const localSectionVisible = await localSection.isVisible().catch(() => false);
console.log('Local Datasets section visible:', localSectionVisible);

if (!localSectionVisible) {
  console.log('❌ Local Datasets section not found!');
  await browser.close();
  process.exit(1);
}

// Find the card for quannguyen204/combined_medical_qa_dataset
const cardLocator = page.locator('[class*="localDatasetCard"]', { hasText: 'quannguyen204/combined_medical_qa_dataset' });
const cardCount = await cardLocator.count();
console.log('Found', cardCount, 'card(s) matching quannguyen204/combined_medical_qa_dataset');

if (cardCount === 0) {
  // Try alternate selectors
  const allCards = await page.locator('[class*="card"]').all();
  console.log('All card elements found:', allCards.length);
  for (const c of allCards) {
    const text = await c.textContent();
    if (text && text.includes('quannguyen')) console.log('  Card:', text.substring(0, 100));
  }
}

// Click the card header to expand the ingest panel
const cardHeader = page.locator('[class*="cardHeader"]').filter({ hasText: 'quannguyen204/combined_medical_qa_dataset' }).first();
const headerVisible = await cardHeader.isVisible().catch(() => false);
console.log('Card header visible:', headerVisible);

if (headerVisible) {
  await cardHeader.click();
  await page.waitForTimeout(1500);
  console.log('Clicked card header');
} else {
  // Try clicking on any part of the card
  const card = page.locator('[class*="localDatasetCard"]').filter({ hasText: 'quannguyen204/combined_medical_qa_dataset' }).first();
  const cardVisible = await card.isVisible().catch(() => false);
  if (cardVisible) {
    await card.click();
    await page.waitForTimeout(1500);
    console.log('Clicked card body');
  }
}

// Look for "Columns to Embed" label
const columnsLabel = page.locator('text=Columns to Embed').first();
const labelVisible = await columnsLabel.isVisible().catch(() => false);
console.log('"Columns to Embed" label visible:', labelVisible);

// Click the MultiSelect dropdown
if (labelVisible) {
  // Find the dropdown button/pill area near "Columns to Embed"
  const multiselect = page.locator('[class*="multiSelectWrapper"]').or(page.locator('[class*="multiSelect"]')).first();
  const mselVisible = await multiselect.isVisible().catch(() => false);
  if (mselVisible) {
    await multiselect.click();
    await page.waitForTimeout(1000);
    console.log('Clicked MultiSelect dropdown');
  } else {
    // Try clicking the label or nearby element
    await columnsLabel.click();
    await page.waitForTimeout(1000);
    console.log('Clicked Columns to Embed label');
  }
} else {
  // Look for any dropdown
  const dropdown = page.locator('[class*="dropdown"], [class*="select"]').filter({ hasText: /column/i }).first();
  const dropVisible = await dropdown.isVisible().catch(() => false);
  console.log('Alternate dropdown visible:', dropVisible);
  if (dropVisible) {
    await dropdown.click();
    await page.waitForTimeout(1000);
  }
}

// Count visible options in the dropdown
await page.waitForTimeout(1500);

// Look for option elements
const options = await page.locator('[class*="option"], [class*="menuItem"], li[class*="item"]').all();
let optionTexts = [];
for (const opt of options) {
  const isVisible = await opt.isVisible().catch(() => false);
  if (isVisible) {
    const text = (await opt.textContent() || '').trim();
    if (text) optionTexts.push(text);
  }
}
console.log('\nVisible options found:', optionTexts.length);
console.log('Options:', optionTexts);

// Check specifically for "question" and "answer"
const hasQuestion = optionTexts.some(t => t.toLowerCase().includes('question'));
const hasAnswer = optionTexts.some(t => t.toLowerCase().includes('answer'));
console.log('\n"question" visible:', hasQuestion);
console.log('"answer" visible:', hasAnswer);

// Also try a more targeted search
const questionOption = page.locator('[class*="option"]', { hasText: /^question$/i }).first();
const answerOption = page.locator('[class*="option"]', { hasText: /^answer$/i }).first();
const questionVisible = await questionOption.isVisible().catch(() => false);
const answerVisible = await answerOption.isVisible().catch(() => false);
console.log('\nTargeted "question" option visible:', questionVisible);
console.log('Targeted "answer" option visible:', answerVisible);

// Get full dropdown content
const dropdownContent = await page.locator('[class*="dropdown"], [class*="menu"], [class*="options"]').filter({ hasText: /question|answer/i }).first().textContent().catch(() => 'not found');
console.log('\nDropdown content snippet:', dropdownContent?.substring(0, 300));

// Screenshot for reference
await page.screenshot({ path: '/tmp/multiselect_test.png', fullPage: true });
console.log('\nScreenshot saved to /tmp/multiselect_test.png');

await browser.close();

console.log('\n=== FINAL REPORT ===');
if (questionVisible && answerVisible) {
  console.log('✅ PASS: Both "question" AND "answer" are visible in the dropdown');
} else if (questionVisible) {
  console.log('⚠️  PARTIAL: Only "question" is visible — "answer" is NOT visible (clipping issue)');
} else {
  console.log('❌ FAIL: Could not verify dropdown options');
}
