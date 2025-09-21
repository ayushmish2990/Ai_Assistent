// Test script to verify frontend-backend integration
import puppeteer from 'puppeteer';

async function testFrontendBackendIntegration() {
  console.log('Starting frontend-backend integration test...');
  
  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();
  
  try {
    // Navigate to the frontend
    await page.goto('http://localhost:3000');
    console.log('Navigated to frontend');
    
    // Wait for the chat interface to load
    await page.waitForSelector('textarea');
    
    // Type a code snippet to analyze
    const codeSnippet = `function add(a, b) {\n  return a + b;\n}\n\nconsole.log(add(5, 'hello'));`;
    
    await page.type('textarea', codeSnippet);
    console.log('Entered code snippet');
    
    // Submit the code for analysis
    await page.keyboard.press('Enter');
    console.log('Submitted code for analysis');
    
    // Wait for the response
    await page.waitForFunction(
      () => document.querySelectorAll('.message').length > 1,
      { timeout: 10000 }
    );
    
    console.log('Received response from backend');
    
    // Get the response text
    const responseText = await page.evaluate(() => {
      const messages = document.querySelectorAll('.message');
      return messages[messages.length - 1].textContent;
    });
    
    console.log('Response:', responseText);
    
    // Check if the response contains expected content
    if (responseText.includes('analyzed') || responseText.includes('code')) {
      console.log('Test PASSED: Frontend-backend integration is working!');
    } else {
      console.log('Test FAILED: Unexpected response content');
    }
  } catch (error) {
    console.error('Test error:', error);
  } finally {
    // Close the browser
    await browser.close();
  }
}

testFrontendBackendIntegration();