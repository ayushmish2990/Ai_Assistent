// Simple test script to verify the backend API
import fetch from 'node-fetch';

async function testBackendAPI() {
  try {
    console.log('Testing backend API at http://localhost:3001/analyze...');
    
    const testCode = `
function add(a, b) {
  return a + b;
}

console.log(add(5, 3));
`;
    
    const response = await fetch('http://localhost:3001/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        code: testCode,
        language: 'javascript'
      })
    });
    
    if (response.ok) {
      const data = await response.json();
      console.log('Backend API response:', data);
      console.log('Backend API is working correctly!');
    } else {
      console.error('Error response:', response.status, response.statusText);
      const errorText = await response.text();
      console.error('Error details:', errorText);
    }
  } catch (error) {
    console.error('Error testing backend API:', error.message);
  }
}

testBackendAPI();