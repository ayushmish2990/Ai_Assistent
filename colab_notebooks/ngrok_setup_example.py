#!/usr/bin/env python3
"""
Ngrok Setup Example for AI Coding Assistant
============================================

This script demonstrates how to set up ngrok tunneling for your AI coding assistant
when running in Google Colab or other cloud environments.

Usage:
    python ngrok_setup_example.py

Features:
- Automatic ngrok authentication
- Tunnel creation for backend services
- Error handling and logging
- Public URL generation
"""

from pyngrok import ngrok
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your ngrok authentication token
NGROK_AUTH_TOKEN = "32lxmVvlcVwe0aIQldrGTv9sB0c_2kNsr5fcetExZj6St12dZ"

def setup_ngrok_tunnel(port=8000, auth_token=None):
    """
    Set up ngrok tunnel for external access to your AI coding assistant.
    
    Args:
        port (int): Local port to tunnel (default: 8000)
        auth_token (str): Ngrok authentication token
        
    Returns:
        str: Public URL if successful, None otherwise
    """
    try:
        # Set authentication token
        if auth_token:
            ngrok.set_auth_token(auth_token)
            logger.info("✅ Ngrok authentication token set successfully!")
        
        # Kill any existing tunnels
        ngrok.kill()
        
        # Create tunnel
        tunnel = ngrok.connect(port)
        public_url = tunnel.public_url
        
        logger.info(f"🌐 Ngrok tunnel created successfully!")
        logger.info(f"📡 Public URL: {public_url}")
        logger.info(f"🔗 Your AI coding assistant is accessible at: {public_url}")
        logger.info(f"📚 API Documentation: {public_url}/docs")
        logger.info(f"🔧 Admin Interface: {public_url}/admin")
        
        return public_url
        
    except Exception as e:
        logger.error(f"❌ Error setting up ngrok tunnel: {e}")
        return None

def main():
    """Main function to demonstrate ngrok setup."""
    print("🚀 Setting up ngrok tunnel for AI Coding Assistant...")
    
    # Setup tunnel
    public_url = setup_ngrok_tunnel(
        port=8000, 
        auth_token=NGROK_AUTH_TOKEN
    )
    
    if public_url:
        print(f"\n✅ Setup complete! Your AI assistant is now publicly accessible.")
        print(f"🔗 Public URL: {public_url}")
        print(f"\n📋 Available endpoints:")
        print(f"   • POST {public_url}/api/v1/ai/generate-code")
        print(f"   • POST {public_url}/api/v1/ai/analyze-code") 
        print(f"   • POST {public_url}/api/v1/ai/suggest-completion")
        print(f"   • POST {public_url}/api/v1/ai/chat")
        print(f"   • GET  {public_url}/api/v1/codebase/status")
        
        print(f"\n⚡ Tunnel is active! Keep this script running to maintain the connection.")
        
        try:
            # Keep the tunnel alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping ngrok tunnel...")
            ngrok.disconnect(public_url)
            ngrok.kill()
            print(f"✅ Ngrok tunnel stopped successfully!")
    else:
        print(f"❌ Failed to set up ngrok tunnel. Please check your configuration.")

if __name__ == "__main__":
    main()