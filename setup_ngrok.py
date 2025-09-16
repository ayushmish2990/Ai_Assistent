#!/usr/bin/env python3
"""
Script to set up ngrok authentication and create a tunnel for the AI coding assistant backend.
"""

from pyngrok import ngrok, conf
import os

def setup_ngrok():
    """Set up ngrok authentication token and create tunnel."""
    
    # Set the authentication token
    auth_token = "32lxmVvlcVwe0aIQldrGTv9sB0c_2kNsr5fcetExZj6St12dZ"
    
    try:
        # Set the auth token
        ngrok.set_auth_token(auth_token)
        print("✅ Ngrok authentication token set successfully!")
        
        # Create a tunnel to the local server (port 8000)
        tunnel = ngrok.connect(8000)
        print(f"🌐 Ngrok tunnel created successfully!")
        print(f"📡 Public URL: {tunnel.public_url}")
        print(f"🔗 You can now access your AI coding assistant at: {tunnel.public_url}")
        print(f"📚 API Documentation: {tunnel.public_url}/docs")
        print(f"🔧 Admin Interface: {tunnel.public_url}/admin")
        
        # Keep the tunnel alive
        print("\n⚡ Tunnel is active! Press Ctrl+C to stop...")
        try:
            # Keep the script running
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping ngrok tunnel...")
            ngrok.disconnect(tunnel.public_url)
            ngrok.kill()
            print("✅ Ngrok tunnel stopped successfully!")
            
    except Exception as e:
        print(f"❌ Error setting up ngrok: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Setting up ngrok tunnel for AI Coding Assistant...")
    setup_ngrok()