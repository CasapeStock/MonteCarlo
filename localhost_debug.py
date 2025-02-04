import socket
import sys
import os

def check_port_availability(port=5000):
    """Check if the port is available."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result == 0:
            print(f"Port {port} is already in use!")
            return False
        else:
            print(f"Port {port} is available.")
            return True
    except Exception as e:
        print(f"Error checking port: {e}")
        return False

def check_network_interfaces():
    """List all network interfaces."""
    try:
        import netifaces
        print("Network Interfaces:")
        for interface in netifaces.interfaces():
            print(f"- {interface}")
    except ImportError:
        print("netifaces module not installed. Install with 'pip install netifaces'")

def check_firewall_settings():
    """Check basic firewall settings."""
    print("\nFirewall Diagnostic:")
    try:
        import subprocess
        
        # Windows Firewall Check
        result = subprocess.run(
            ["netsh", "advfirewall", "show", "currentprofile"], 
            capture_output=True, 
            text=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"Error checking firewall: {e}")

def check_python_environment():
    """Print Python environment details."""
    print("\nPython Environment:")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Current Working Directory: {os.getcwd()}")

def test_localhost_connection():
    """Attempt to connect to localhost."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 5000))
        
        if result == 0:
            print("Successfully connected to localhost:5000")
        else:
            print(f"Failed to connect to localhost:5000. Error code: {result}")
        
        sock.close()
    except Exception as e:
        print(f"Connection test failed: {e}")

def main():
    print("Localhost Connection Diagnostic Tool\n")
    
    # Run diagnostics
    check_port_availability()
    check_network_interfaces()
    check_firewall_settings()
    check_python_environment()
    test_localhost_connection()

if __name__ == '__main__':
    main()
