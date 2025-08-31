#!/usr/bin/env python3
"""
Runner script for the IMO25 agent that avoids relative import issues.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Add the code directory to the path so we can import modules directly
imo25_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(imo25_dir, 'code')
sys.path.insert(0, code_dir)

if __name__ == "__main__":
    import sys
    # Import and run the agent directly
    from code import agent
    # Call main function if it exists, otherwise run the script logic directly
    if hasattr(agent, 'main'):
        # Pass command line arguments to the agent
        original_argv = sys.argv
        sys.argv = ['agent.py'] + sys.argv[1:]  # Keep the script name and add arguments
        try:
            agent.main()
        finally:
            sys.argv = original_argv  # Restore original argv
    else:
        # If there's no main function, we need to execute the script logic
        # This is a workaround for scripts that don't have a main function
        pass

