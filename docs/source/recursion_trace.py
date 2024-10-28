import sys
import inspect

def trace(frame, event, arg):
    if event == "call":
        print(f"Recursion depth: {len(inspect.stack())}")
    return trace

sys.settrace(trace)

# Add a sample recursive function to test the trace function
def recursive_function(n):
    if n > 0:
        recursive_function(n - 1)
    else:
        print("Reached base case")

# Run the recursive function to test recursion depth tracking
recursive_function(10)
