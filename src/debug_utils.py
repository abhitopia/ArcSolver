#%%
import threading
from contextlib import contextmanager

# Thread-local storage to handle debug state in multi-threaded environments
_debug_context = threading.local()

def debug_print(message):
    """
    Prints a debug message with concatenated prefixes if debugging is enabled.

    Args:
        message (str): The debug message to print.
    """
    if hasattr(_debug_context, 'enabled_stack') and _debug_context.enabled_stack:
        if _debug_context.enabled_stack[-1]:
            # Join all active prefixes with '/'
            prefix = '/'.join(_debug_context.prefixes)
            print(f"{prefix} {message}")

@contextmanager
def debug_context(prefix=None, enabled=True):
    """
    Context manager to enable or disable debug printing with concatenated prefixes.

    Args:
        prefix (str, optional): The prefix to prepend to debug messages. Defaults to None.
        enabled (bool, optional): Whether to enable debug printing within this context.
                                   Defaults to True.

    Usage:
        with debug_context(prefix="L1", enabled=True):
            debug_print("This is a debug message.")
    """
    # Initialize thread-local lists if they don't exist
    if not hasattr(_debug_context, 'prefixes'):
        _debug_context.prefixes = []
    if not hasattr(_debug_context, 'enabled_stack'):
        _debug_context.enabled_stack = []

    # Determine the new enabled state based on the parent context
    if _debug_context.enabled_stack:
        if _debug_context.enabled_stack[-1]:
            # Parent context is enabled; current enabled state depends on 'enabled' parameter
            current_enabled = enabled
        else:
            # Parent context is disabled; current enabled state must be False
            current_enabled = False
    else:
        # No parent context; use the 'enabled' parameter directly
        current_enabled = enabled

    # Push the new enabled state onto the stack
    _debug_context.enabled_stack.append(current_enabled)

    # Append the prefix if the current context is enabled and a prefix is provided
    if current_enabled and prefix:
        _debug_context.prefixes.append(prefix)

    try:
        yield
    finally:
        # Clean up: remove prefix if it was added
        if current_enabled and prefix:
            _debug_context.prefixes.pop()
        # Pop the enabled state from the stack
        _debug_context.enabled_stack.pop()
#%%

# def func_x():
#     debug_print("This is a debug message from func_x.")

# def func_y():
#     debug_print("This is a debug message from func_y.")

# print("=== Outside any debug context ===")
# func_x()  # No output
# func_y()  # No output

# print("\n=== Inside debug context 'L1' (enabled=True) ===")
# with debug_context(prefix="L1", enabled=True):
#     func_x()  # Output: L1 This is a debug message from func_x.
#     func_y()  # Output: L1 This is a debug message from func_y.

# print("\n=== Inside debug context 'L2' (enabled=False) ===")
# with debug_context(prefix="L2", enabled=False):
#     func_x()  # No output
#     func_y()  # No output

# print("\n=== Inside debug context 'L3' (enabled=True) ===")
# with debug_context(prefix="L3", enabled=True):
#     func_x()  # Output: L3 This is a debug message from func_x.
#     func_y()  # Output: L3 This is a debug message from func_y.

# print("\n=== Outside debug context again ===")
# func_x()  # No output
# func_y()  # No output

# print("\n=== Nested debug contexts ===")
# with debug_context(prefix="Outer", enabled=True):
#     func_x()  # Output: Outer This is a debug message from func_x.
#     with debug_context(prefix="Inner", enabled=True):
#         func_y()  # Output: Outer/Inner This is a debug message from func_y.
#     func_x()  # Output: Outer This is a debug message from func_x.

# print("\n=== Nested debug contexts with inner disabled ===")
# with debug_context(prefix="Outer", enabled=True):
#     func_x()  # Output: Outer This is a debug message from func_x.
#     with debug_context(prefix="Inner", enabled=False):
#         func_y()  # No output
#     func_x()  # Output: Outer This is a debug message from func_x.

# print("\n=== Nested debug contexts with outer disabled and inner enabled ===")
# with debug_context(prefix="Outer", enabled=False):
#     func_x()  # No output
#     with debug_context(prefix="Inner", enabled=True):
#         func_y()  # No output
#     func_x()  # No output
# %%
