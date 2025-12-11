---
title: Python Interview Questions
description: 100+ Python interview questions for cracking Data Science, Backend, and Software Engineering interviews
---

# Python Interview Questions

This comprehensive guide contains **100+ Python interview questions** commonly asked at top tech companies like Google, Amazon, Meta, Microsoft, and Netflix. Each premium question includes detailed explanations, code examples, and interviewer insights.

---

## Premium Interview Questions

Master these frequently asked Python questions with detailed explanations, code examples, and insights into what interviewers really look for.

---

### What is the Global Interpreter Lock (GIL) in Python? - Google, Meta, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Internals`, `Concurrency`, `Performance` | **Asked by:** Google, Meta, Amazon, Apple, Netflix

??? success "View Answer"

    **What is the GIL?**
    
    The Global Interpreter Lock is a mutex (lock) that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously in CPython.
    
    **Why Does It Exist?**
    
    - CPython's memory management (reference counting) is not thread-safe
    - Without GIL, race conditions would corrupt object reference counts
    - Simplifies the implementation of CPython
    
    **Impact on Performance:**
    
    | Task Type | Impact | Solution |
    |-----------|--------|----------|
    | CPU-bound | Significant slowdown | Use `multiprocessing` |
    | I/O-bound | Minimal impact | `threading` works fine |
    | C extensions | Can release GIL | NumPy, Pandas are efficient |
    
    ```python
    import threading
    import multiprocessing
    import time
    
    def cpu_bound_task(n):
        """CPU-intensive calculation"""
        return sum(i*i for i in range(n))
    
    # Threading (limited by GIL for CPU-bound)
    def run_threaded():
        threads = [threading.Thread(target=cpu_bound_task, args=(10**7,)) 
                   for _ in range(4)]
        start = time.time()
        for t in threads: t.start()
        for t in threads: t.join()
        print(f"Threaded: {time.time() - start:.2f}s")  # ~4s (no parallelism)
    
    # Multiprocessing (bypasses GIL)
    def run_multiprocess():
        with multiprocessing.Pool(4) as pool:
            start = time.time()
            pool.map(cpu_bound_task, [10**7]*4)
            print(f"Multiprocess: {time.time() - start:.2f}s")  # ~1s (true parallelism)
    
    # For I/O-bound, asyncio is often best
    import asyncio
    
    async def io_bound_task():
        await asyncio.sleep(1)  # Simulates I/O wait
        return "done"
    
    async def run_async():
        start = time.time()
        await asyncio.gather(*[io_bound_task() for _ in range(4)])
        print(f"Async: {time.time() - start:.2f}s")  # ~1s (concurrent I/O)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Deep understanding of Python internals and concurrency.
        
        **Strong answer signals:**
        
        - Knows GIL only affects CPython (not Jython, PyPy with STM)
        - Can explain reference counting and why GIL exists
        - Gives practical solutions: multiprocessing, asyncio, C extensions
        - Mentions upcoming work: "Python 3.12+ has per-interpreter GIL, free-threading in 3.13"

---

### Explain Python Decorators with Examples - Google, Amazon, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Functions`, `Advanced`, `Metaprogramming` | **Asked by:** Google, Amazon, Netflix, Meta, Microsoft

??? success "View Answer"

    **What Are Decorators?**
    
    Decorators are functions that modify the behavior of other functions or classes without changing their source code. They're a form of metaprogramming.
    
    **Basic Syntax:**
    
    ```python
    @decorator
    def function():
        pass
    
    # Equivalent to:
    function = decorator(function)
    ```
    
    **Common Use Cases:**
    
    ```python
    import functools
    import time
    
    # 1. Timing Decorator
    def timer(func):
        @functools.wraps(func)  # Preserves function metadata
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"{func.__name__} took {end - start:.4f}s")
            return result
        return wrapper
    
    @timer
    def slow_function():
        time.sleep(1)
        return "done"
    
    # 2. Memoization (Caching)
    def memoize(func):
        cache = {}
        @functools.wraps(func)
        def wrapper(*args):
            if args not in cache:
                cache[args] = func(*args)
            return cache[args]
        return wrapper
    
    @memoize  # Or use @functools.lru_cache(maxsize=128)
    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    # 3. Decorator with Arguments
    def repeat(times):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                for _ in range(times):
                    result = func(*args, **kwargs)
                return result
            return wrapper
        return decorator
    
    @repeat(times=3)
    def greet(name):
        print(f"Hello {name}")
    
    # 4. Class-based Decorator
    class CountCalls:
        def __init__(self, func):
            functools.update_wrapper(self, func)
            self.func = func
            self.count = 0
        
        def __call__(self, *args, **kwargs):
            self.count += 1
            return self.func(*args, **kwargs)
    
    @CountCalls
    def say_hello():
        print("Hello!")
    
    say_hello()
    print(f"Called {say_hello.count} times")
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of higher-order functions and Python idioms.
        
        **Strong answer signals:**
        
        - Always uses `@functools.wraps` (preserves `__name__`, `__doc__`)
        - Can write decorators with and without arguments
        - Knows built-in decorators: `@property`, `@staticmethod`, `@classmethod`
        - Mentions real use cases: logging, authentication, rate limiting

---

### What is the Difference Between `*args` and `**kwargs`? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Functions`, `Basics`, `Syntax` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **The Basics:**
    
    - `*args`: Collects positional arguments into a **tuple**
    - `**kwargs`: Collects keyword arguments into a **dictionary**
    
    ```python
    def demo(*args, **kwargs):
        print(f"args (tuple): {args}")
        print(f"kwargs (dict): {kwargs}")
    
    demo(1, 2, 3, name="Alice", age=30)
    # args (tuple): (1, 2, 3)
    # kwargs (dict): {'name': 'Alice', 'age': 30}
    ```
    
    **Order Matters:**
    
    ```python
    def function(regular, *args, keyword_only, **kwargs):
        pass
    
    # Call: function(1, 2, 3, keyword_only="required", extra="optional")
    ```
    
    **Practical Use Cases:**
    
    ```python
    # 1. Wrapper functions (decorators)
    def wrapper(func):
        def inner(*args, **kwargs):
            print("Before")
            result = func(*args, **kwargs)  # Pass all args through
            print("After")
            return result
        return inner
    
    # 2. Unpacking for function calls
    def greet(name, age, city):
        print(f"{name}, {age}, from {city}")
    
    data = ("Alice", 30, "NYC")
    greet(*data)  # Unpacks tuple
    
    info = {"name": "Bob", "age": 25, "city": "LA"}
    greet(**info)  # Unpacks dict
    
    # 3. Combining dictionaries (Python 3.5+)
    defaults = {"theme": "dark", "lang": "en"}
    overrides = {"theme": "light"}
    settings = {**defaults, **overrides}  # {'theme': 'light', 'lang': 'en'}
    
    # 4. Forcing keyword-only arguments
    def api_call(endpoint, *, method="GET", timeout=30):
        # method and timeout MUST be passed as keywords
        pass
    
    api_call("/users", method="POST")  # Valid
    # api_call("/users", "POST")  # TypeError!
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Basic Python fluency.
        
        **Strong answer signals:**
        
        - Knows the names are convention (`*args`, `**kwargs`), asterisks matter
        - Can show argument unpacking (`*` and `**` in function calls)
        - Mentions keyword-only arguments (after `*`)
        - Gives practical example: "I use this in decorators to pass through all arguments"

---

### How Does Python Manage Memory? Explain Garbage Collection - Google, Meta, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Internals`, `Memory`, `Garbage Collection` | **Asked by:** Google, Meta, Netflix, Amazon, Spotify

??? success "View Answer"

    **Python's Memory Management:**
    
    Python uses a private heap to store all objects. Memory management is handled by:
    
    1. **Reference Counting** (Main mechanism)
    2. **Garbage Collector** (For cyclic references)
    
    **Reference Counting:**
    
    Every object has a reference count. When count → 0, memory is freed immediately.
    
    ```python
    import sys
    
    a = [1, 2, 3]
    print(sys.getrefcount(a))  # 2 (a + getrefcount's reference)
    
    b = a  # Reference count increases
    print(sys.getrefcount(a))  # 3
    
    del b  # Reference count decreases
    print(sys.getrefcount(a))  # 2
    ```
    
    **The Problem: Cyclic References**
    
    ```python
    # Reference counting can't handle cycles
    class Node:
        def __init__(self):
            self.parent = None
            self.children = []
    
    # Create a cycle
    parent = Node()
    child = Node()
    parent.children.append(child)
    child.parent = parent  # Cycle: parent ↔ child
    
    del parent, child  # Objects still exist! (cycle keeps refcount > 0)
    ```
    
    **Generational Garbage Collector:**
    
    Python's GC uses 3 generations based on object age:
    
    | Generation | Contains | Collection Frequency |
    |------------|----------|---------------------|
    | 0 | New objects | Most frequent |
    | 1 | Survived gen 0 | Less frequent |
    | 2 | Long-lived | Least frequent |
    
    ```python
    import gc
    
    # Manual GC control
    gc.collect()  # Force collection of all generations
    gc.collect(0)  # Only generation 0
    
    # Check GC thresholds
    print(gc.get_threshold())  # (700, 10, 10)
    # After 700 gen-0 allocations, collect gen 0
    # After 10 gen-0 collections, collect gen 1
    
    # Disable GC (for performance in specific scenarios)
    gc.disable()  # Be careful!
    ```
    
    **Memory Optimization Tips:**
    
    ```python
    # Use __slots__ to reduce memory
    class Point:
        __slots__ = ['x', 'y']  # No __dict__, saves ~40%
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    # Use generators for large sequences
    def squares(n):
        for i in range(n):
            yield i * i  # Memory: O(1) instead of O(n)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Deep Python internals knowledge.
        
        **Strong answer signals:**
        
        - Explains both reference counting AND generational GC
        - Knows cyclic references require GC (refcount alone won't work)
        - Can discuss `__slots__`, `weakref` for memory optimization
        - Mentions debugging: `gc.get_objects()`, `tracemalloc` module

---

### What is the Difference Between `deepcopy` and `shallow copy`? - Google, Amazon, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Objects`, `Memory`, `Data Structures` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **The Core Difference:**
    
    | Copy Type | Nested Objects | Memory |
    |-----------|----------------|--------|
    | Shallow | Shared (references) | Less |
    | Deep | New copies | More |
    
    ```python
    import copy
    
    original = [[1, 2, 3], [4, 5, 6]]
    
    # Shallow copy - nested lists are SHARED
    shallow = copy.copy(original)
    shallow[0][0] = 999
    print(original[0][0])  # 999 - Original is affected!
    
    # Deep copy - everything is copied
    original = [[1, 2, 3], [4, 5, 6]]
    deep = copy.deepcopy(original)
    deep[0][0] = 999
    print(original[0][0])  # 1 - Original is NOT affected
    ```
    
    **Visual Representation:**
    
    ```
    Shallow Copy:
    original ──→ [ptr1, ptr2]
                   ↓      ↓
    shallow  ──→ [ptr1, ptr2]  (same pointers!)
                   ↓      ↓
                [1,2,3] [4,5,6]
    
    Deep Copy:
    original ──→ [ptr1, ptr2]
                   ↓      ↓
                [1,2,3] [4,5,6]
    
    deep     ──→ [ptr3, ptr4]  (different pointers!)
                   ↓      ↓
                [1,2,3] [4,5,6]  (new objects)
    ```
    
    **Common Ways to Copy:**
    
    ```python
    # Shallow copies
    list2 = list1[:]           # Slice
    list2 = list(list1)        # Constructor
    list2 = list1.copy()       # Method
    list2 = copy.copy(list1)   # copy module
    
    dict2 = dict1.copy()       # Method
    dict2 = {**dict1}          # Unpacking
    
    # Deep copy
    deep = copy.deepcopy(original)
    
    # Custom deep copy behavior
    class MyClass:
        def __deepcopy__(self, memo):
            # Custom logic here
            return MyClass(copy.deepcopy(self.data, memo))
    ```
    
    **When to Use Which:**
    
    | Scenario | Use |
    |----------|-----|
    | Flat data structures | Shallow |
    | Nested mutable objects | Deep |
    | Performance critical | Shallow + be careful |
    | Immutable nested data | Shallow (safe) |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of Python's object model.
        
        **Strong answer signals:**
        
        - Can draw memory diagrams showing shared references
        - Knows list slicing `[:]` is shallow, not deep
        - Mentions immutable objects don't need copying
        - Knows `deepcopy` handles cycles (uses memo dict)

---

### What Are Python Generators? When Should You Use Them? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Iterators`, `Memory`, `Performance` | **Asked by:** Google, Amazon, Netflix, Meta, Apple

??? success "View Answer"

    **What Are Generators?**
    
    Generators are functions that return an iterator and yield values one at a time, maintaining state between calls. They're memory-efficient for large sequences.
    
    **Why Use Generators:**
    
    | List | Generator |
    |------|-----------|
    | All items in memory | One item at a time |
    | O(n) memory | O(1) memory |
    | Can iterate multiple times | Single-use (exhausted) |
    | Random access | Sequential only |
    
    ```python
    # Generator Function (uses yield)
    def count_up_to(n):
        i = 1
        while i <= n:
            yield i  # Pauses here, resumes on next()
            i += 1
    
    gen = count_up_to(5)
    print(next(gen))  # 1
    print(next(gen))  # 2
    # Or iterate
    for num in count_up_to(5):
        print(num)
    
    # Generator Expression (like list comprehension)
    squares_list = [x**2 for x in range(1000000)]  # 8MB in memory
    squares_gen = (x**2 for x in range(1000000))   # ~128 bytes!
    
    # Practical Example: Reading Large Files
    def read_large_file(file_path):
        with open(file_path) as f:
            for line in f:  # Already a generator!
                yield line.strip()
    
    # Process 1TB file with constant memory
    for line in read_large_file("huge_file.txt"):
        process(line)
    
    # Chaining Generators (Pipeline)
    def parse_lines(lines):
        for line in lines:
            yield line.split(',')
    
    def filter_valid(records):
        for record in records:
            if len(record) == 3:
                yield record
    
    # Lazy pipeline - nothing executes until iterated
    pipeline = filter_valid(parse_lines(read_large_file("data.csv")))
    for record in pipeline:
        print(record)
    ```
    
    **Advanced: `yield from` and `send()`**
    
    ```python
    # Delegate to sub-generator
    def chain(*iterables):
        for it in iterables:
            yield from it  # Equivalent to: for x in it: yield x
    
    list(chain([1, 2], [3, 4]))  # [1, 2, 3, 4]
    
    # Two-way communication with send()
    def accumulator():
        total = 0
        while True:
            value = yield total
            if value is not None:
                total += value
    
    acc = accumulator()
    next(acc)  # Initialize
    print(acc.send(10))  # 10
    print(acc.send(5))   # 15
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of lazy evaluation and memory efficiency.
        
        **Strong answer signals:**
        
        - Compares memory: list vs generator for large data
        - Knows files are already generators (line by line)
        - Can explain `yield from` for sub-generators
        - Mentions use cases: ETL pipelines, infinite sequences, streaming

---

### Explain Python's Method Resolution Order (MRO) - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `OOP`, `Inheritance`, `Internals` | **Asked by:** Google, Meta, Amazon, Microsoft

??? success "View Answer"

    **What is MRO?**
    
    Method Resolution Order determines the order in which base classes are searched when looking for a method. Python uses the **C3 Linearization** algorithm.
    
    **The Diamond Problem:**
    
    ```python
    class A:
        def method(self):
            print("A")
    
    class B(A):
        def method(self):
            print("B")
    
    class C(A):
        def method(self):
            print("C")
    
    class D(B, C):  # Diamond inheritance
        pass
    
    d = D()
    d.method()  # Prints "B" - but why?
    
    # Check the MRO
    print(D.__mro__)
    # (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
    ```
    
    **C3 Linearization Rules:**
    
    1. Children come before parents
    2. Parents order is preserved (left to right)
    3. Common parent appears only once, after all children
    
    **Using `super()` Correctly:**
    
    ```python
    class A:
        def __init__(self):
            print("A init")
            super().__init__()
    
    class B(A):
        def __init__(self):
            print("B init")
            super().__init__()
    
    class C(A):
        def __init__(self):
            print("C init")
            super().__init__()
    
    class D(B, C):
        def __init__(self):
            print("D init")
            super().__init__()
    
    d = D()
    # Output:
    # D init
    # B init
    # C init
    # A init
    
    # super() follows MRO, not just parent class!
    ```
    
    **Practical Implications:**
    
    ```python
    # Mixins should use super() to play nice with MRO
    class LoggingMixin:
        def save(self):
            print(f"Saving {self}")
            super().save()  # Calls next in MRO
    
    class TimestampMixin:
        def save(self):
            self.updated_at = datetime.now()
            super().save()
    
    class Model:
        def save(self):
            print("Saved to database")
    
    class User(LoggingMixin, TimestampMixin, Model):
        pass
    
    user = User()
    user.save()
    # Saving <User>
    # (sets updated_at)
    # Saved to database
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of Python's object model and multiple inheritance.
        
        **Strong answer signals:**
        
        - Can explain C3 linearization (children first, left-to-right)
        - Knows `super()` follows MRO, not just immediate parent
        - Mentions the diamond problem by name
        - Gives practical example: mixins using `super()`

---

### What is the Difference Between `is` and `==`? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Operators`, `Identity`, `Equality` | **Asked by:** Google, Amazon, Meta, Microsoft, Apple

??? success "View Answer"

    **The Core Difference:**
    
    | Operator | Compares | Question |
    |----------|----------|----------|
    | `==` | Value/Equality | Do they have the same content? |
    | `is` | Identity | Are they the same object in memory? |
    
    ```python
    a = [1, 2, 3]
    b = [1, 2, 3]
    c = a
    
    print(a == b)  # True (same values)
    print(a is b)  # False (different objects)
    print(a is c)  # True (same object)
    
    # Check with id()
    print(id(a), id(b), id(c))
    # 140234... 140235... 140234...  (a and c share id)
    ```
    
    **Python's Integer Caching (Gotcha!):**
    
    ```python
    # Small integers (-5 to 256) are cached
    a = 256
    b = 256
    print(a is b)  # True (cached!)
    
    a = 257
    b = 257
    print(a is b)  # False (not cached)
    
    # String interning (similar behavior)
    s1 = "hello"
    s2 = "hello"
    print(s1 is s2)  # True (interned)
    
    s1 = "hello world!"
    s2 = "hello world!"
    print(s1 is s2)  # False (not interned - has space/special chars)
    ```
    
    **When to Use Which:**
    
    | Use Case | Operator |
    |----------|----------|
    | Compare values | `==` (almost always) |
    | Check for `None` | `is` (`if x is None`) |
    | Check for singletons | `is` (True, False, None) |
    | Compare object identity | `is` (rare) |
    
    ```python
    # Correct: Use 'is' for None
    if result is None:
        print("No result")
    
    # Correct: Use 'is' for True/False in rare cases
    if flag is True:  # Stricter than 'if flag:'
        print("Explicitly True, not just truthy")
    
    # Wrong: Don't use 'is' for string/number comparison
    if name is "John":  # Bad! Use ==
        pass
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of Python's object model.
        
        **Strong answer signals:**
        
        - Immediately mentions identity vs equality
        - Knows about integer caching (-5 to 256)
        - Correctly states: "Use `is` for `None`, `==` for everything else"
        - Can explain why `is` works for small integers

---

### What Are Context Managers (`with` statement)? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Resource Management`, `Context Managers`, `Best Practices` | **Asked by:** Google, Amazon, Netflix, Meta, Microsoft

??? success "View Answer"

    **What Are Context Managers?**
    
    Context managers ensure proper acquisition and release of resources (files, locks, connections) using the `with` statement. They guarantee cleanup even if exceptions occur.
    
    **The Protocol:**
    
    ```python
    with expression as variable:
        # code block
    
    # Equivalent to:
    manager = expression
    variable = manager.__enter__()
    try:
        # code block
    finally:
        manager.__exit__(exc_type, exc_val, exc_tb)
    ```
    
    **Common Built-in Context Managers:**
    
    ```python
    # File handling
    with open('file.txt', 'r') as f:
        content = f.read()
    # File automatically closed, even if exception
    
    # Thread locks
    import threading
    lock = threading.Lock()
    with lock:
        # Critical section
        pass
    
    # Database connections
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute("SELECT * FROM users")
    # Connection closed automatically
    ```
    
    **Creating Custom Context Managers:**
    
    ```python
    # Method 1: Class-based
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.elapsed = time.perf_counter() - self.start
            print(f"Elapsed: {self.elapsed:.4f}s")
            return False  # Don't suppress exceptions
    
    with Timer() as t:
        time.sleep(1)
    # Prints: Elapsed: 1.00XXs
    
    # Method 2: Using contextlib (simpler)
    from contextlib import contextmanager
    
    @contextmanager
    def timer():
        start = time.perf_counter()
        try:
            yield  # Code inside 'with' block runs here
        finally:
            elapsed = time.perf_counter() - start
            print(f"Elapsed: {elapsed:.4f}s")
    
    with timer():
        time.sleep(1)
    
    # Method 3: Suppress specific exceptions
    from contextlib import suppress
    
    with suppress(FileNotFoundError):
        os.remove('nonexistent.txt')
    # No error raised!
    ```
    
    **Exception Handling in `__exit__`:**
    
    ```python
    class SuppressError:
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is ValueError:
                print(f"Suppressed: {exc_val}")
                return True  # Suppress the exception
            return False  # Re-raise other exceptions
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of resource management and Python idioms.
        
        **Strong answer signals:**
        
        - Knows `__enter__` and `__exit__` methods
        - Can use `@contextmanager` decorator for simple cases
        - Mentions exception suppression (`return True` in `__exit__`)
        - Gives practical examples: file handling, database connections, timing

---

### How Do You Handle Exceptions in Python? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Error Handling`, `Exceptions`, `Best Practices` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **Basic Exception Handling:**
    
    ```python
    try:
        result = risky_operation()
    except ValueError as e:
        print(f"Value error: {e}")
    except (TypeError, KeyError) as e:
        print(f"Type or Key error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise  # Re-raise after logging
    else:
        print("Success! No exceptions occurred")
    finally:
        cleanup()  # Always runs
    ```
    
    **Exception Hierarchy (Partial):**
    
    ```
    BaseException
    ├── SystemExit
    ├── KeyboardInterrupt
    └── Exception
        ├── ValueError
        ├── TypeError
        ├── KeyError
        ├── IndexError
        ├── FileNotFoundError
        └── ...
    ```
    
    **Best Practices:**
    
    ```python
    # ✅ Catch specific exceptions
    try:
        value = int(user_input)
    except ValueError:
        print("Invalid number")
    
    # ❌ Don't catch bare Exception (hides bugs)
    try:
        do_something()
    except:  # Bad! Catches KeyboardInterrupt too
        pass
    
    # ✅ Use exception chaining
    try:
        process_data()
    except ValueError as original:
        raise DataProcessingError("Failed to process") from original
    
    # ✅ Create custom exceptions
    class ValidationError(Exception):
        def __init__(self, field, message):
            self.field = field
            self.message = message
            super().__init__(f"{field}: {message}")
    
    raise ValidationError("email", "Invalid format")
    
    # ✅ EAFP (Easier to Ask Forgiveness than Permission)
    # Pythonic:
    try:
        value = dictionary[key]
    except KeyError:
        value = default
    
    # Also good (for this specific case):
    value = dictionary.get(key, default)
    
    # ✅ Context managers for cleanup
    with open('file.txt') as f:
        data = f.read()  # File closed even if exception
    ```
    
    **Advanced: Exception Groups (Python 3.11+):**
    
    ```python
    # Handle multiple exceptions at once
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(task1())
            tg.create_task(task2())
    except* ValueError as eg:
        for e in eg.exceptions:
            print(f"ValueError: {e}")
    except* TypeError as eg:
        for e in eg.exceptions:
            print(f"TypeError: {e}")
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of error handling patterns.
        
        **Strong answer signals:**
        
        - Always catches specific exceptions, never bare `except:`
        - Knows the difference between `else` and `finally`
        - Uses exception chaining (`from original`)
        - Mentions EAFP vs LBYL (Look Before You Leap)

---

### What is `__init__` vs `__new__` in Python? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `OOP`, `Object Creation`, `Internals` | **Asked by:** Google, Meta, Amazon, Microsoft

??? success "View Answer"

    **The Difference:**
    
    | Method | Purpose | Returns | When Called |
    |--------|---------|---------|-------------|
    | `__new__` | Creates the instance | Instance object | Before `__init__` |
    | `__init__` | Initializes the instance | None | After `__new__` |
    
    **Normal Flow:**
    
    ```python
    class MyClass:
        def __new__(cls, *args, **kwargs):
            print("1. __new__ called")
            instance = super().__new__(cls)  # Create instance
            return instance
        
        def __init__(self, value):
            print("2. __init__ called")
            self.value = value
    
    obj = MyClass(42)
    # Output:
    # 1. __new__ called
    # 2. __init__ called
    ```
    
    **When to Override `__new__`:**
    
    ```python
    # 1. Singleton Pattern
    class Singleton:
        _instance = None
        
        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    s1 = Singleton()
    s2 = Singleton()
    print(s1 is s2)  # True
    
    # 2. Immutable Types (can't modify in __init__)
    class Point(tuple):
        def __new__(cls, x, y):
            return super().__new__(cls, (x, y))
        
        @property
        def x(self): return self[0]
        
        @property
        def y(self): return self[1]
    
    p = Point(3, 4)
    print(p.x, p.y)  # 3 4
    
    # 3. Instance Caching
    class CachedClass:
        _cache = {}
        
        def __new__(cls, key):
            if key not in cls._cache:
                instance = super().__new__(cls)
                cls._cache[key] = instance
            return cls._cache[key]
    
    # 4. Returning Different Types
    class Factory:
        def __new__(cls, animal_type):
            if animal_type == "dog":
                return Dog()
            elif animal_type == "cat":
                return Cat()
            return super().__new__(cls)
    ```
    
    **Critical Rule:**
    
    `__init__` is only called if `__new__` returns an instance of `cls`:
    
    ```python
    class Weird:
        def __new__(cls):
            return "I'm a string!"  # Not an instance of Weird
        
        def __init__(self):
            print("This never runs!")
    
    obj = Weird()
    print(obj)  # "I'm a string!"
    print(type(obj))  # <class 'str'>
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Deep OOP understanding.
        
        **Strong answer signals:**
        
        - Knows `__new__` creates, `__init__` initializes
        - Can implement singleton using `__new__`
        - Mentions subclassing immutables (tuple, str, int) requires `__new__`
        - Knows `__init__` not called if `__new__` returns different type

---

### What is the Difference Between `staticmethod` and `classmethod`? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `OOP`, `Methods`, `Decorators` | **Asked by:** Google, Amazon, Meta, Microsoft, Apple

??? success "View Answer"

    **The Three Method Types:**
    
    | Method Type | First Argument | Access to |
    |-------------|----------------|-----------|
    | Instance | `self` | Instance + class attributes |
    | Class | `cls` | Class attributes only |
    | Static | None | Nothing (just a function) |
    
    ```python
    class MyClass:
        class_attr = "I'm a class attribute"
        
        def __init__(self, value):
            self.instance_attr = value
        
        def instance_method(self):
            # Has access to self and cls (via self.__class__)
            return f"Instance: {self.instance_attr}, Class: {self.class_attr}"
        
        @classmethod
        def class_method(cls):
            # Has access to cls, not self
            return f"Class: {cls.class_attr}"
        
        @staticmethod
        def static_method(x, y):
            # No access to self or cls
            return x + y
    
    obj = MyClass("hello")
    print(obj.instance_method())  # Instance: hello, Class: I'm a class attribute
    print(MyClass.class_method())  # Class: I'm a class attribute
    print(MyClass.static_method(2, 3))  # 5
    ```
    
    **When to Use Each:**
    
    ```python
    class Pizza:
        def __init__(self, ingredients):
            self.ingredients = ingredients
        
        # Factory method - use @classmethod
        @classmethod
        def margherita(cls):
            return cls(["mozzarella", "tomatoes"])
        
        @classmethod
        def pepperoni(cls):
            return cls(["mozzarella", "pepperoni"])
        
        # Utility function - use @staticmethod
        @staticmethod
        def calculate_price(base, toppings_count):
            return base + toppings_count * 1.5
    
    p = Pizza.margherita()  # Factory pattern
    print(Pizza.calculate_price(10, 3))  # 14.5
    ```
    
    **Inheritance Behavior:**
    
    ```python
    class Parent:
        name = "Parent"
        
        @classmethod
        def who_am_i(cls):
            return cls.name  # Uses cls, not hardcoded
    
    class Child(Parent):
        name = "Child"
    
    print(Parent.who_am_i())  # Parent
    print(Child.who_am_i())   # Child (polymorphic!)
    
    # Static method doesn't get this benefit
    class Parent2:
        @staticmethod
        def who_am_i():
            return Parent2.name  # Hardcoded!
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of Python's method types.
        
        **Strong answer signals:**
        
        - Knows `@classmethod` for factory methods
        - Knows `@staticmethod` is just a namespaced function
        - Mentions inheritance: `@classmethod` is polymorphic
        - Gives real use case: alternative constructors

---

### What Are List Comprehensions and When Should You Use Them? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Syntax`, `Comprehensions`, `Performance` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **Basic Syntax:**
    
    ```python
    # List comprehension
    [expression for item in iterable if condition]
    
    # Equivalent loop
    result = []
    for item in iterable:
        if condition:
            result.append(expression)
    ```
    
    **Types of Comprehensions:**
    
    ```python
    # List comprehension
    squares = [x**2 for x in range(10)]
    
    # Dict comprehension
    square_dict = {x: x**2 for x in range(10)}
    
    # Set comprehension
    unique_squares = {x**2 for x in [-3, -2, -1, 0, 1, 2, 3]}
    
    # Generator expression (lazy, memory efficient)
    squares_gen = (x**2 for x in range(10))
    ```
    
    **Practical Examples:**
    
    ```python
    # Filtering
    evens = [x for x in range(20) if x % 2 == 0]
    
    # Transformation
    names = ["alice", "bob", "charlie"]
    capitalized = [name.title() for name in names]
    
    # Nested loops (flattening)
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flat = [num for row in matrix for num in row]
    
    # Conditional expression
    labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]
    
    # Multiple conditions
    filtered = [x for x in range(100) if x % 2 == 0 if x % 3 == 0]
    
    # Dict from two lists
    keys = ["a", "b", "c"]
    values = [1, 2, 3]
    combined = {k: v for k, v in zip(keys, values)}
    
    # Filtering dict
    original = {"a": 1, "b": 2, "c": 3, "d": 4}
    filtered = {k: v for k, v in original.items() if v > 2}
    ```
    
    **When NOT to Use:**
    
    ```python
    # ❌ Too complex - use regular loop
    result = [
        process(x) for x in data 
        if validate(x) 
        for y in get_related(x) 
        if check(y)
    ]
    
    # ✅ Better: regular loop for readability
    result = []
    for x in data:
        if validate(x):
            for y in get_related(x):
                if check(y):
                    result.append(process(x))
    
    # ❌ Side effects - use regular loop
    [print(x) for x in items]  # Works but bad practice
    
    # ✅ Better
    for x in items:
        print(x)
    ```
    
    **Performance Note:**
    
    ```python
    # List comprehension is ~20-30% faster than equivalent loop
    # Because it's optimized at bytecode level
    
    import timeit
    
    # Comprehension
    timeit.timeit('[x**2 for x in range(1000)]', number=10000)
    
    # Loop
    timeit.timeit('''
    result = []
    for x in range(1000):
        result.append(x**2)
    ''', number=10000)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Python idiom knowledge.
        
        **Strong answer signals:**
        
        - Knows all types: list, dict, set, generator expression
        - Knows when NOT to use (too complex, side effects)
        - Mentions performance advantage (~20-30% faster)
        - Uses generator expression for memory efficiency

---

### What is Asyncio in Python? How Does It Work? - Google, Meta, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Concurrency`, `Asyncio`, `Async/Await` | **Asked by:** Google, Meta, Netflix, Amazon, Apple

??? success "View Answer"

    **What is Asyncio?**
    
    Asyncio is Python's library for writing concurrent code using async/await syntax. It's designed for I/O-bound tasks (network requests, file I/O, database queries).
    
    **Key Concepts:**
    
    | Concept | Description |
    |---------|-------------|
    | Coroutine | `async def` function, can be paused/resumed |
    | Event Loop | Runs coroutines, manages I/O |
    | `await` | Pause coroutine, wait for result |
    | Task | Wrapped coroutine, runs concurrently |
    
    ```python
    import asyncio
    
    # A coroutine
    async def fetch_data(url):
        print(f"Fetching {url}")
        await asyncio.sleep(1)  # Simulate I/O
        return f"Data from {url}"
    
    # Running coroutines
    async def main():
        # Sequential (takes 3 seconds)
        result1 = await fetch_data("url1")
        result2 = await fetch_data("url2")
        result3 = await fetch_data("url3")
        
        # Concurrent (takes 1 second!)
        results = await asyncio.gather(
            fetch_data("url1"),
            fetch_data("url2"),
            fetch_data("url3")
        )
        print(results)
    
    asyncio.run(main())
    ```
    
    **Practical Example: HTTP Requests**
    
    ```python
    import aiohttp
    import asyncio
    
    async def fetch(session, url):
        async with session.get(url) as response:
            return await response.text()
    
    async def fetch_all(urls):
        async with aiohttp.ClientSession() as session:
            tasks = [fetch(session, url) for url in urls]
            return await asyncio.gather(*tasks)
    
    urls = [f"https://httpbin.org/delay/1" for _ in range(10)]
    
    # Synchronous: 10 seconds
    # Async: ~1 second
    results = asyncio.run(fetch_all(urls))
    ```
    
    **Common Patterns:**
    
    ```python
    # Task with timeout
    async def with_timeout():
        try:
            result = await asyncio.wait_for(slow_operation(), timeout=5.0)
        except asyncio.TimeoutError:
            print("Operation timed out")
    
    # Producer-Consumer with Queue
    async def producer(queue):
        for i in range(5):
            await queue.put(i)
            await asyncio.sleep(0.1)
    
    async def consumer(queue):
        while True:
            item = await queue.get()
            print(f"Consumed {item}")
            queue.task_done()
    
    async def main():
        queue = asyncio.Queue()
        producers = [asyncio.create_task(producer(queue))]
        consumers = [asyncio.create_task(consumer(queue)) for _ in range(3)]
        
        await asyncio.gather(*producers)
        await queue.join()
        for c in consumers:
            c.cancel()
    
    # Semaphore for rate limiting
    async def rate_limited_fetch(sem, url):
        async with sem:  # Max 10 concurrent
            return await fetch(url)
    
    sem = asyncio.Semaphore(10)
    tasks = [rate_limited_fetch(sem, url) for url in urls]
    ```
    
    **Asyncio vs Threading:**
    
    | Aspect | Asyncio | Threading |
    |--------|---------|-----------|
    | Best for | I/O-bound | I/O-bound (simpler code) |
    | Concurrency | Cooperative (explicit `await`) | Preemptive (OS switches) |
    | Debugging | Easier (predictable switching) | Harder (race conditions) |
    | Libraries | Need async versions | Work with any library |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of async programming model.
        
        **Strong answer signals:**
        
        - Knows it's for I/O-bound, not CPU-bound tasks
        - Can explain event loop and cooperative multitasking
        - Uses `asyncio.gather()` for concurrency
        - Mentions `aiohttp` for HTTP, `asyncpg` for databases

---


### What Are Python Dataclasses? When Should You Use Them? - Google, Amazon, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `OOP`, `Dataclasses`, `Python 3.7+` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **What Are Dataclasses?**
    
    Dataclasses are a decorator-based way to create classes primarily used for storing data, with automatic `__init__`, `__repr__`, `__eq__`, and more.
    
    ```python
    from dataclasses import dataclass, field
    from typing import List
    
    @dataclass
    class User:
        name: str
        email: str
        age: int = 0  # Default value
        tags: List[str] = field(default_factory=list)  # Mutable default
        
        def is_adult(self) -> bool:
            return self.age >= 18
    
    # Auto-generated __init__, __repr__, __eq__
    user = User("Alice", "alice@example.com", 25)
    print(user)  # User(name='Alice', email='alice@example.com', age=25, tags=[])
    
    # Comparison works
    user2 = User("Alice", "alice@example.com", 25)
    print(user == user2)  # True
    ```
    
    **Advanced Features:**
    
    ```python
    from dataclasses import dataclass, field, asdict, astuple
    
    @dataclass(frozen=True)  # Immutable
    class Point:
        x: float
        y: float
    
    @dataclass(order=True)  # Adds __lt__, __le__, __gt__, __ge__
    class SortableItem:
        sort_key: int = field(init=False, repr=False)
        name: str
        value: int
        
        def __post_init__(self):
            self.sort_key = self.value
    
    # Convert to dict/tuple
    user_dict = asdict(user)
    user_tuple = astuple(user)
    ```
    
    **When to Use:**
    
    | Use Dataclass | Use Regular Class |
    |---------------|-------------------|
    | Data containers | Complex behavior |
    | Simple defaults | Custom __init__ logic |
    | Need __eq__, __hash__ | Inheritance-heavy |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Modern Python knowledge.
        
        **Strong answer signals:**
        
        - Uses `field(default_factory=list)` for mutable defaults
        - Knows `frozen=True` for immutability
        - Mentions `__post_init__` for validation
        - Compares to NamedTuple, attrs, Pydantic

---

### What Are Type Hints in Python? How Do You Use Them? - Google, Meta, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Type Hints`, `Static Analysis`, `Python 3.5+` | **Asked by:** Google, Meta, Microsoft, Amazon

??? success "View Answer"

    **What Are Type Hints?**
    
    Type hints are optional annotations that indicate expected types. They don't enforce types at runtime but enable static analysis with tools like mypy.
    
    ```python
    from typing import List, Dict, Optional, Union, Callable, TypeVar, Generic
    
    # Basic type hints
    def greet(name: str) -> str:
        return f"Hello, {name}"
    
    # Collections
    def process_items(items: List[int]) -> Dict[str, int]:
        return {"sum": sum(items), "count": len(items)}
    
    # Optional (can be None)
    def find_user(user_id: int) -> Optional[dict]:
        return None if user_id < 0 else {"id": user_id}
    
    # Union (multiple types)
    def handle_input(value: Union[str, int]) -> str:
        return str(value)
    
    # Python 3.10+ simplified syntax
    def handle_input_new(value: str | int | None) -> str:
        return str(value) if value else ""
    
    # Callable
    def apply_func(func: Callable[[int, int], int], a: int, b: int) -> int:
        return func(a, b)
    
    # TypeVar for generics
    T = TypeVar('T')
    def first(items: List[T]) -> Optional[T]:
        return items[0] if items else None
    ```
    
    **Class Type Hints:**
    
    ```python
    from typing import ClassVar
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        name: str
        debug: bool = False
        max_retries: ClassVar[int] = 3  # Class variable
    
    # Forward references (class not yet defined)
    class Node:
        def __init__(self, value: int, next: "Node" = None):
            self.value = value
            self.next = next
    
    # Python 3.11+ Self type
    from typing import Self
    class Builder:
        def set_name(self, name: str) -> Self:
            self.name = name
            return self
    ```
    
    **Type Checking:**
    
    ```bash
    # mypy static analysis
    pip install mypy
    mypy script.py
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Code quality and maintainability practices.
        
        **Strong answer signals:**
        
        - Knows type hints are optional (not enforced at runtime)
        - Uses `Optional` correctly (not `Union[X, None]`)
        - Mentions mypy for static checking
        - Knows `TypeVar` for generic functions

---

### Explain `__slots__` in Python - When and Why to Use It? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Memory Optimization`, `OOP`, `Performance` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **What Are `__slots__`?**
    
    `__slots__` declares fixed set of attributes, preventing creation of `__dict__` and `__weakref__`, reducing memory usage by ~40%.
    
    ```python
    import sys
    
    # Without __slots__
    class RegularPoint:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    # With __slots__
    class SlottedPoint:
        __slots__ = ['x', 'y']
        
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    # Memory comparison
    regular = RegularPoint(1, 2)
    slotted = SlottedPoint(1, 2)
    
    print(sys.getsizeof(regular.__dict__))  # ~104 bytes for dict
    # slotted has no __dict__
    
    # With millions of instances, savings are significant
    ```
    
    **Limitations:**
    
    ```python
    class Slotted:
        __slots__ = ['x', 'y']
    
    obj = Slotted()
    obj.x = 1
    obj.y = 2
    obj.z = 3  # AttributeError: 'Slotted' object has no attribute 'z'
    
    # Can't use __dict__ for dynamic attributes
    # Can't use __weakref__ by default (add to __slots__ if needed)
    ```
    
    **Inheritance with `__slots__`:**
    
    ```python
    class Base:
        __slots__ = ['x']
    
    class Child(Base):
        __slots__ = ['y']  # Only add new slots, don't repeat 'x'
        
    c = Child()
    c.x = 1  # From Base
    c.y = 2  # From Child
    
    # If child doesn't define __slots__, it gets __dict__
    class ChildWithDict(Base):
        pass  # Has __dict__, can add any attribute
    ```
    
    **When to Use:**
    
    | Use __slots__ | Avoid __slots__ |
    |---------------|-----------------|
    | Millions of instances | Need dynamic attributes |
    | Fixed attributes known | Multiple inheritance |
    | Memory-constrained | Metaclasses that need __dict__ |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Python memory model understanding.
        
        **Strong answer signals:**
        
        - Knows __slots__ prevents __dict__ creation
        - Mentions ~40% memory savings
        - Understands inheritance implications
        - Gives use case: "Many instances of same class"

---

### What is the Walrus Operator `:=`? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Python 3.8+`, `Assignment Expression`, `Syntax` | **Asked by:** Google, Meta, Amazon, Microsoft

??? success "View Answer"

    **What is the Walrus Operator?**
    
    The walrus operator (`:=`) assigns values as part of an expression (assignment expression), introduced in Python 3.8.
    
    ```python
    # Without walrus operator
    line = input()
    while line != 'quit':
        process(line)
        line = input()
    
    # With walrus operator
    while (line := input()) != 'quit':
        process(line)
    
    # List comprehension filtering
    # Without
    results = []
    for x in data:
        y = expensive_function(x)
        if y > threshold:
            results.append(y)
    
    # With walrus operator
    results = [y for x in data if (y := expensive_function(x)) > threshold]
    ```
    
    **Common Use Cases:**
    
    ```python
    # 1. Regex matching
    import re
    if (match := re.search(r'\d+', text)):
        print(f"Found: {match.group()}")
    
    # 2. File reading
    while (chunk := file.read(8192)):
        process(chunk)
    
    # 3. Dictionary get with check
    if (value := data.get('key')) is not None:
        print(f"Value: {value}")
    
    # 4. Avoiding repeated function calls
    if (n := len(items)) > 10:
        print(f"Too many items: {n}")
    ```
    
    **When NOT to Use:**
    
    ```python
    # ❌ Don't use for simple assignments
    x := 5  # SyntaxError (needs parentheses in statements)
    
    # ✅ Regular assignment is clearer
    x = 5
    
    # ❌ Don't sacrifice readability
    result = [(y := f(x), y**2) for x in data]  # Confusing
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Knowledge of modern Python features.
        
        **Strong answer signals:**
        
        - Knows it's Python 3.8+
        - Gives practical use case (while loop, comprehension)
        - Knows when NOT to use (simple assignments)
        - Mentions readability concerns

---

### What Are Metaclasses in Python? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Metaprogramming`, `OOP`, `Advanced` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **What Are Metaclasses?**
    
    A metaclass is the class of a class. Just as a class defines how instances behave, a metaclass defines how classes behave.
    
    ```python
    # Everything is an object
    class MyClass:
        pass
    
    obj = MyClass()
    
    print(type(obj))       # <class '__main__.MyClass'>
    print(type(MyClass))   # <class 'type'>
    
    # 'type' is the default metaclass
    # MyClass is an instance of 'type'
    ```
    
    **Creating a Metaclass:**
    
    ```python
    class Meta(type):
        def __new__(mcs, name, bases, namespace):
            # Called when class is created
            print(f"Creating class: {name}")
            
            # Add methods or modify class
            namespace['created_by'] = 'Meta'
            
            return super().__new__(mcs, name, bases, namespace)
        
        def __init__(cls, name, bases, namespace):
            # Called after class is created
            super().__init__(name, bases, namespace)
    
    class MyClass(metaclass=Meta):
        pass
    
    print(MyClass.created_by)  # 'Meta'
    ```
    
    **Practical Use Cases:**
    
    ```python
    # 1. Singleton pattern
    class SingletonMeta(type):
        _instances = {}
        
        def __call__(cls, *args, **kwargs):
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]
    
    class Database(metaclass=SingletonMeta):
        pass
    
    # 2. Registry pattern (like Django models)
    class ModelMeta(type):
        registry = {}
        
        def __new__(mcs, name, bases, namespace):
            cls = super().__new__(mcs, name, bases, namespace)
            if name != 'Model':
                mcs.registry[name] = cls
            return cls
    
    class Model(metaclass=ModelMeta):
        pass
    
    class User(Model):
        pass
    
    print(ModelMeta.registry)  # {'User': <class 'User'>}
    ```
    
    **When to Use:**
    
    | Use Metaclass | Use Decorator/Inheritance |
    |---------------|---------------------------|
    | Modify class creation | Modify instance behavior |
    | Class registry | Add methods |
    | API enforcement | Simple transformations |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Deep Python understanding.
        
        **Strong answer signals:**
        
        - Knows "type is the metaclass of all classes"
        - Can implement singleton with metaclass
        - Mentions `__new__` vs `__init__` at metaclass level
        - Knows: "If you're not sure you need it, you don't"

---

### Explain Python Descriptors - What Are `__get__`, `__set__`, `__delete__`? - Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Descriptors`, `OOP`, `Advanced` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **What Are Descriptors?**
    
    Descriptors are objects that define `__get__`, `__set__`, or `__delete__` methods. They customize attribute access on classes.
    
    ```python
    class Descriptor:
        def __get__(self, obj, objtype=None):
            print(f"Getting from {obj}")
            return self.value
        
        def __set__(self, obj, value):
            print(f"Setting {value} on {obj}")
            self.value = value
        
        def __delete__(self, obj):
            print(f"Deleting from {obj}")
            del self.value
    
    class MyClass:
        attr = Descriptor()  # Descriptor instance as class attribute
    
    obj = MyClass()
    obj.attr = 10       # Setting 10 on <MyClass object>
    print(obj.attr)     # Getting from <MyClass object> -> 10
    del obj.attr        # Deleting from <MyClass object>
    ```
    
    **Practical Example - Validation:**
    
    ```python
    class Validated:
        def __init__(self, min_value=None, max_value=None):
            self.min_value = min_value
            self.max_value = max_value
            self.name = None
        
        def __set_name__(self, owner, name):
            self.name = name
        
        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            return obj.__dict__.get(self.name)
        
        def __set__(self, obj, value):
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"{self.name} must be >= {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"{self.name} must be <= {self.max_value}")
            obj.__dict__[self.name] = value
    
    class Order:
        quantity = Validated(min_value=1, max_value=1000)
        price = Validated(min_value=0)
    
    order = Order()
    order.quantity = 50   # OK
    order.quantity = 0    # ValueError: quantity must be >= 1
    ```
    
    **Built-in Descriptors:**
    
    | Decorator | Uses Descriptors |
    |-----------|------------------|
    | `@property` | Data descriptor |
    | `@classmethod` | Non-data descriptor |
    | `@staticmethod` | Non-data descriptor |
    | `__slots__` | Data descriptor |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of Python's attribute access protocol.
        
        **Strong answer signals:**
        
        - Knows descriptors power `@property`
        - Data (has `__set__`) vs non-data (only `__get__`) descriptors
        - Uses `__set_name__` for automatic name binding
        - Gives practical use case: validation, lazy loading, caching

---

### Explain Threading vs Multiprocessing in Python - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Concurrency`, `Threading`, `Multiprocessing` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **Threading vs Multiprocessing:**
    
    | Aspect | Threading | Multiprocessing |
    |--------|-----------|-----------------|
    | GIL | Shared (blocked) | Separate per process |
    | Memory | Shared | Separate (IPC needed) |
    | Overhead | Light | Heavy (process spawn) |
    | Best for | I/O-bound | CPU-bound |
    | Debugging | Harder (race conditions) | Easier (isolated) |
    
    ```python
    import threading
    import multiprocessing
    import time
    
    def cpu_task(n):
        """CPU-intensive task"""
        return sum(i*i for i in range(n))
    
    # Threading (limited by GIL for CPU)
    def thread_example():
        threads = []
        start = time.time()
        for _ in range(4):
            t = threading.Thread(target=cpu_task, args=(10**7,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        print(f"Threading: {time.time() - start:.2f}s")  # ~4s
    
    # Multiprocessing (true parallelism)
    def process_example():
        start = time.time()
        with multiprocessing.Pool(4) as pool:
            results = pool.map(cpu_task, [10**7]*4)
        print(f"Multiprocessing: {time.time() - start:.2f}s")  # ~1s
    ```
    
    **Communication Between Processes:**
    
    ```python
    from multiprocessing import Queue, Pipe, Manager
    
    # Queue
    def worker(q):
        q.put("result")
    
    q = Queue()
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()
    print(q.get())  # "result"
    p.join()
    
    # Manager for shared state
    with Manager() as manager:
        shared_dict = manager.dict()
        shared_list = manager.list()
    ```
    
    **concurrent.futures (High-level API):**
    
    ```python
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    
    # Thread pool for I/O
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(io_task, urls))
    
    # Process pool for CPU
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cpu_task, data))
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of Python's concurrency limitations.
        
        **Strong answer signals:**
        
        - Immediately connects to GIL discussion
        - Knows threading for I/O, multiprocessing for CPU
        - Uses concurrent.futures as preferred API
        - Mentions IPC overhead for multiprocessing

---

### How Do You Profile and Optimize Python Code? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Performance`, `Profiling`, `Optimization` | **Asked by:** Amazon, Google, Meta, Netflix

??? success "View Answer"

    **Profiling Tools:**
    
    | Tool | Purpose |
    |------|---------|
    | `time` | Basic timing |
    | `cProfile` | Function-level profiling |
    | `line_profiler` | Line-by-line timing |
    | `memory_profiler` | Memory usage |
    | `py-spy` | Sampling profiler (production) |
    
    ```python
    # 1. Basic timing
    import time
    start = time.perf_counter()
    result = expensive_function()
    elapsed = time.perf_counter() - start
    
    # 2. timeit for benchmarking
    import timeit
    timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
    
    # 3. cProfile
    import cProfile
    cProfile.run('main()', sort='cumulative')
    
    # Or as decorator
    def profile(func):
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            result = profiler.runcall(func, *args, **kwargs)
            profiler.print_stats(sort='cumulative')
            return result
        return wrapper
    
    # 4. line_profiler (pip install line_profiler)
    @profile  # Add decorator, run with kernprof -l script.py
    def slow_function():
        result = []
        for i in range(10000):
            result.append(i**2)
        return result
    ```
    
    **Common Optimizations:**
    
    ```python
    # 1. Use built-ins
    # ❌ Slow
    result = []
    for x in data:
        result.append(x.upper())
    
    # ✅ Fast
    result = [x.upper() for x in data]
    
    # 2. Local variables (faster than global)
    def process(items):
        append = result.append  # Cache method lookup
        for item in items:
            append(item)
    
    # 3. Use appropriate data structures
    # ❌ O(n) lookup
    if item in large_list:
        pass
    
    # ✅ O(1) lookup
    large_set = set(large_list)
    if item in large_set:
        pass
    
    # 4. NumPy for numerical operations
    # ❌ Pure Python
    result = [x**2 for x in range(1000000)]
    
    # ✅ NumPy (100x faster)
    import numpy as np
    result = np.arange(1000000)**2
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Performance engineering skills.
        
        **Strong answer signals:**
        
        - Uses cProfile first, then line_profiler
        - Knows "profile first, then optimize"
        - Mentions algorithmic optimizations before micro-optimizations
        - Suggests NumPy/Cython for heavy computation

---

### How Do You Write Unit Tests in Python? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Testing`, `pytest`, `unittest` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **Testing Frameworks:**
    
    | Framework | Pros |
    |-----------|------|
    | `pytest` | Simple, powerful, most popular |
    | `unittest` | Built-in, Java-like |
    | `doctest` | Tests in docstrings |
    
    **pytest Example:**
    
    ```python
    # calculator.py
    def add(a, b):
        return a + b
    
    def divide(a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    # test_calculator.py
    import pytest
    from calculator import add, divide
    
    def test_add():
        assert add(2, 3) == 5
        assert add(-1, 1) == 0
    
    def test_divide():
        assert divide(10, 2) == 5
    
    def test_divide_by_zero():
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10, 0)
    
    # Parametrized tests
    @pytest.mark.parametrize("a,b,expected", [
        (2, 3, 5),
        (-1, 1, 0),
        (0, 0, 0),
    ])
    def test_add_parametrized(a, b, expected):
        assert add(a, b) == expected
    
    # Fixtures
    @pytest.fixture
    def sample_data():
        return {"users": [1, 2, 3]}
    
    def test_with_fixture(sample_data):
        assert len(sample_data["users"]) == 3
    ```
    
    **Mocking:**
    
    ```python
    from unittest.mock import Mock, patch, MagicMock
    
    # Mock external API
    @patch('module.requests.get')
    def test_api_call(mock_get):
        mock_get.return_value.json.return_value = {"key": "value"}
        result = fetch_data()
        assert result == {"key": "value"}
    
    # Mock class method
    with patch.object(MyClass, 'method', return_value=42):
        obj = MyClass()
        assert obj.method() == 42
    ```
    
    **Running Tests:**
    
    ```bash
    pytest                     # All tests
    pytest test_file.py        # Specific file
    pytest -k "test_add"       # Match pattern
    pytest -v                  # Verbose
    pytest --cov=mymodule     # Coverage
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Software engineering practices.
        
        **Strong answer signals:**
        
        - Uses pytest over unittest
        - Knows fixtures and parametrization
        - Uses mocking for external dependencies
        - Mentions test coverage

---

### How Do You Handle Files in Python? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `File I/O`, `Context Managers`, `Basics` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Basic File Operations:**
    
    ```python
    # ✅ Always use context manager
    with open('file.txt', 'r') as f:
        content = f.read()
    # File automatically closed
    
    # Modes
    # 'r' - read (default)
    # 'w' - write (overwrite)
    # 'a' - append
    # 'x' - exclusive create (fail if exists)
    # 'b' - binary mode
    # '+' - read and write
    
    # Reading methods
    with open('file.txt') as f:
        content = f.read()         # Entire file as string
        lines = f.readlines()      # List of lines
        for line in f:             # Iterate (memory efficient)
            process(line)
    
    # Writing
    with open('output.txt', 'w') as f:
        f.write("Hello\n")
        f.writelines(["line1\n", "line2\n"])
    ```
    
    **Working with Paths (pathlib):**
    
    ```python
    from pathlib import Path
    
    # Modern way to handle paths
    path = Path('data') / 'file.txt'
    
    # Check existence
    if path.exists():
        content = path.read_text()
    
    # Write
    path.write_text("content")
    
    # List directory
    for file in Path('.').glob('*.py'):
        print(file.name)
    
    # Common operations
    path.parent      # Parent directory
    path.stem        # Filename without extension
    path.suffix      # Extension (.txt)
    path.is_file()   # Is it a file?
    path.mkdir(parents=True, exist_ok=True)  # Create directory
    ```
    
    **Binary Files and JSON:**
    
    ```python
    import json
    
    # JSON
    with open('data.json', 'r') as f:
        data = json.load(f)
    
    with open('output.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    # Binary
    with open('image.png', 'rb') as f:
        binary_data = f.read()
    
    # CSV
    import csv
    with open('data.csv', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row['column_name'])
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Fundamental I/O skills.
        
        **Strong answer signals:**
        
        - Always uses context managers (`with`)
        - Uses `pathlib` for paths (not os.path)
        - Knows encoding: `open('f.txt', encoding='utf-8')`
        - Mentions streaming for large files

---

### What Are Virtual Environments and How Do You Manage Dependencies? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Virtual Environments`, `Dependencies`, `Best Practices` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **What Are Virtual Environments?**
    
    Isolated Python environments that have their own packages, avoiding conflicts between projects.
    
    **Creating and Using Virtual Environments:**
    
    ```bash
    # Built-in venv
    python -m venv myenv
    source myenv/bin/activate  # Linux/Mac
    myenv\Scripts\activate     # Windows
    
    deactivate  # Exit virtual environment
    
    # With pip
    pip install requests
    pip freeze > requirements.txt
    pip install -r requirements.txt
    ```
    
    **Modern Tools:**
    
    | Tool | Purpose |
    |------|---------|
    | `venv` | Built-in, basic |
    | `virtualenv` | More features |
    | `conda` | Data science, binary packages |
    | `poetry` | Dependency + packaging |
    | `pipenv` | Pip + venv combined |
    | `uv` | Fast, Rust-based (new) |
    
    **Poetry Example:**
    
    ```bash
    # Initialize project
    poetry init
    
    # Add dependencies
    poetry add requests
    poetry add --group dev pytest
    
    # Install from lock file
    poetry install
    
    # Run in virtual environment
    poetry run python script.py
    poetry shell  # Activate shell
    ```
    
    **pyproject.toml:**
    
    ```toml
    [tool.poetry]
    name = "myproject"
    version = "0.1.0"
    
    [tool.poetry.dependencies]
    python = "^3.10"
    requests = "^2.28"
    
    [tool.poetry.group.dev.dependencies]
    pytest = "^7.0"
    black = "^23.0"
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Development workflow knowledge.
        
        **Strong answer signals:**
        
        - Uses virtual environments for every project
        - Knows requirements.txt vs lock files
        - Mentions Poetry or modern tooling
        - Understands reproducible builds

---

### Explain Python's Logging Module - How Do You Use It Effectively? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Logging`, `Debugging`, `Best Practices` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Logging Levels:**
    
    | Level | When to Use |
    |-------|-------------|
    | DEBUG | Detailed diagnostic info |
    | INFO | Confirm things work |
    | WARNING | Unexpected but handled |
    | ERROR | Serious problem |
    | CRITICAL | Program may crash |
    
    **Basic Usage:**
    
    ```python
    import logging
    
    # Basic configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()  # Console
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    logger.debug("Debugging info")
    logger.info("User logged in")
    logger.warning("Disk space low")
    logger.error("Failed to connect")
    logger.exception("Error with traceback")  # Includes stack trace
    ```
    
    **Structured Logging:**
    
    ```python
    import logging
    import json
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                'time': self.formatTime(record),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
            })
    
    # For production, use structured logging libraries
    # Example: structlog, python-json-logger
    ```
    
    **Best Practices:**
    
    ```python
    # ✅ Use __name__ for logger
    logger = logging.getLogger(__name__)
    
    # ✅ Use lazy formatting
    logger.info("User %s logged in at %s", username, time)
    
    # ❌ Avoid string formatting in log call
    logger.info(f"User {username} logged in")  # String built even if not logged
    
    # ✅ Use extra for structured data
    logger.info("Order placed", extra={'order_id': 123, 'amount': 99.99})
    
    # ✅ Don't use print() in production
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Production code practices.
        
        **Strong answer signals:**
        
        - Uses `__name__` for hierarchical loggers
        - Knows lazy formatting for performance
        - Mentions structured logging for production
        - Configures different handlers (file, console, remote)

---

### What Are Regular Expressions in Python? - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Regex`, `Text Processing`, `Pattern Matching` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Basic Regex Operations:**
    
    ```python
    import re
    
    text = "Contact us at support@example.com or sales@company.org"
    
    # match() - Start of string only
    result = re.match(r'Contact', text)
    
    # search() - Find first occurrence
    result = re.search(r'\w+@\w+\.\w+', text)
    if result:
        print(result.group())  # support@example.com
    
    # findall() - Find all occurrences
    emails = re.findall(r'\w+@\w+\.\w+', text)
    print(emails)  # ['support@example.com', 'sales@company.org']
    
    # sub() - Replace
    new_text = re.sub(r'\w+@\w+\.\w+', '[EMAIL]', text)
    
    # split() - Split by pattern
    parts = re.split(r'\s+', text)
    ```
    
    **Common Patterns:**
    
    ```python
    # Email
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    
    # Phone (US)
    phone_pattern = r'\d{3}[-.]?\d{3}[-.]?\d{4}'
    
    # URL
    url_pattern = r'https?://[\w\.-]+(?:/[\w\.-]*)*'
    
    # Date (YYYY-MM-DD)
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    
    # IP Address
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    ```
    
    **Groups and Named Groups:**
    
    ```python
    # Capturing groups
    pattern = r'(\w+)@(\w+)\.(\w+)'
    match = re.search(pattern, 'user@example.com')
    print(match.groups())  # ('user', 'example', 'com')
    print(match.group(1))  # 'user'
    
    # Named groups
    pattern = r'(?P<user>\w+)@(?P<domain>\w+)\.(?P<tld>\w+)'
    match = re.search(pattern, 'user@example.com')
    print(match.group('user'))  # 'user'
    print(match.groupdict())  # {'user': 'user', 'domain': 'example', 'tld': 'com'}
    
    # Non-capturing group
    pattern = r'(?:https?://)?www\.example\.com'
    ```
    
    **Compile for Performance:**
    
    ```python
    # Compile pattern for reuse
    pattern = re.compile(r'\w+@\w+\.\w+', re.IGNORECASE)
    
    # Reuse compiled pattern
    emails = pattern.findall(text)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Text processing skills.
        
        **Strong answer signals:**
        
        - Knows difference between `match()` and `search()`
        - Uses named groups for readability
        - Compiles patterns for repeated use
        - Knows raw strings `r''` prevent escape issues

---

### What is `__name__ == "__main__"`? Why Use It? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Modules`, `Entry Point`, `Best Practices` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **What Does It Mean?**
    
    `__name__` is a special variable set by Python:
    - `"__main__"` when the file is run directly
    - Module name when imported
    
    ```python
    # my_module.py
    def main():
        print("Running main function")
    
    def utility_function():
        return "I'm a utility"
    
    print(f"__name__ is: {__name__}")
    
    if __name__ == "__main__":
        main()
    ```
    
    ```bash
    # Run directly
    python my_module.py
    # Output:
    # __name__ is: __main__
    # Running main function
    
    # Import
    python -c "import my_module"
    # Output:
    # __name__ is: my_module
    # (main() not called)
    ```
    
    **Why It Matters:**
    
    ```python
    # ❌ Without guard - runs on import
    import expensive_module  # Starts computation/server/etc.
    
    # ✅ With guard - only runs when executed directly
    if __name__ == "__main__":
        start_server()
        run_tests()
    ```
    
    **Best Practice Structure:**
    
    ```python
    #!/usr/bin/env python3
    """Module docstring."""
    
    import sys
    from typing import List
    
    def main(args: List[str]) -> int:
        """Main entry point."""
        # Your code here
        return 0
    
    if __name__ == "__main__":
        sys.exit(main(sys.argv[1:]))
    ```
    
    **Use Cases:**
    
    | With Guard | Without Guard |
    |------------|---------------|
    | CLI scripts | Library modules (pure functions) |
    | One-off tests | Constant definitions |
    | Server startup | Class definitions |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Basic Python module understanding.
        
        **Strong answer signals:**
        
        - Explains the dual nature of Python files (script vs module)
        - Knows it prevents code execution on import
        - Uses `sys.exit(main())` pattern for CLI tools
        - Mentions testability: "Can import and test functions separately"

---


### What is `itertools`? How Do You Use It? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Iterators`, `Functional`, `Standard Library` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **What is itertools?**
    
    The `itertools` module provides memory-efficient tools for working with iterators.
    
    **Infinite Iterators:**
    
    ```python
    from itertools import count, cycle, repeat
    
    # count(start, step) - infinite counter
    for i in count(10, 2):  # 10, 12, 14, ...
        if i > 20:
            break
        print(i)
    
    # cycle(iterable) - repeat infinitely
    colors = cycle(['red', 'green', 'blue'])
    for _ in range(6):
        print(next(colors))
    
    # repeat(elem, n) - repeat n times
    list(repeat('A', 3))  # ['A', 'A', 'A']
    ```
    
    **Combinatoric Functions:**
    
    ```python
    from itertools import permutations, combinations, product
    
    # permutations - ordered arrangements
    list(permutations('ABC', 2))  # [('A','B'), ('A','C'), ('B','A'), ...]
    
    # combinations - unordered selections
    list(combinations('ABC', 2))  # [('A','B'), ('A','C'), ('B','C')]
    
    # product - cartesian product
    list(product([1,2], ['a','b']))  # [(1,'a'), (1,'b'), (2,'a'), (2,'b')]
    ```
    
    **Practical Functions:**
    
    ```python
    from itertools import chain, groupby, islice, accumulate
    
    # chain - flatten iterables
    list(chain([1,2], [3,4]))  # [1, 2, 3, 4]
    
    # groupby - group consecutive elements
    data = [('a', 1), ('a', 2), ('b', 3)]
    for key, group in groupby(data, key=lambda x: x[0]):
        print(key, list(group))
    
    # islice - slice iterator
    list(islice(range(100), 5, 10))  # [5, 6, 7, 8, 9]
    
    # accumulate - running totals
    list(accumulate([1,2,3,4]))  # [1, 3, 6, 10]
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Efficient iteration patterns.
        
        **Strong answer signals:**
        
        - Knows itertools is memory-efficient (no list creation)
        - Uses chain.from_iterable for nested iteration
        - Knows groupby requires sorted input for full grouping
        - Mentions more_itertools for extended functionality

---

### What is `functools`? Explain `lru_cache` and `partial` - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Functional Programming`, `Caching`, `Standard Library` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **functools Module:**
    
    Higher-order functions and operations on callable objects.
    
    **lru_cache - Memoization:**
    
    ```python
    from functools import lru_cache
    
    # Cache function results
    @lru_cache(maxsize=128)
    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    fibonacci(100)  # Computed instantly with cache
    
    # Python 3.9+ cache (no size limit)
    from functools import cache
    
    @cache
    def expensive_computation(x, y):
        return x ** y
    
    # Cache info
    print(fibonacci.cache_info())
    fibonacci.cache_clear()  # Clear cache
    ```
    
    **partial - Partial Function Application:**
    
    ```python
    from functools import partial
    
    def power(base, exponent):
        return base ** exponent
    
    square = partial(power, exponent=2)
    cube = partial(power, exponent=3)
    
    print(square(4))  # 16
    print(cube(4))    # 64
    
    # Useful with map
    def multiply(x, y):
        return x * y
    
    double = partial(multiply, 2)
    list(map(double, [1, 2, 3]))  # [2, 4, 6]
    ```
    
    **Other functools:**
    
    ```python
    from functools import reduce, wraps, total_ordering
    
    # reduce - fold/accumulate
    reduce(lambda x, y: x + y, [1, 2, 3, 4])  # 10
    
    # wraps - preserve metadata in decorators
    def my_decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper
    
    # total_ordering - auto-generate comparison methods
    @total_ordering
    class Person:
        def __init__(self, name):
            self.name = name
        def __eq__(self, other):
            return self.name == other.name
        def __lt__(self, other):
            return self.name < other.name
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Functional programming knowledge.
        
        **Strong answer signals:**
        
        - Uses lru_cache for expensive computations
        - Knows lru_cache only works with hashable arguments
        - Uses partial to create specialized functions
        - Mentions @wraps for proper decorator metadata

---

### What is the `collections` Module? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Structures`, `Collections`, `Standard Library` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **collections Module:**
    
    Specialized container datatypes beyond built-in dict, list, set, tuple.
    
    **defaultdict:**
    
    ```python
    from collections import defaultdict
    
    # Auto-create missing keys
    word_count = defaultdict(int)
    for word in ['apple', 'banana', 'apple']:
        word_count[word] += 1  # No KeyError
    
    # Grouping
    groups = defaultdict(list)
    for item in [('a', 1), ('b', 2), ('a', 3)]:
        groups[item[0]].append(item[1])
    print(groups)  # {'a': [1, 3], 'b': [2]}
    ```
    
    **Counter:**
    
    ```python
    from collections import Counter
    
    # Count elements
    c = Counter(['a', 'b', 'a', 'c', 'a'])
    print(c)  # Counter({'a': 3, 'b': 1, 'c': 1})
    print(c.most_common(2))  # [('a', 3), ('b', 1)]
    
    # Operations
    c1 = Counter('aab')
    c2 = Counter('abc')
    c1 + c2  # Counter({'a': 3, 'b': 2, 'c': 1})
    c1 - c2  # Counter({'a': 1})
    ```
    
    **deque (Double-ended queue):**
    
    ```python
    from collections import deque
    
    d = deque([1, 2, 3])
    d.appendleft(0)   # O(1) prepend
    d.popleft()       # O(1) pop from front
    d.rotate(1)       # Rotate right
    
    # Max length (automatic removal)
    d = deque(maxlen=3)
    d.extend([1, 2, 3, 4])  # deque([2, 3, 4])
    ```
    
    **namedtuple:**
    
    ```python
    from collections import namedtuple
    
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(3, 4)
    print(p.x, p.y)  # 3 4
    print(p[0])      # 3 (tuple access)
    
    # With defaults (Python 3.7+)
    Point = namedtuple('Point', ['x', 'y'], defaults=[0])
    Point(1)  # Point(x=1, y=0)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data structure knowledge.
        
        **Strong answer signals:**
        
        - Uses defaultdict to avoid key checking
        - Knows Counter for frequency counting
        - Uses deque for O(1) operations on both ends
        - Mentions OrderedDict is redundant in Python 3.7+

---

### How Do You Serialize Python Objects? Explain `pickle` and `json` - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Serialization`, `Persistence`, `Data Exchange` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Serialization Options:**
    
    | Format | Use Case | Human Readable | Speed |
    |--------|----------|----------------|-------|
    | JSON | API, config | Yes | Fast |
    | pickle | Python objects | No | Faster |
    | YAML | Config files | Yes | Slow |
    | MessagePack | Binary JSON | No | Very fast |
    
    **JSON:**
    
    ```python
    import json
    
    data = {'name': 'Alice', 'scores': [95, 87, 91]}
    
    # Serialize
    json_str = json.dumps(data, indent=2)
    with open('data.json', 'w') as f:
        json.dump(data, f)
    
    # Deserialize
    parsed = json.loads(json_str)
    with open('data.json') as f:
        loaded = json.load(f)
    
    # Custom encoder
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
    
    json.dumps(data, cls=CustomEncoder)
    ```
    
    **pickle:**
    
    ```python
    import pickle
    
    # Can serialize any Python object
    model = train_model()  # sklearn model
    
    # Save
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Load
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # ⚠️ Security warning
    # Never unpickle untrusted data - arbitrary code execution!
    ```
    
    **When to Use:**
    
    | Use JSON | Use pickle |
    |----------|------------|
    | Cross-language | Python-only |
    | API responses | ML models |
    | Config files | Complex objects |
    | Security matters | Trusted source |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data persistence knowledge.
        
        **Strong answer signals:**
        
        - Warns about pickle security risks
        - Uses json for interoperability
        - Knows pickle preserves Python types
        - Mentions joblib for numpy/sklearn objects

---

### What is `subprocess`? How Do You Run Shell Commands? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `System`, `Shell`, `Process Management` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **subprocess Module:**
    
    Run external commands and capture output.
    
    **Basic Usage:**
    
    ```python
    import subprocess
    
    # Simple command (Python 3.5+)
    result = subprocess.run(['ls', '-la'], capture_output=True, text=True)
    print(result.stdout)
    print(result.returncode)  # 0 = success
    
    # With shell=True (careful with user input!)
    result = subprocess.run('ls -la | grep .py', shell=True, capture_output=True, text=True)
    
    # Check for errors
    result = subprocess.run(['ls', 'nonexistent'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    
    # Raise exception on failure
    subprocess.run(['false'], check=True)  # Raises CalledProcessError
    ```
    
    **Advanced Usage:**
    
    ```python
    # With timeout
    try:
        result = subprocess.run(['sleep', '10'], timeout=5)
    except subprocess.TimeoutExpired:
        print("Command timed out")
    
    # Piping between processes
    p1 = subprocess.Popen(['cat', 'file.txt'], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['grep', 'pattern'], stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()
    output = p2.communicate()[0]
    
    # Pass input
    result = subprocess.run(['cat'], input='hello', text=True, capture_output=True)
    ```
    
    **Best Practices:**
    
    ```python
    # ✅ Use list of arguments (safe)
    subprocess.run(['ls', '-la', directory])
    
    # ❌ Avoid shell=True with user input
    # Vulnerable to injection!
    subprocess.run(f'ls {user_input}', shell=True)  # DANGEROUS
    
    # ✅ Use shlex for complex commands
    import shlex
    cmd = shlex.split('ls -la "my file.txt"')
    subprocess.run(cmd)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** System integration skills.
        
        **Strong answer signals:**
        
        - Uses run() over deprecated os.system()
        - Warns about shell=True security risks
        - Knows capture_output=True shortcut
        - Handles timeout and errors properly

---

### What is `argparse`? How Do You Build CLI Tools? - Amazon, Google Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `CLI`, `Arguments`, `Tool Building` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **argparse for CLI:**
    
    ```python
    import argparse
    
    def main():
        parser = argparse.ArgumentParser(
            description='Process some files.'
        )
        
        # Positional argument
        parser.add_argument('filename', help='File to process')
        
        # Optional arguments
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Enable verbose output')
        parser.add_argument('-n', '--count', type=int, default=10,
                            help='Number of items (default: 10)')
        parser.add_argument('-o', '--output', required=True,
                            help='Output file (required)')
        
        # Choices
        parser.add_argument('--format', choices=['json', 'csv', 'xml'],
                            default='json', help='Output format')
        
        # Multiple values
        parser.add_argument('--files', nargs='+', help='Multiple files')
        
        args = parser.parse_args()
        
        print(f"Processing {args.filename}")
        print(f"Verbose: {args.verbose}")
        print(f"Count: {args.count}")
        print(f"Output: {args.output}")
    
    if __name__ == '__main__':
        main()
    ```
    
    **Usage:**
    
    ```bash
    python script.py input.txt -o output.json -v -n 20
    python script.py --help
    ```
    
    **Subcommands:**
    
    ```python
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    # git add
    add_parser = subparsers.add_parser('add', help='Add files')
    add_parser.add_argument('files', nargs='+')
    
    # git commit
    commit_parser = subparsers.add_parser('commit', help='Commit changes')
    commit_parser.add_argument('-m', '--message', required=True)
    
    args = parser.parse_args()
    if args.command == 'add':
        print(f"Adding: {args.files}")
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical tool building.
        
        **Strong answer signals:**
        
        - Uses type= for automatic conversion
        - Knows action='store_true' for flags
        - Uses subparsers for git-like commands
        - Mentions click or typer as modern alternatives

---

### How Do You Work with Datetime in Python? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Datetime`, `Timezone`, `Time Handling` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Basic datetime:**
    
    ```python
    from datetime import datetime, date, time, timedelta
    
    # Current time
    now = datetime.now()
    today = date.today()
    
    # Create specific date/time
    dt = datetime(2024, 12, 25, 10, 30, 0)
    d = date(2024, 12, 25)
    t = time(10, 30, 0)
    
    # Parsing strings
    dt = datetime.strptime('2024-12-25', '%Y-%m-%d')
    dt = datetime.fromisoformat('2024-12-25T10:30:00')
    
    # Formatting
    dt.strftime('%B %d, %Y')  # 'December 25, 2024'
    dt.isoformat()  # '2024-12-25T10:30:00'
    ```
    
    **Date Arithmetic:**
    
    ```python
    from datetime import timedelta
    
    now = datetime.now()
    
    # Add/subtract time
    tomorrow = now + timedelta(days=1)
    last_week = now - timedelta(weeks=1)
    in_2_hours = now + timedelta(hours=2)
    
    # Difference between dates
    delta = datetime(2024, 12, 31) - datetime(2024, 1, 1)
    print(delta.days)  # 365
    print(delta.total_seconds())
    ```
    
    **Timezones (Python 3.9+):**
    
    ```python
    from datetime import timezone
    from zoneinfo import ZoneInfo
    
    # UTC
    utc_now = datetime.now(timezone.utc)
    
    # Specific timezone
    ny = ZoneInfo('America/New_York')
    la = ZoneInfo('America/Los_Angeles')
    
    ny_time = datetime.now(ny)
    la_time = ny_time.astimezone(la)
    
    # Convert naive to aware
    naive = datetime.now()
    aware = naive.replace(tzinfo=ZoneInfo('UTC'))
    ```
    
    **Common Patterns:**
    
    ```python
    # Start/end of day
    start_of_day = datetime.combine(date.today(), time.min)
    end_of_day = datetime.combine(date.today(), time.max)
    
    # First day of month
    first_of_month = date.today().replace(day=1)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Time handling skills.
        
        **Strong answer signals:**
        
        - Uses zoneinfo for timezones (not pytz in Python 3.9+)
        - Knows strftime format codes
        - Always stores UTC, displays local
        - Uses isoformat for serialization

---

### What is the Difference Between `copy` and `deepcopy`? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Copying`, `References`, `Memory` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **copy vs deepcopy:**
    
    | Type | What It Does |
    |------|--------------|
    | Assignment (=) | Same object, new reference |
    | Shallow copy | New object, same nested references |
    | Deep copy | New object, recursive copies |
    
    ```python
    import copy
    
    original = [[1, 2, 3], [4, 5, 6]]
    
    # Assignment - same object
    assigned = original
    assigned[0][0] = 'X'
    print(original)  # [['X', 2, 3], [4, 5, 6]] - Modified!
    
    # Shallow copy - new outer list, same inner lists
    original = [[1, 2, 3], [4, 5, 6]]
    shallow = copy.copy(original)
    # or shallow = original[:]
    # or shallow = list(original)
    
    shallow[0][0] = 'X'
    print(original)  # [['X', 2, 3], [4, 5, 6]] - Inner modified!
    
    shallow.append([7, 8, 9])
    print(original)  # [['X', 2, 3], [4, 5, 6]] - Outer not modified
    
    # Deep copy - completely independent
    original = [[1, 2, 3], [4, 5, 6]]
    deep = copy.deepcopy(original)
    
    deep[0][0] = 'X'
    print(original)  # [[1, 2, 3], [4, 5, 6]] - Unchanged!
    ```
    
    **Visual:**
    
    ```
    Original:  [list1, list2]
                  |      |
    Shallow:   [list1, list2]  <- Same inner lists
    
    Deep:      [list1', list2']  <- New inner lists
    ```
    
    **Custom Objects:**
    
    ```python
    class Node:
        def __init__(self, value):
            self.value = value
            self.children = []
        
        def __copy__(self):
            # Custom shallow copy
            new = Node(self.value)
            new.children = self.children  # Same list
            return new
        
        def __deepcopy__(self, memo):
            # Custom deep copy
            new = Node(copy.deepcopy(self.value, memo))
            new.children = copy.deepcopy(self.children, memo)
            return new
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of object references.
        
        **Strong answer signals:**
        
        - Knows list[:] or list.copy() is shallow
        - Uses deepcopy for nested structures
        - Knows deepcopy handles circular references
        - Implements __copy__ and __deepcopy__ for custom classes

---

### What is `heapq`? How Do You Use Priority Queues? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Heap`, `Priority Queue`, `Algorithms` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **heapq Module:**
    
    Binary heap implementation providing priority queue functionality.
    
    ```python
    import heapq
    
    # Create heap (min-heap)
    heap = []
    heapq.heappush(heap, 3)
    heapq.heappush(heap, 1)
    heapq.heappush(heap, 4)
    heapq.heappush(heap, 1)
    
    print(heap)  # [1, 1, 4, 3] - heap property maintained
    
    # Pop smallest
    smallest = heapq.heappop(heap)  # 1
    
    # Peek without removing
    smallest = heap[0]
    
    # Heapify existing list
    data = [3, 1, 4, 1, 5]
    heapq.heapify(data)  # In-place, O(n)
    ```
    
    **Useful Operations:**
    
    ```python
    # N smallest/largest - O(n log k)
    nums = [3, 1, 4, 1, 5, 9, 2, 6]
    heapq.nsmallest(3, nums)  # [1, 1, 2]
    heapq.nlargest(3, nums)   # [9, 6, 5]
    
    # With key function
    data = [('a', 3), ('b', 1), ('c', 2)]
    heapq.nsmallest(2, data, key=lambda x: x[1])  # [('b', 1), ('c', 2)]
    
    # Push and pop in one operation
    heapq.heapreplace(heap, new_item)  # Pop then push
    heapq.heappushpop(heap, new_item)  # Push then pop
    ```
    
    **Max-Heap (workaround):**
    
    ```python
    # Python only has min-heap, negate values for max
    max_heap = []
    heapq.heappush(max_heap, -3)
    heapq.heappush(max_heap, -1)
    heapq.heappush(max_heap, -4)
    
    largest = -heapq.heappop(max_heap)  # 4
    ```
    
    **Priority Queue with Class:**
    
    ```python
    from dataclasses import dataclass, field
    from typing import Any
    
    @dataclass(order=True)
    class PrioritizedItem:
        priority: int
        item: Any = field(compare=False)
    
    pq = []
    heapq.heappush(pq, PrioritizedItem(2, 'task2'))
    heapq.heappush(pq, PrioritizedItem(1, 'task1'))
    
    heapq.heappop(pq).item  # 'task1'
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data structure knowledge.
        
        **Strong answer signals:**
        
        - Knows heapq is min-heap only
        - Uses nsmallest/nlargest for top-k problems
        - Knows O(log n) push/pop, O(n) heapify
        - Uses tuple (priority, data) for custom ordering

---

### What is `bisect`? How Do You Use Binary Search? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Binary Search`, `Sorted Lists`, `Algorithms` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **bisect Module:**
    
    Binary search operations on sorted lists.
    
    ```python
    import bisect
    
    sorted_list = [1, 3, 5, 7, 9]
    
    # Find insertion point
    bisect.bisect_left(sorted_list, 5)   # 2 - insert before existing
    bisect.bisect_right(sorted_list, 5)  # 3 - insert after existing
    bisect.bisect(sorted_list, 5)        # Same as bisect_right
    
    # Insert while maintaining order
    bisect.insort(sorted_list, 4)
    print(sorted_list)  # [1, 3, 4, 5, 7, 9]
    
    # insort_left vs insort_right for duplicates
    ```
    
    **Finding Exact Values:**
    
    ```python
    def binary_search(sorted_list, x):
        """Check if x exists in sorted list."""
        i = bisect.bisect_left(sorted_list, x)
        if i != len(sorted_list) and sorted_list[i] == x:
            return i
        return -1
    
    # Usage
    idx = binary_search([1, 3, 5, 7], 5)  # 2
    idx = binary_search([1, 3, 5, 7], 4)  # -1
    ```
    
    **With Key Function (Python 3.10+):**
    
    ```python
    # Binary search with key
    data = [('a', 1), ('b', 3), ('c', 5)]
    keys = [x[1] for x in data]  # Pre-computed keys
    
    idx = bisect.bisect_left(keys, 3)  # 1
    
    # Python 3.10+
    # bisect.bisect_left(data, 3, key=lambda x: x[1])
    ```
    
    **Applications:**
    
    ```python
    # Grade assignment
    def grade(score):
        breakpoints = [60, 70, 80, 90]
        grades = 'FDCBA'
        return grades[bisect.bisect(breakpoints, score)]
    
    grade(65)  # 'D'
    grade(85)  # 'B'
    
    # Maintaining sorted collection
    class SortedList:
        def __init__(self):
            self.data = []
        
        def add(self, x):
            bisect.insort(self.data, x)
        
        def find_closest(self, x):
            pos = bisect.bisect_left(self.data, x)
            if pos == len(self.data):
                return self.data[-1]
            if pos == 0:
                return self.data[0]
            before = self.data[pos - 1]
            after = self.data[pos]
            return before if x - before <= after - x else after
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Binary search application.
        
        **Strong answer signals:**
        
        - Knows bisect_left vs bisect_right difference
        - Uses bisect for O(log n) lookup in sorted lists
        - Can implement exact value search
        - Mentions SortedContainers for advanced use

---

### Explain `zip`, `enumerate`, `map`, and `filter` - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Built-ins`, `Functional`, `Iteration` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **zip - Combine Iterables:**
    
    ```python
    names = ['Alice', 'Bob', 'Charlie']
    scores = [95, 87, 91]
    
    # Pair elements
    for name, score in zip(names, scores):
        print(f"{name}: {score}")
    
    # Create dict
    name_score = dict(zip(names, scores))
    
    # Unzip
    pairs = [('a', 1), ('b', 2), ('c', 3)]
    letters, numbers = zip(*pairs)
    
    # Stops at shortest
    list(zip([1, 2, 3], ['a', 'b']))  # [(1, 'a'), (2, 'b')]
    
    # Longest with fill (Python 3.10+)
    from itertools import zip_longest
    list(zip_longest([1, 2, 3], ['a'], fillvalue=None))
    ```
    
    **enumerate - Index with Values:**
    
    ```python
    fruits = ['apple', 'banana', 'cherry']
    
    for i, fruit in enumerate(fruits):
        print(f"{i}: {fruit}")
    
    # Start index
    for i, fruit in enumerate(fruits, start=1):
        print(f"{i}: {fruit}")
    
    # Create indexed dict
    indexed = {i: v for i, v in enumerate(fruits)}
    ```
    
    **map - Transform Each Element:**
    
    ```python
    numbers = [1, 2, 3, 4]
    
    # Apply function to each
    squared = list(map(lambda x: x**2, numbers))
    
    # Multiple iterables
    list(map(lambda x, y: x + y, [1, 2], [10, 20]))  # [11, 22]
    
    # With named function
    list(map(str.upper, ['a', 'b', 'c']))  # ['A', 'B', 'C']
    ```
    
    **filter - Keep Matching Elements:**
    
    ```python
    numbers = [1, 2, 3, 4, 5, 6]
    
    # Keep evens
    evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4, 6]
    
    # Remove falsy values
    data = [0, 1, '', 'hello', None, True]
    truthy = list(filter(None, data))  # [1, 'hello', True]
    ```
    
    **List Comprehension Alternative:**
    
    ```python
    # Often prefer comprehensions for readability
    squared = [x**2 for x in numbers]  # vs map
    evens = [x for x in numbers if x % 2 == 0]  # vs filter
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Core Python iteration patterns.
        
        **Strong answer signals:**
        
        - Knows map/filter return iterators (lazy)
        - Uses enumerate instead of range(len())
        - Prefers comprehensions for simple cases
        - Uses zip_longest for mismatched lengths

---

### What Are Lambda Functions? When to Use Them? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Lambda`, `Functional`, `Anonymous Functions` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Lambda Syntax:**
    
    ```python
    # lambda arguments: expression
    square = lambda x: x ** 2
    add = lambda x, y: x + y
    
    print(square(5))   # 25
    print(add(2, 3))   # 5
    ```
    
    **Common Use Cases:**
    
    ```python
    # Sorting key
    data = [('bob', 3), ('alice', 2), ('charlie', 1)]
    sorted(data, key=lambda x: x[1])  # Sort by second element
    
    # max/min key
    max(data, key=lambda x: x[1])  # ('bob', 3)
    
    # With map/filter
    numbers = [1, 2, 3, 4]
    list(map(lambda x: x * 2, numbers))
    list(filter(lambda x: x > 2, numbers))
    
    # Default argument factory
    from collections import defaultdict
    d = defaultdict(lambda: 'unknown')
    ```
    
    **Limitations:**
    
    ```python
    # ❌ Single expression only (no statements)
    # lambda x: print(x); return x  # SyntaxError
    
    # ❌ No annotations
    # lambda x: int: x * 2  # SyntaxError
    
    # ❌ No docstrings
    ```
    
    **When NOT to Use:**
    
    ```python
    # ❌ Complex logic - use def
    bad = lambda x: x if x > 0 else -x if x < 0 else 0  # Hard to read
    
    # ✅ Use regular function
    def better(x):
        """Return absolute value."""
        if x > 0:
            return x
        elif x < 0:
            return -x
        else:
            return 0
    
    # ❌ Assigning to variable - use def
    double = lambda x: x * 2  # PEP 8 discourages this
    
    # ✅ Better
    def double(x):
        return x * 2
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Functional programming basics.
        
        **Strong answer signals:**
        
        - Uses lambda inline (sorting keys)
        - Knows limitations (single expression)
        - Uses def for reusable/documented functions
        - PEP 8: "Don't assign lambda to variable"

---

### What Are Closures in Python? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Closures`, `Scope`, `Functional` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **What is a Closure?**
    
    A closure is a function that remembers values from its enclosing scope even when called outside that scope.
    
    ```python
    def make_multiplier(n):
        def multiplier(x):
            return x * n  # 'n' is captured from outer scope
        return multiplier
    
    double = make_multiplier(2)
    triple = make_multiplier(3)
    
    print(double(5))  # 10
    print(triple(5))  # 15
    print(double.__closure__[0].cell_contents)  # 2
    ```
    
    **Closure Requirements:**
    
    1. Nested function
    2. Inner function references outer variables
    3. Outer function returns inner function
    
    **Common Use Cases:**
    
    ```python
    # Function factory
    def make_counter():
        count = 0
        def counter():
            nonlocal count
            count += 1
            return count
        return counter
    
    c = make_counter()
    print(c())  # 1
    print(c())  # 2
    
    # Data hiding
    def make_bank_account(initial):
        balance = initial
        
        def deposit(amount):
            nonlocal balance
            balance += amount
            return balance
        
        def withdraw(amount):
            nonlocal balance
            if amount <= balance:
                balance -= amount
            return balance
        
        def get_balance():
            return balance
        
        return deposit, withdraw, get_balance
    ```
    
    **The nonlocal Keyword:**
    
    ```python
    def outer():
        x = 10
        
        def inner():
            nonlocal x  # Required to modify outer variable
            x = 20
        
        inner()
        print(x)  # 20
    
    # Without nonlocal, creates new local variable
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of scope and functions.
        
        **Strong answer signals:**
        
        - Explains how variables are captured
        - Uses nonlocal for modification
        - Knows closures enable state without classes
        - Can inspect __closure__ attribute

---

### What is the `@property` Decorator? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Properties`, `OOP`, `Encapsulation` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **@property Decorator:**
    
    Define getters/setters with attribute-like syntax.
    
    ```python
    class Temperature:
        def __init__(self, celsius=0):
            self._celsius = celsius
        
        @property
        def celsius(self):
            """Get celsius value."""
            return self._celsius
        
        @celsius.setter
        def celsius(self, value):
            if value < -273.15:
                raise ValueError("Below absolute zero!")
            self._celsius = value
        
        @celsius.deleter
        def celsius(self):
            del self._celsius
        
        @property
        def fahrenheit(self):
            """Computed property."""
            return self._celsius * 9/5 + 32
        
        @fahrenheit.setter
        def fahrenheit(self, value):
            self.celsius = (value - 32) * 5/9
    
    # Usage - looks like attribute access
    t = Temperature(25)
    print(t.celsius)      # 25
    print(t.fahrenheit)   # 77.0
    
    t.fahrenheit = 100
    print(t.celsius)      # 37.78
    
    t.celsius = -300      # Raises ValueError
    ```
    
    **Read-Only Property:**
    
    ```python
    class Circle:
        def __init__(self, radius):
            self._radius = radius
        
        @property
        def area(self):
            """Area is computed, read-only."""
            return 3.14159 * self._radius ** 2
    
    c = Circle(5)
    print(c.area)  # 78.54
    c.area = 100   # AttributeError - no setter
    ```
    
    **Lazy/Cached Property:**
    
    ```python
    class ExpensiveObject:
        @property
        def data(self):
            if not hasattr(self, '_data'):
                print("Computing...")
                self._data = expensive_computation()
            return self._data
    
    # Python 3.8+ cached_property
    from functools import cached_property
    
    class Better:
        @cached_property
        def data(self):
            print("Computing...")
            return expensive_computation()
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Encapsulation and API design.
        
        **Strong answer signals:**
        
        - Uses properties for validation
        - Knows read-only properties (no setter)
        - Uses cached_property for expensive computations
        - Understands when to use properties vs methods

---

### What Are Abstract Base Classes (ABCs)? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `OOP`, `Abstraction`, `Interfaces` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Abstract Base Classes:**
    
    Define interfaces that subclasses must implement.
    
    ```python
    from abc import ABC, abstractmethod
    
    class Shape(ABC):
        @abstractmethod
        def area(self):
            """Calculate area - must be implemented."""
            pass
        
        @abstractmethod
        def perimeter(self):
            """Calculate perimeter - must be implemented."""
            pass
        
        def describe(self):
            """Concrete method - inherited."""
            return f"A shape with area {self.area()}"
    
    class Circle(Shape):
        def __init__(self, radius):
            self.radius = radius
        
        def area(self):
            return 3.14159 * self.radius ** 2
        
        def perimeter(self):
            return 2 * 3.14159 * self.radius
    
    # Usage
    c = Circle(5)
    print(c.area())       # 78.54
    print(c.describe())   # "A shape with area 78.54"
    
    # Cannot instantiate abstract class
    s = Shape()  # TypeError: Can't instantiate abstract class
    ```
    
    **Abstract Properties:**
    
    ```python
    class DataStore(ABC):
        @property
        @abstractmethod
        def connection_string(self):
            """Connection string must be defined."""
            pass
        
        @abstractmethod
        def connect(self):
            pass
    
    class SQLStore(DataStore):
        @property
        def connection_string(self):
            return "postgresql://..."
        
        def connect(self):
            return f"Connected to {self.connection_string}"
    ```
    
    **Duck Typing vs ABCs:**
    
    ```python
    # Duck typing - no formal interface
    def process(obj):
        obj.save()  # Assumes .save() exists
    
    # ABC - explicit interface
    class Saveable(ABC):
        @abstractmethod
        def save(self):
            pass
    
    def process(obj: Saveable):
        obj.save()  # Guaranteed to exist
    
    # Check implementation
    isinstance(obj, Saveable)  # True if implements interface
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Interface design and polymorphism.
        
        **Strong answer signals:**
        
        - Uses ABCs for API contracts
        - Knows @abstractmethod forces implementation
        - Can combine abstract and concrete methods
        - Understands Python's duck typing philosophy

---

### What is `multiprocessing.Pool`? How Do You Parallelize? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Parallelism`, `Multiprocessing`, `Performance` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Pool for Parallel Processing:**
    
    ```python
    from multiprocessing import Pool
    import time
    
    def expensive_func(x):
        time.sleep(1)  # Simulate work
        return x ** 2
    
    # Sequential - slow
    start = time.time()
    results = [expensive_func(x) for x in range(4)]
    print(f"Sequential: {time.time() - start:.2f}s")  # ~4s
    
    # Parallel - fast
    start = time.time()
    with Pool(processes=4) as pool:
        results = pool.map(expensive_func, range(4))
    print(f"Parallel: {time.time() - start:.2f}s")  # ~1s
    ```
    
    **Pool Methods:**
    
    ```python
    from multiprocessing import Pool
    
    def process(x):
        return x * 2
    
    with Pool(4) as pool:
        # map - blocks until all complete
        results = pool.map(process, range(10))
        
        # imap - iterator, more memory efficient
        for result in pool.imap(process, range(10)):
            print(result)
        
        # imap_unordered - faster, arbitrary order
        for result in pool.imap_unordered(process, range(10)):
            print(result)
        
        # apply_async - non-blocking
        future = pool.apply_async(process, (5,))
        result = future.get(timeout=10)
        
        # starmap - multiple arguments
        pool.starmap(pow, [(2, 3), (2, 4), (2, 5)])  # [8, 16, 32]
    ```
    
    **concurrent.futures (Preferred):**
    
    ```python
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    def process(x):
        return x ** 2
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit individual tasks
        futures = [executor.submit(process, x) for x in range(10)]
        
        # Get results as completed (any order)
        for future in as_completed(futures):
            print(future.result())
        
        # Or use map (ordered results)
        results = list(executor.map(process, range(10)))
    ```
    
    **Gotchas:**
    
    ```python
    # Functions must be picklable (top-level, not lambda)
    # pool.map(lambda x: x*2, range(10))  # Error!
    
    # Shared state is complex - use Manager
    from multiprocessing import Manager
    
    with Manager() as manager:
        shared_dict = manager.dict()
        shared_list = manager.list()
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Parallelization knowledge.
        
        **Strong answer signals:**
        
        - Uses context manager for cleanup
        - Knows concurrent.futures is more modern
        - Understands pickling requirement
        - Chooses ProcessPool for CPU, ThreadPool for I/O

---

### What is `itertools.groupby`? How Do You Use It? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Grouping`, `Iteration`, `Data Processing` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **groupby Basics:**
    
    Groups consecutive elements with the same key.
    
    ```python
    from itertools import groupby
    
    # Simple grouping
    data = [1, 1, 2, 2, 2, 3, 1, 1]
    for key, group in groupby(data):
        print(f"{key}: {list(group)}")
    # 1: [1, 1]
    # 2: [2, 2, 2]
    # 3: [3]
    # 1: [1, 1]  <- Note: 1 appears again!
    ```
    
    **⚠️ Critical: Sort First for Full Grouping:**
    
    ```python
    # For complete grouping, sort first!
    data = [('a', 1), ('b', 2), ('a', 3), ('b', 4)]
    
    # Sort by key
    sorted_data = sorted(data, key=lambda x: x[0])
    
    for key, group in groupby(sorted_data, key=lambda x: x[0]):
        print(f"{key}: {list(group)}")
    # a: [('a', 1), ('a', 3)]
    # b: [('b', 2), ('b', 4)]
    ```
    
    **Common Patterns:**
    
    ```python
    # Count consecutive occurrences
    text = "aaabbbccaaa"
    result = [(k, len(list(g))) for k, g in groupby(text)]
    # [('a', 3), ('b', 3), ('c', 2), ('a', 3)]
    
    # Group by attribute
    from collections import namedtuple
    Person = namedtuple('Person', ['name', 'age', 'city'])
    
    people = [
        Person('Alice', 30, 'NYC'),
        Person('Bob', 25, 'NYC'),
        Person('Charlie', 30, 'LA'),
    ]
    
    # Group by city
    by_city = sorted(people, key=lambda p: p.city)
    for city, group in groupby(by_city, key=lambda p: p.city):
        print(f"{city}: {[p.name for p in group]}")
    ```
    
    **Alternative: defaultdict:**
    
    ```python
    from collections import defaultdict
    
    # Often simpler than groupby
    groups = defaultdict(list)
    for item in data:
        groups[item[0]].append(item)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data grouping patterns.
        
        **Strong answer signals:**
        
        - Knows groupby works on consecutive, not all matching
        - Always sorts first for full grouping
        - Mentions defaultdict as simpler alternative
        - Knows group is an iterator (consume immediately)

---

### What Are `@staticmethod` and `@classmethod`? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `OOP`, `Methods`, `Decorators` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Method Types:**
    
    | Type | First Arg | Access | Use Case |
    |------|-----------|--------|----------|
    | Instance | self | Instance + class | Normal methods |
    | Class | cls | Class only | Factory, class-level |
    | Static | None | Neither | Utility functions |
    
    ```python
    class MyClass:
        class_var = 0
        
        def __init__(self, value):
            self.value = value
        
        # Instance method - needs instance
        def instance_method(self):
            return f"Instance: {self.value}"
        
        # Class method - gets class, not instance
        @classmethod
        def class_method(cls):
            return f"Class: {cls.class_var}"
        
        # Static method - no access to self or cls
        @staticmethod
        def static_method(x, y):
            return x + y
    
    # Usage
    obj = MyClass(10)
    obj.instance_method()      # "Instance: 10"
    MyClass.class_method()     # "Class: 0"
    MyClass.static_method(1,2) # 3
    ```
    
    **Factory Pattern with classmethod:**
    
    ```python
    class Date:
        def __init__(self, year, month, day):
            self.year = year
            self.month = month
            self.day = day
        
        @classmethod
        def from_string(cls, date_string):
            """Alternative constructor."""
            year, month, day = map(int, date_string.split('-'))
            return cls(year, month, day)
        
        @classmethod
        def today(cls):
            """Another alternative constructor."""
            import datetime
            d = datetime.date.today()
            return cls(d.year, d.month, d.day)
    
    d1 = Date(2024, 12, 25)
    d2 = Date.from_string('2024-12-25')
    d3 = Date.today()
    ```
    
    **Inheritance Difference:**
    
    ```python
    class Parent:
        @classmethod
        def create(cls):
            return cls()  # Creates correct subclass!
        
        @staticmethod
        def utility():
            return "Same for all"
    
    class Child(Parent):
        pass
    
    Parent.create()  # <Parent>
    Child.create()   # <Child> - classmethod knows about subclass!
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** OOP fundamentals.
        
        **Strong answer signals:**
        
        - Uses classmethod for factory patterns
        - Knows classmethod receives cls, works with inheritance
        - Uses staticmethod for utilities that don't need class/instance
        - Can explain when to use each

---

## Quick Reference: 100+ Interview Questions

This document provides a curated list of Python interview questions commonly asked in technical interviews for Data Scientists, Backend Engineers, and Python Developers. It covers core concepts, internals, concurrency, and advanced language features.

This is updated frequently but right now this is the most exhaustive list of type of questions being asked.


---

| Sno | Question Title | Practice Links | Companies Asking | Difficulty | Topics |
|-----|----------------|----------------|------------------|------------|--------|
| 1 | What is Python? Interpreted or Compiled? | [Python Docs](https://docs.python.org/3/glossary.html#term-interpreted) | Google, Amazon, Meta | Easy | Basics |
| 2 | What is PEP 8? | [PEP 8](https://peps.python.org/pep-0008/) | Most Tech Companies | Easy | Standards |
| 3 | Mutable vs Immutable types in Python | [Real Python](https://realpython.com/python-mutable-immutable/) | Google, Amazon, Microsoft | Easy | Data Structures |
| 4 | Explain List vs Tuple | [GeeksforGeeks](https://www.geeksforgeeks.org/python-list-vs-tuples/) | Most Tech Companies | Easy | Data Structures |
| 5 | What is a Dictionary in Python? | [Python Docs](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) | Most Tech Companies | Easy | Data Structures |
| 6 | How is memory managed in Python? | [Real Python](https://realpython.com/python-memory-management/) | Google, Amazon, Meta, Netflix | Medium | Internals, Memory |
| 7 | What is the Global Interpreter Lock (GIL)? | [Real Python](https://realpython.com/python-gil/) | Google, Amazon, Meta, Apple | Hard | Internals, Concurrency |
| 8 | Explain Garbage Collection in Python | [Python Docs](https://docs.python.org/3/library/gc.html) | Google, Amazon, Spotify | Medium | Internals, GC |
| 9 | What are decorators? | [Real Python](https://realpython.com/primer-on-python-decorators/) | Google, Amazon, Meta, Netflix | Medium | Functions, Advanced |
| 10 | Difference between `@staticmethod` and `@classmethod` | [Stack Overflow](https://stackoverflow.com/questions/12179271/meaning-of-classmethod-and-staticmethod-for-beginner) | Google, Amazon, Meta | Easy | OOP |
| 11 | What are lambda functions? | [Real Python](https://realpython.com/python-lambda/) | Most Tech Companies | Easy | Functions |
| 12 | Explain `*args` and `**kwargs` | [Real Python](https://realpython.com/python-kwargs-and-args/) | Most Tech Companies | Easy | Functions |
| 13 | What are generators and the `yield` keyword? | [Real Python](https://realpython.com/introduction-to-python-generators/) | Google, Amazon, Meta, Netflix | Medium | Iterators, Generators |
| 14 | Difference between range() and xrange() | [GeeksforGeeks](https://www.geeksforgeeks.org/range-vs-xrange-python/) | Legacy Companies | Easy | Python 2 vs 3 |
| 15 | What is a docstring? | [Python Docs](https://peps.python.org/pep-0257/) | Most Tech Companies | Easy | Documentation |
| 16 | How to copy an object? (Deep vs Shallow copy) | [Real Python](https://realpython.com/copying-python-objects/) | Google, Amazon, Meta | Medium | Objects, Memory |
| 17 | What is `__init__`? | [Python Docs](https://docs.python.org/3/reference/datamodel.html#object.__init__) | Most Tech Companies | Easy | OOP |
| 18 | What is `__str__` vs `__repr__`? | [Stack Overflow](https://stackoverflow.com/questions/1436703/what-is-the-difference-between-str-and-repr) | Google, Amazon, Meta | Medium | OOP, Magic Methods |
| 19 | How typically does inheritance work in Python? | [Real Python](https://realpython.com/python-inheritance-composition/) | Most Tech Companies | Easy | OOP |
| 20 | What is Multiple Inheritance and MRO? | [Python Docs](https://docs.python.org/3/tutorial/classes.html#multiple-inheritance) | Google, Amazon, Meta | Hard | OOP, MRO |
| 21 | Explain `is` vs `==` | [Real Python](https://realpython.com/python-is-identity-vs-equality/) | Most Tech Companies | Easy | Operators |
| 22 | What are packing and unpacking? | [Real Python](https://realpython.com/python-kwargs-and-args/) | Google, Amazon, Meta | Medium | Basics |
| 23 | How to handle exceptions? (try/except/finally) | [Python Docs](https://docs.python.org/3/tutorial/errors.html) | Most Tech Companies | Easy | Error Handling |
| 24 | What are assertions? | [Real Python](https://realpython.com/python-assert-statement/) | Google, Amazon, Meta | Medium | Debugging |
| 25 | What implies `pass` statement? | [Python Docs](https://docs.python.org/3/reference/simple_stmts.html#the-pass-statement) | Most Tech Companies | Easy | Control Flow |
| 26 | What are Context Managers (`with` statement)? | [Real Python](https://realpython.com/python-with-statement/) | Google, Amazon, Meta, Netflix | Medium | Context Managers |
| 27 | Difference between lists and arrays (array module)? | [GeeksforGeeks](https://www.geeksforgeeks.org/difference-between-list-and-array-in-python/) | Google, Amazon | Easy | Data Structures |
| 28 | What is a Set? | [Python Docs](https://docs.python.org/3/tutorial/datastructures.html#sets) | Most Tech Companies | Easy | Data Structures |
| 29 | How does Python handle large numbers? | [Python Docs](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) | Google, Amazon, HFT Firms | Medium | Internals |
| 30 | What are namespaces? | [Real Python](https://realpython.com/python-namespaces-scope/) | Google, Amazon, Meta | Medium | Scoping |
| 31 | Explain Local, Global, and Nonlocal scope | [Real Python](https://realpython.com/python-scope-legb-rule/) | Google, Amazon, Meta | Medium | Scoping |
| 32 | What is a module vs a package? | [Real Python](https://realpython.com/python-modules-packages/) | Most Tech Companies | Easy | Modules |
| 33 | How to use `pip`? | [Python Docs](https://packaging.python.org/en/latest/tutorials/installing-packages/) | Most Tech Companies | Easy | Packaging |
| 34 | What is venv/virtualenv? | [Real Python](https://realpython.com/python-virtual-environments-a-primer/) | Most Tech Companies | Easy | Environment |
| 35 | How to read/write files? | [Python Docs](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files) | Most Tech Companies | Easy | I/O |
| 36 | What is pickling and unpickling? | [Python Docs](https://docs.python.org/3/library/pickle.html) | Google, Amazon, Meta | Medium | Serialization |
| 37 | What are iterators and lazy evaluation? | [Real Python](https://realpython.com/python-iterators-iterables/) | Google, Amazon, Meta | Medium | Iterators |
| 38 | What is the `zip()` function? | [Real Python](https://realpython.com/python-zip-function/) | Most Tech Companies | Easy | Built-ins |
| 39 | What is `map()` and `filter()`? | [Real Python](https://realpython.com/python-map-function/) | Google, Amazon, Meta | Easy | Functional Programming |
| 40 | What is `functools.reduce()`? | [Real Python](https://realpython.com/python-reduce-function/) | Google, Amazon, Meta | Medium | Functional Programming |
| 41 | Difference between .py and .pyc files | [Stack Overflow](https://stackoverflow.com/questions/2998215/if-python-is-interpreted-what-are-pyc-files) | Google, Amazon, Meta | Medium | Internals |
| 42 | What is `__name__ == "__main__"`? | [Real Python](https://realpython.com/python-main-function/) | Most Tech Companies | Easy | Modules |
| 43 | How to create a singleton class? | [GeeksforGeeks](https://www.geeksforgeeks.org/singleton-pattern-in-python-a-complete-guide/) | Google, Amazon, Meta | Hard | Patterns |
| 44 | What are Metaclasses? | [Real Python](https://realpython.com/python-metaclasses/) | Google, Amazon, Meta, Netflix | Hard | Metaclasses |
| 45 | What is `__slots__`? | [GeeksforGeeks](https://www.geeksforgeeks.org/slots-in-python/) | Google, Amazon, HFT Firms | Hard | Optimization, Memory |
| 46 | Difference between `func` and `func()` | [Stack Overflow](https://stackoverflow.com/questions/60718507/what-is-the-difference-between-def-func-and-def-func) | Google, Amazon | Easy | Functions |
| 47 | What is slicing? | [Python Docs](https://docs.python.org/3/library/functions.html#slice) | Most Tech Companies | Easy | Data Structures |
| 48 | Negative indexing in Python | [GeeksforGeeks](https://www.geeksforgeeks.org/python-negative-indexing/) | Most Tech Companies | Easy | Indexing |
| 49 | What is a hash map in Python? | [Python Docs](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict) | Most Tech Companies | Easy | Data Structures |
| 50 | Does Python support pointer arithmetic? | [Stack Overflow](https://stackoverflow.com/questions/15474668/pointer-operations-in-python) | Google, Amazon | Medium | Internals |
| 51 | What are default arguments? Pitfalls? | [Real Python](https://realpython.com/defining-your-own-python-function/#default-argument-values) | Google, Amazon, Meta | Medium | Functions, Pitfalls |
| 52 | What is `collections` module? | [Python Docs](https://docs.python.org/3/library/collections.html) | Google, Amazon, Meta | Medium | Standard Library |
| 53 | Explain `defaultdict` and `Counter` | [Real Python](https://realpython.com/python-defaultdict/) | Google, Amazon, Meta | Medium | Standard Library |
| 54 | What is `NamedTuple`? | [Real Python](https://realpython.com/python-namedtuple/) | Google, Amazon, Meta | Medium | Data Structures |
| 55 | What is `itertools`? | [Real Python](https://realpython.com/python-itertools/) | Google, Amazon, Meta, HFT Firms | Hard | Iterators |
| 56 | What are Threading vs Multiprocessing? | [Real Python](https://realpython.com/python-concurrency/) | Google, Amazon, Meta, Netflix | Medium | Concurrency |
| 57 | What is Asyncio? | [Real Python](https://realpython.com/async-io-python/) | Google, Amazon, Meta, Netflix | Hard | Concurrency |
| 58 | Difference between `await` and `yield` | [Stack Overflow](https://stackoverflow.com/questions/40571786/async-await-vs-yield-yield-from) | Google, Amazon, Meta | Hard | Concurrency |
| 59 | What is a coroutine? | [Python Docs](https://docs.python.org/3/library/asyncio-task.html#coroutines) | Google, Amazon, Meta | Hard | Concurrency |
| 60 | How to debug python code? (pdb) | [Real Python](https://realpython.com/python-debugging-pdb/) | Most Tech Companies | Medium | Debugging |
| 61 | What are type hints (Type Annotation)? | [Real Python](https://realpython.com/python-type-checking/) | Google, Amazon, Meta, Microsoft | Medium | Typing |
| 62 | What is `MyPy`? | [MyPy Docs](https://mypy.readthedocs.io/en/stable/) | Google, Amazon, Meta | Medium | Typing |
| 63 | What is a Data Class? | [Real Python](https://realpython.com/python-data-classes/) | Google, Amazon, Meta | Medium | Data Structures |
| 64 | Difference between `copy()` and `deepcopy()` | [Python Docs](https://docs.python.org/3/library/copy.html) | Google, Amazon, Meta | Medium | Memory |
| 65 | How to reverse a list? | [Real Python](https://realpython.com/python-reverse-list/) | Most Tech Companies | Easy | Data Structures |
| 66 | String formatting options (f-strings vs format) | [Real Python](https://realpython.com/python-f-strings/) | Most Tech Companies | Easy | Strings |
| 67 | What is `sys.path`? | [Python Docs](https://docs.python.org/3/library/sys.html#sys.path) | Google, Amazon, Meta | Medium | Modules |
| 68 | What is `__call__` method? | [GeeksforGeeks](https://www.geeksforgeeks.org/__call__-in-python/) | Google, Amazon, Meta | Medium | OOP |
| 69 | What is `__new__` vs `__init__`? | [Stack Overflow](https://stackoverflow.com/questions/674304/pythons-use-of-new-and-init) | Google, Amazon, Meta | Hard | OOP |
| 70 | What is Monkey Patching? | [Stack Overflow](https://stackoverflow.com/questions/5626193/what-is-monkey-patching) | Google, Amazon, Meta | Medium | Dynamic Programming |
| 71 | What is Duck Typing? | [Real Python](https://realpython.com/python-duck-typing/) | Google, Amazon, Meta | Medium | OOP |
| 72 | What is `dir()` function? | [Python Docs](https://docs.python.org/3/library/functions.html#dir) | Most Tech Companies | Easy | Introspection |
| 73 | What is `help()` function? | [Python Docs](https://docs.python.org/3/library/functions.html#help) | Most Tech Companies | Easy | Introspection |
| 74 | What is `enumerate()`? | [Real Python](https://realpython.com/python-enumerate/) | Most Tech Companies | Easy | Iteration |
| 75 | How to merge two dicts? | [Real Python](https://realpython.com/python-merge-dictionaries/) | Most Tech Companies | Easy | Data Structures |
| 76 | Comprehensions (List, Dict, Set) | [Real Python](https://realpython.com/list-comprehension-python/) | Most Tech Companies | Medium | Syntax |
| 77 | What is `__future__` module? | [Python Docs](https://docs.python.org/3/library/__future__.html) | Google, Amazon | Medium | Compatibility |
| 78 | What is `pd.DataFrame` vs Python List? | [Pandas Docs](https://pandas.pydata.org/) | Most Tech Companies | Easy | Data Analysis |
| 79 | How to handle circular imports? | [Stack Overflow](https://stackoverflow.com/questions/744373/circular-or-cyclic-imports-in-python) | Google, Amazon, Meta | Medium | Modules |
| 80 | What is `getattr`, `setattr`, `hasattr`? | [Python Docs](https://docs.python.org/3/library/functions.html) | Google, Amazon, Meta | Medium | Introspection |
| 81 | What is `__dict__` attribute? | [Python Docs](https://docs.python.org/3/library/stdtypes.html#object.__dict__) | Google, Amazon, Meta | Medium | Internals |
| 82 | Explain the `with` statement protocol (`__enter__`, `__exit__`) | [Real Python](https://realpython.com/python-with-statement/#the-context-management-protocol) | Google, Amazon, Meta | Hard | Context Managers |
| 83 | What are property decorators? | [Real Python](https://realpython.com/python-property/) | Google, Amazon, Meta | Medium | OOP |
| 84 | What is Operator Overloading? | [GeeksforGeeks](https://www.geeksforgeeks.org/operator-overloading-in-python/) | Google, Amazon, Meta | Medium | OOP |
| 85 | What is ternary operator in Python? | [Real Python](https://realpython.com/python-conditional-statements/#conditional-expressions) | Most Tech Companies | Easy | Syntax |
| 86 | How to optimize Python code speed? | [Wiki](https://wiki.python.org/moin/PythonSpeed/PerformanceTips) | Google, Amazon, HFT Firms | Hard | Performance |
| 87 | Why is Python slow? | [Real Python](https://realpython.com/python-performance/) | Google, Amazon, Meta | Medium | Performance |
| 88 | What is PyPy? | [PyPy](https://www.pypy.org/) | Google, Amazon | Hard | Interpreters |
| 89 | What is Cython? | [Cython](https://cython.org/) | Google, Amazon, HFT Firms | Hard | Performance |
| 90 | Difference between `os` and `sys` modules? | [Stack Overflow](https://stackoverflow.com/questions/18389656/what-is-the-difference-between-the-sys-and-os-modules-in-python) | Google, Amazon | Medium | Standard Library |
| 91 | What is `re` module (Regular Expressions)? | [Real Python](https://realpython.com/regex-python/) | Most Tech Companies | Medium | Standard Library |
| 92 | **[HARD]** How does Reference Counting vs Garbage Collection work? | [DevGuide](https://devguide.python.org/internals/garbage-collector/) | Google, Meta, Netflix | Hard | Internals |
| 93 | **[HARD]** How to implement custom Metaclass? | [Real Python](https://realpython.com/python-metaclasses/) | Google, Meta, Frameworks | Hard | Metaprogramming |
| 94 | **[HARD]** Explain method resolution order (MRO) C3 algorithm | [Python Docs](https://www.python.org/download/releases/2.3/mro/) | Google, Meta | Hard | OOP Internals |
| 95 | **[HARD]** How to avoid the GIL (multiprocessing, C extensions)? | [Real Python](https://realpython.com/python-gil/) | Google, Amazon, HFT Firms | Hard | Performance |
| 96 | **[HARD]** Memory leaks in Python: Causes and fixes | [TechBlog](https://tech.trivago.com/2019/02/25/diagnosing-memory-leaks-in-python/) | Google, Meta, Netflix | Hard | Memory |
| 97 | **[HARD]** How `asyncio` event loop works internally | [Real Python](https://realpython.com/async-io-python/) | Google, Meta, Netflix | Hard | Asyncio Internals |
| 98 | **[HARD]** Difference between Threading and Asyncio concurrency models | [Real Python](https://realpython.com/python-concurrency/) | Google, Meta, Netflix | Hard | Concurrency |
| 99 | **[HARD]** How to implement non-blocking I/O? | [Python Docs](https://docs.python.org/3/library/asyncio.html) | Google, Amazon | Hard | I/O |
| 100| **[HARD]** What is `__import__` vs `importlib`? | [Python Docs](https://docs.python.org/3/library/importlib.html) | Google, Meta, Frameworks | Hard | Internals |
| 101 | **[HARD]** How are Python dictionaries implemented (Hash Table)? | [PyCon Talk](https://www.youtube.com/watch?v=p33CVV29OG8) | Google, Meta, Amazon | Hard | Internals |
| 102 | **[HARD]** Explain descriptor protocol | [Real Python](https://realpython.com/python-descriptors/) | Google, Meta, Frameworks | Hard | Descriptors |
| 103 | **[HARD]** How to use `sys.settrace` for debugging/profiling? | [Python Docs](https://docs.python.org/3/library/sys.html#sys.settrace) | Google, Meta | Hard | Internals |
| 104 | **[HARD]** Difference between Process and Thread in Python context | [Real Python](https://realpython.com/python-concurrency/) | Google, Amazon, Meta | Hard | OS Concepts |
| 105 | **[HARD]** How to manage weak references (`weakref`)? | [Python Docs](https://docs.python.org/3/library/weakref.html) | Google, Meta | Hard | Memory |
| 106 | **[HARD]** What is the Disassembler (`dis` module)? | [Python Docs](https://docs.python.org/3/library/dis.html) | Google, Meta | Hard | Bytecode |
| 107 | **[HARD]** How to optimize dictionary memory usage? | [Stack Overflow](https://stackoverflow.com/questions/39913211/how-to-optimize-the-memory-usage-of-a-dictionary-in-python) | Google, Amazon | Hard | Memory |
| 108 | **[HARD]** Explain Coroutines vs Generators | [Real Python](https://realpython.com/python-concurrency/) | Google, Meta | Hard | Concurrency |
| 109 | **[HARD]** How to implement custom context managers (contextlib)? | [Python Docs](https://docs.python.org/3/library/contextlib.html) | Google, Amazon | Hard | Advanced Patterns |
| 110 | **[HARD]** How to perform zero-copy data transfer (Buffer Protocol)? | [Python Docs](https://docs.python.org/3/c-api/buffer.html) | Google, HFT Firms | Hard | Performance |

---

## Code Examples

### 1. Decorator for Timing Functions

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
    ```python
    import time
    import functools

    def timer_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
            return result
        return wrapper

    @timer_decorator
    def complex_calculation(n):
        return sum(i**2 for i in range(n))

    complex_calculation(1000000)
    ```

### 2. Context Manager for Files

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
    ```python
    class FileManager:
        def __init__(self, filename, mode):
            self.filename = filename
            self.mode = mode
            self.file = None

        def __enter__(self):
            self.file = open(self.filename, self.mode)
            return self.file

        def __exit__(self, exc_type, exc_value, exc_traceback):
            if self.file:
                self.file.close()

    # Usage
    # with FileManager('test.txt', 'w') as f:
    #     f.write('Hello, World!')
    ```

### 3. Asynchronous Pattern

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
    ```python
    import asyncio

    async def fetch_data(delay, id):
        print(f"Fetching data {id}...")
        await asyncio.sleep(delay)  # Simulate I/O op
        print(f"Data {id} fetched")
        return {"id": id, "data": "sample"}

    async def main():
        # Run tasks concurrently
        tasks = [fetch_data(1, 1), fetch_data(2, 2), fetch_data(1.5, 3)]
        results = await asyncio.gather(*tasks)
        print(results)

    # asyncio.run(main())
    ```

---

## Questions asked in Google interview
- Explain the Global Interpreter Lock (GIL) and its impact on multi-threading.
- How does Python's garbage collection mechanism work? (Reference counting vs Generational GC).
- Write a custom decorator that caches function results (Memoization).
- How would you debug a memory leak in a long-running Python process?
- Explain the method resolution order (MRO) works with multiple inheritance.
- Write code to implement a thread-safe singleton.
- How to optimize a CPU-bound Python script?
- Explain the key differences between Python 2 and Python 3.
- How to implement a context manager using `contextlib`.
- Write code to parse a large log file without loading it entirely into memory.

## Questions asked in Meta interview
- How are dictionaries implemented in Python? (Hash collision handling).
- Explain the difference between `__new__` and `__init__`.
- Write code to flatten a deeply nested dictionary.
- How does `asyncio` differ from threading? When to use which?
- Explain the concept of metaclasses and a use case.
- Write a generator that yields Fibonacci numbers indefinitely.
- How to handle circular imports in a large project?
- Explain the descriptor protocol and how properties work.
- How would you implement a custom iterator?
- Write code to validate and parse JSON data using `dataclasses` or `pydantic`.

## Questions asked in Amazon interview
- Write code to reverse a string without using built-in methods.
- Explain the difference between deep copy and shallow copy.
- How to handle exceptions in a large-scale application?
- Write code to find the most frequent element in a list efficiently.
- Explain the use of `*args` and `**kwargs`.
- How to implement a producer-consumer problem using `queue`?
- Explain the difference between `@staticmethod`, `@classmethod`, and instance methods.
- Write code to sort a list of dictionaries by a specific key.
- How does variable scope work in Python (LEGB rule)?
- Explain what `if __name__ == "__main__":` does.

## Questions asked in Netflix interview
- How to optimize Python for high-throughput network applications?
- Explain the internals of CPython execution loop.
- Write code to implement a rate limiter using Redis and Python.
- How to handle dependency management in a microservices architecture?
- Explain how `gunicorn` or `uwsgi` works with Python web apps.
- Write code to implement async HTTP requests using `aiohttp`.
- How to profile Python code to find bottlenecks? (cProfile, py-spy).
- Explain the challenge of serialization (pickling) in distributed systems.
- How to implement rigorous unit testing with `pytest`?
- Write code to process a stream of data using generators.
---

## Additional Resources

- [Official Python Documentation](https://docs.python.org/3/)
- [Real Python Tutorials](https://realpython.com/)
- [Fluent Python (Book)](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)
- [Python Internals (GitHub)](https://github.com/zpoint/CPython-Internals)
- [Hitchhiker's Guide to Python](https://docs.python-guide.org/)

