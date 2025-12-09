---
title: Python Interview Questions
description: 100+ Python interview questions for cracking Data Science, Backend, and Software Engineering interviews
---

# Python Interview Questions

<!-- [TOC] -->

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
