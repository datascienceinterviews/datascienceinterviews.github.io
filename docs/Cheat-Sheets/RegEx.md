
---
title: Regular Expressions (RegEx) Cheat Sheet
description: A comprehensive reference guide for regular expressions, covering syntax, character classes, quantifiers, anchors, groups, and more, with Python examples.
---

# Regular Expressions (RegEx) Cheat Sheet

[TOC]

This cheat sheet provides an exhaustive overview of regular expressions (RegEx), covering syntax, character classes, quantifiers, anchors, groups, flags, and advanced techniques. It aims to be a complete reference for using regular expressions, with a focus on Python examples.

## RegEx Pattern Matching Flow

```
    ┌──────────┐
    │  Input   │
    │  String  │
    └─────┬────┘
          ↓
    ┌──────────────┐
    │   Pattern    │
    │   Compile    │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │   Match      │
    │   Attempt    │
    └──────┬───────┘
           │
     ┌─────┴──────┐
     ↓            ↓
  ┌──────┐    ┌─────────┐
  │Match │    │No Match │
  └──────┘    └─────────┘
```

## Basic Syntax

### Literals

Characters match themselves literally, except for special characters.

*   `abc`: Matches the literal string "abc".
*   `hello`: Matches the literal string "hello".

```python
import re

# Basic literal matching
text = "abc def ghi"
pattern = r"abc"
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: abc

# Multiple literal matches
text = "hello world hello"
matches = re.findall(r"hello", text)
print(matches)  # Output: ['hello', 'hello']
```

### Special Characters

These characters have special meanings in RegEx:

*   `. ^ $ * + ? { } [ ] \ | ( )`

To match these characters literally, escape them with a backslash (`\`):

*   `\.`: Matches a literal dot (`.`).
*   `\\`: Matches a literal backslash (`\`).

```python
import re

# Escaping special characters
text = "123.456"
pattern = r"\."  # Matches a literal dot
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: .

# Matching literal backslash
text = "path\\to\\file"
pattern = r"\\"  # Matches a literal backslash
matches = re.findall(pattern, text)
print(matches)  # Output: ['\\', '\\']

# Matching special characters
text = "Price: $100 + $50 = $150"
pattern = r"\$\d+"  # Matches dollar sign followed by digits
matches = re.findall(pattern, text)
print(matches)  # Output: ['$100', '$50', '$150']
```

### Character Classes

*   `.`: Matches any character except a newline (unless the `s` flag is used).
*   `[abc]`: Matches any character inside the brackets (a, b, or c).
*   `[^abc]`: Matches any character *not* inside the brackets.
*   `[a-z]`: Matches any character in the range a to z (lowercase).
*   `[A-Z]`: Matches any character in the range A to Z (uppercase).
*   `[0-9]`: Matches any digit.
*   `[a-zA-Z0-9]`: Matches any alphanumeric character.

```
    Character Classes:
    
    [abc]      [^abc]     [a-z]
      │           │          │
      ↓           ↓          ↓
    Match      Negate    Range
     a,b,c     NOT a,b,c  a-z
```

```python
import re

# Match specific characters
text = "apple banana cherry"
pattern = r"[abc]"  # Matches 'a', 'b', or 'c'
matches = re.findall(pattern, text)
print(matches)  # Output: ['a', 'a', 'b', 'a', 'a', 'a', 'a', 'c']

# Negated character class
text = "apple banana cherry"
pattern = r"[^abc]"  # Matches any character except 'a', 'b', or 'c'
matches = re.findall(pattern, text)
print(matches)  # Output: ['p', 'p', 'l', 'e', ' ', 'n', 'n', ' ', 'h', 'e', 'r', 'r', 'y']

# Range matching
text = "apple1 banana2 cherry3"
pattern = r"[0-9]"  # Matches any digit
matches = re.findall(pattern, text)
print(matches)  # Output: ['1', '2', '3']

# Alphanumeric matching
text = "Hello123 World456"
pattern = r"[a-zA-Z0-9]+"  # Matches alphanumeric sequences
matches = re.findall(pattern, text)
print(matches)  # Output: ['Hello123', 'World456']

# Dot wildcard
text = "Hello. World!"
pattern = r"."  # Matches any character except newline
matches = re.findall(pattern, text)
print(matches)  # Output: ['H', 'e', 'l', 'l', 'o', '.', ' ', 'W', 'o', 'r', 'l', 'd', '!']
```

### Predefined Character Classes

*   `\d`: Matches any digit (equivalent to `[0-9]`).
*   `\D`: Matches any non-digit (equivalent to `[^0-9]`).
*   `\w`: Matches any word character (alphanumeric + underscore, equivalent to `[a-zA-Z0-9_]`).
*   `\W`: Matches any non-word character (equivalent to `[^a-zA-Z0-9_]`).
*   `\s`: Matches any whitespace character (space, tab, newline, etc.).
*   `\S`: Matches any non-whitespace character.
*   `\b`: Matches a word boundary.
*   `\B`: Matches a non-word boundary.
*   `\t`: Matches a tab character.
*   `\n`: Matches a newline character.
*   `\r`: Matches a carriage return character.
*   `\f`: Matches a form feed character.
*   `\v`: Matches a vertical tab character.
*   `\0`: Matches a null character.
*   `[\b]`: Matches a backspace character (inside a character class).

```
    Predefined Classes:
    
    \d → [0-9]       \D → [^0-9]
    \w → [a-zA-Z0-9_]  \W → [^a-zA-Z0-9_]
    \s → [ \t\n\r\f\v]  \S → [^ \t\n\r\f\v]
```

```python
import re

# Digit matching
text = "123 abc 456"
pattern = r"\d+"  # Matches one or more digits
matches = re.findall(pattern, text)
print(matches)  # Output: ['123', '456']

# Word character matching
text = "user_name123 = value"
pattern = r"\w+"  # Matches word characters
matches = re.findall(pattern, text)
print(matches)  # Output: ['user_name123', 'value']

# Word boundary
text = "Hello World, HelloWorld"
pattern = r"\bWorld\b"  # Matches "World" at word boundary
matches = re.findall(pattern, text)
print(matches)  # Output: ['World']

# Whitespace matching
text = "Hello\tWorld\n"
pattern = r"\s+"  # Matches one or more whitespace characters
matches = re.findall(pattern, text)
print(matches)  # Output: ['\t', '\n']

# Non-digit matching
text = "abc123def456"
pattern = r"\D+"  # Matches non-digits
matches = re.findall(pattern, text)
print(matches)  # Output: ['abc', 'def']
```

### Quantifiers

```
    Greedy vs Lazy Quantifiers:
    
    a+       a+?
     ↓        ↓
   Greedy    Lazy
   (max)    (min)
    
    *  → 0 or more    *?  → 0 or more (lazy)
    +  → 1 or more    +?  → 1 or more (lazy)
    ?  → 0 or 1       ??  → 0 or 1 (lazy)
    {n}   → exactly n
    {n,}  → n or more   {n,}?  → n or more (lazy)
    {n,m} → n to m      {n,m}? → n to m (lazy)
```

*   `*`: Matches 0 or more occurrences.
*   `+`: Matches 1 or more occurrences.
*   `?`: Matches 0 or 1 occurrence.
*   `{n}`: Matches exactly `n` occurrences.
*   `{n,}`: Matches `n` or more occurrences.
*   `{n,m}`: Matches between `n` and `m` occurrences (inclusive).
*   `*?`, `+?`, `??`, `{n,}?`, `{n,m}?`: Non-greedy (lazy) versions.

```python
import re

# One or more
text = "aaabbbccc"
pattern = r"a+"  # Matches one or more 'a's
matches = re.findall(pattern, text)
print(matches)  # Output: ['aaa']

# Exact range
text = "ab abb abbb abbbb"
pattern = r"ab{2,4}"  # Matches 'ab' with 2 to 4 'b's
matches = re.findall(pattern, text)
print(matches)  # Output: ['abb', 'abbb', 'abbbb']

# Optional character
text = "color colour"
pattern = r"colou?r"  # Matches 'color' or 'colour'
matches = re.findall(pattern, text)
print(matches)  # Output: ['color', 'colour']

# Greedy vs lazy
text = "<div>content</div><div>more</div>"
greedy = re.findall(r"<div>.*</div>", text)
print(greedy)  # Output: ['<div>content</div><div>more</div>']

lazy = re.findall(r"<div>.*?</div>", text)
print(lazy)  # Output: ['<div>content</div>', '<div>more</div>']

# Exact count
text = "The year is 2024"
pattern = r"\d{4}"  # Matches exactly 4 digits
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: 2024
```

### Anchors

```
    String Anchors:
    
    ^───────────────text───────────────$
    ↑                                  ↑
    Start                             End
    
    ^      → Start of string/line
    $      → End of string/line
    \A     → Start of string only
    \Z     → End of string (before final newline)
    \z     → End of string (absolute)
    \b     → Word boundary
    \B     → Non-word boundary
```

*   `^`: Matches the beginning of the string (or line).
*   `$`: Matches the end of the string (or line).
*   `\A`: Matches the beginning of the string.
*   `\Z`: Matches the end of the string, or before a newline at the end.
*   `\z`: Matches the end of the string.

```python
import re

# Start anchor
text = "hello world"
pattern = r"^hello"  # Matches 'hello' at the beginning
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: hello

# End anchor
text = "world\nhello"
pattern = r"hello$"  # Matches 'hello' at the end
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: hello

# Both anchors (exact match)
text = "hello"
pattern = r"^hello$"  # Matches entire string
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: hello

# String start (\A)
text = "hello world\n"
pattern = r"\Ahello"
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: hello

# String end before newline (\Z)
text = "hello world\n"
pattern = r"world\Z"
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: world

# Absolute string end (\z)
text = "hello world"
pattern = r"world\z"
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: world
```

### Alternation

```
    Alternation Flow:
    
         Pattern
            │
      ┌─────┴─────┐
      │           │
      ↓           ↓
    cat         dog
      │           │
      └─────┬─────┘
            ↓
         Match
```

*   `|`: Matches either the expression before or the expression after the `|`.

```python
import re

# Simple alternation
text = "cat and dog"
pattern = r"cat|dog"  # Matches 'cat' or 'dog'
matches = re.findall(pattern, text)
print(matches)  # Output: ['cat', 'dog']

# Multiple alternatives
text = "I like apples, oranges, and bananas"
pattern = r"apples|oranges|bananas"
matches = re.findall(pattern, text)
print(matches)  # Output: ['apples', 'oranges', 'bananas']

# Alternation with groups
text = "file.txt file.pdf file.doc"
pattern = r"file\.(txt|pdf|doc)"
matches = re.findall(pattern, text)
print(matches)  # Output: ['txt', 'pdf', 'doc']

# Complex alternation
text = "http://example.com https://secure.com"
pattern = r"https?://[\w.]+"
matches = re.findall(pattern, text)
print(matches)  # Output: ['http://example.com', 'https://secure.com']
```

### Grouping and Capturing

*   `( )`: Groups expressions and creates a capturing group.
*   `(?: )`: Groups expressions *without* creating a capturing group.
*   `\1`, `\2`, etc.: Backreferences to captured groups.

```python
import re

text = "apple apple"
pattern = r"(\w+) \1"  # Matches repeated words
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match")  # Output: apple apple
print(match.group(1)) if match else print("No match")  # Output: apple

text = "12-34-56"
pattern = r"(\d{2})-(?:(\d{2})-(\d{2}))"  # Non-capturing group for the middle part
match = re.search(pattern, text)
if match:
    print(match.groups())  # Output: ('12', '34', '56')
```

### Lookarounds

```
    Lookahead & Lookbehind:
    
    (?=...)   Positive Lookahead
    (?!...)   Negative Lookahead
    (?<=...)  Positive Lookbehind
    (?<!...)  Negative Lookbehind
    
    Example: \w+(?=\d)
            ↑    ↑
            │    └─→ Must be followed by digit
            └──────→ Matches word chars
```

*   `(?= )`: Positive lookahead - matches if followed by pattern.
*   `(?! )`: Negative lookahead - matches if NOT followed by pattern.
*   `(?<= )`: Positive lookbehind - matches if preceded by pattern.
*   `(?<! )`: Negative lookbehind - matches if NOT preceded by pattern.

```python
import re

# Positive lookahead
text = "apple123 banana456"
pattern = r"[a-zA-Z]+(?=\d)"  # Matches letters followed by digits
matches = re.findall(pattern, text)
print(matches)  # Output: ['apple', 'banana']

# Negative lookahead
text = "file1.txt file2.pdf file3.txt"
pattern = r"file\d+(?!\.txt)"  # Matches files NOT ending with .txt
matches = re.findall(pattern, text)
print(matches)  # Output: ['file2']

# Positive lookbehind
text = "$100 €200 $300"
pattern = r"(?<=\$)\d+"  # Matches numbers preceded by dollar sign
matches = re.findall(pattern, text)
print(matches)  # Output: ['100', '300']

# Negative lookbehind
text = "cat bat rat mat"
pattern = r"(?<!c)at"  # Matches 'at' NOT preceded by 'c'
matches = re.findall(pattern, text)
print(matches)  # Output: ['at', 'at', 'at']

# Password validation (lookahead)
password = "Abc123!@"
# Must contain uppercase, lowercase, digit, and special char
pattern = r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
match = re.match(pattern, password)
print("Valid" if match else "Invalid")  # Output: Valid

# Extract price without currency symbol
text = "Price: $50.99"
pattern = r"(?<=\$)\d+\.\d+"
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: 50.99
```

### Flags (Modifiers)

```
    Common Flags:
    
    re.IGNORECASE (re.I)  → Case-insensitive
    re.MULTILINE (re.M)   → ^ and $ match line boundaries
    re.DOTALL (re.S)      → . matches newlines
    re.VERBOSE (re.X)     → Allow comments and whitespace
    re.ASCII (re.A)       → ASCII-only matching
    re.UNICODE (re.U)     → Unicode matching (default in Python 3)
```

*   `i` (`re.IGNORECASE`): Case-insensitive matching.
*   `m` (`re.MULTILINE`): `^` and `$` match line boundaries.
*   `s` (`re.DOTALL`): `.` matches newline characters.
*   `x` (`re.VERBOSE`): Allow whitespace and comments in pattern.
*   `a` (`re.ASCII`): ASCII-only matching.
*   `u` (`re.UNICODE`): Full Unicode matching.

```python
import re

# Case-insensitive
text = "Hello World"
pattern = r"world"
match = re.search(pattern, text, re.IGNORECASE)
print(match.group(0) if match else "No match")  # Output: World

# Multiline mode
text = "Line 1\nLine 2\nLine 3"
pattern = r"^Line"
matches = re.findall(pattern, text, re.MULTILINE)
print(matches)  # Output: ['Line', 'Line', 'Line']

# Dotall mode
text = "Hello\nWorld"
pattern = r"Hello.World"
match = re.search(pattern, text, re.DOTALL)
print(match.group(0) if match else "No match")  # Output: Hello\nWorld

# Verbose mode
pattern = re.compile(r"""
    ^                 # Start of string
    [a-zA-Z0-9._%+-]+ # Local part
    @                 # At symbol
    [a-zA-Z0-9.-]+    # Domain
    \.                # Dot
    [a-zA-Z]{2,}      # TLD
    $                 # End of string
""", re.VERBOSE)
email = "test@example.com"
match = pattern.match(email)
print("Valid" if match else "Invalid")  # Output: Valid

# Combining multiple flags
text = "Hello\nWORLD"
pattern = r"hello.*world"
match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
print(match.group(0) if match else "No match")  # Output: Hello\nWORLD
```

### Character Properties (Unicode)

Note: Python's `re` module doesn't directly support `\p{Property}` syntax. Use the `regex` module for full Unicode property support.

*   `\p{Property}`: Matches a character with the specified Unicode property.
*   `\P{Property}`: Matches a character *without* the specified Unicode property.

Common Unicode categories:
*   `\p{L}`: Letters
*   `\p{N}`: Numbers
*   `\p{P}`: Punctuation
*   `\p{S}`: Symbols
*   `\p{Z}`: Separators

```python
# Using regex module (not standard re)
try:
    import regex
    
    # Match Unicode letters
    text = "Hello 123 こんにちは 你好"
    pattern = regex.compile(r"\p{L}+")
    matches = pattern.findall(text)
    print(matches)  # Output: ['Hello', 'こんにちは', '你好']
    
    # Match Unicode numbers
    text = "Hello 123 ១២៣"
    pattern = regex.compile(r"\p{N}+")
    matches = pattern.findall(text)
    print(matches)  # Output: ['123', '១២៣']
    
except ImportError:
    print("Install regex module: pip install regex")

# Alternative using standard re module
import re

# Basic Unicode support
text = "Hello 123 こんにちは"
pattern = r"[^\W\d_]+"  # Matches letter sequences
matches = re.findall(pattern, text)
print(matches)  # Output: ['Hello', 'こんにちは']
```

### Examples

*   **Email:** `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`

```python
import re
email = "test@example.com"
pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
match = re.match(pattern, email)
print(bool(match))  # Output: True
```

*   **URL:** `^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$`

```python
import re
url = "https://www.example.com/path/to/page.html"
pattern = r"^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$"
match = re.match(pattern, url)
print(bool(match))  # Output: True
```

*   **IP Address:** `^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$`

```python
import re
ip = "192.168.1.1"
pattern = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
match = re.match(pattern, ip)
print(bool(match))  # Output: True
```

*   **Hex Color Code:** `^#?([a-fA-F0-9]{6}|[a-fA-F0-9]{3})$`

```python
import re
hex_code = "#FF0000"
pattern = r"^#?([a-fA-F0-9]{6}|[a-fA-F0-9]{3})$"
match = re.match(pattern, hex_code)
print(bool(match))  # Output: True
```

*   **Date (YYYY-MM-DD):** `^\d{4}-\d{2}-\d{2}$`

```python
import re
date = "2024-01-08"
pattern = r"^\d{4}-\d{2}-\d{2}$"
match = re.match(pattern, date)
print(bool(match))  # Output: True
```

*   **HTML Tag:** `<([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)`

```python
import re
html = "<p>This is a paragraph.</p>"
pattern = r"<([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)"
match = re.search(pattern, html)
print(match.group(1)) if match else print("No match")  # Output: p
```

*   **Phone Number (US):** `\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}`

```python
import re
phone = "(555) 123-4567"
pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
match = re.search(pattern, phone)
print(bool(match))  # Output: True
```

## Advanced Techniques

### Atomic Groups

Note: Python's `re` module doesn't support atomic groups `(?>...)`. This feature is available in other regex engines like PCRE.

*   `(?> )`: Atomic group (prevents backtracking).

```python
import re

# Greedy quantifier with backtracking
text = "aaaaaaaaab"
pattern = r"a+b"  # Regular + (will backtrack to find 'b')
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: aaaaaaaaab

# Non-greedy quantifier
text = "aaaaaaaaab"
pattern = r"a+?b"  # Non-greedy +?
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: ab

# Note: Atomic groups would prevent backtracking
# In engines that support it: (?>a+)b would fail to match
# because a+ consumes all 'a's and doesn't backtrack
```

### Recursive Patterns

Note: Python's `re` module doesn't support recursive patterns. Use the `regex` module or alternative approaches.

*   `(?R)` or `(?0)`: Recursively matches the entire pattern.
*   `(?1)`, `(?2)`, etc.: Recursively matches a specific capturing group.

```python
# Using regex module for recursive patterns
try:
    import regex
    
    # Match balanced parentheses
    text = "((abc)def(ghi))"
    pattern = regex.compile(r"\(([^()]|(?R))*\)")
    match = pattern.search(text)
    print(match.group(0) if match else "No match")  # Output: ((abc)def(ghi))
    
except ImportError:
    print("Install regex module: pip install regex")

# Alternative using standard re module (limited depth)
import re

# Match one level of nested parentheses
text = "(abc(def)ghi)"
pattern = r"\([^()]*(?:\([^()]*\)[^()]*)*\)"
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: (abc(def)ghi)
```

### Conditional Expressions

Note: Python's `re` module has limited support for conditional patterns.

*   `(?(id/name)yes-pattern|no-pattern)`: Matches based on whether a group matched.

```python
import re

# Conditional based on captured group
text1 = "abc123"
text2 = "xyz456"
# If group 1 exists, match digits; otherwise, match letters
pattern = r"^(abc)?(?(1)\d+|\w+)$"

match1 = re.match(pattern, text1)
print(f"{text1}: {match1.group(0) if match1 else 'No match'}")  # Output: abc123: abc123

match2 = re.match(pattern, text2)
print(f"{text2}: {match2.group(0) if match2 else 'No match'}")  # Output: xyz456: xyz456

# Match email with optional angle brackets
text = "<user@example.com>"
pattern = r"^(<)?(\S+@\S+)(?(1)>)$"
match = re.match(pattern, text)
print(match.group(2) if match else "No match")  # Output: user@example.com
```

### Named Capture Groups

```
    Named Groups:
    
    (?P<name>...)
       │
       └─→ Name for capturing group
    
    (?P=name)  → Backreference by name
    \g<name>   → Replacement reference
```

*   `(?P<name> )`: Creates a named capturing group.
*   `(?P=name)`: Backreference to a named capturing group.

```python
import re

# Named groups for repeated words
text = "apple apple"
pattern = r"(?P<word>\w+) (?P=word)"  # Matches repeated words
match = re.search(pattern, text)
if match:
    print(match.group('word'))  # Output: apple
    print(match.group(0))       # Output: apple apple

# Extract structured data
text = "Date: 2024-01-08"
pattern = r"Date: (?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
match = re.search(pattern, text)
if match:
    print(f"Year: {match.group('year')}")    # Output: Year: 2024
    print(f"Month: {match.group('month')}")  # Output: Month: 01
    print(f"Day: {match.group('day')}")      # Output: Day: 08
    print(match.groupdict())  # Output: {'year': '2024', 'month': '01', 'day': '08'}

# Using named groups in substitution
text = "John Doe"
pattern = r"(?P<first>\w+) (?P<last>\w+)"
result = re.sub(pattern, r"\g<last>, \g<first>", text)
print(result)  # Output: Doe, John
```

### Comments

*   `(?#comment)`: Inline comment.

```python
import re

text = "123-4567"
pattern = r"\d{3}(?#This is a comment)-?\d{4}"
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match")  # Output: 123-4567
```

### Free-Spacing Mode (`x` flag)

```python
import re

text = "123-4567"
pattern = r"""
    \d{3}  # Area code
    -?     # Optional separator
    \d{4}  # Phone number
"""
match = re.search(pattern, text, re.VERBOSE)  # Use re.VERBOSE or re.X
print(match.group(0)) if match else print("No match")  # Output: 123-4567
```

### Branch Reset Groups

Note: Branch reset groups `(?|...)` are not supported in Python's `re` module.

This feature is available in PCRE and some other regex engines. It resets capture group numbering within alternatives.

### Subroutine Calls

Note: Subroutine calls `(?&name)` are not supported in Python's `re` module.

For complex nested patterns, consider using the `regex` module or parsing libraries.

## Python `re` Module Methods

```
    Python re Module Workflow:
    
    ┌─────────────┐
    │   Pattern    │
    └──────┬──────┘
           │
      ┌────┴────┐
      ↓          ↓
   compile    Direct
      │          │
      └───┬─────┘
          ↓
    ┌───────────┐
    │  Methods:  │
    │  search    │
    │  match     │
    │  findall   │
    │  finditer  │
    │  sub       │
    │  split     │
    └───────────┘
```

### re.search() - Find First Match

```python
import re

text = "The price is $19.99 and discount is 25%"

# Find first occurrence
match = re.search(r"\d+\.?\d*", text)
if match:
    print(f"Found: {match.group(0)}")  # Output: Found: 19.99
    print(f"Position: {match.start()}-{match.end()}")  # Output: Position: 13-18
```

### re.match() - Match from Beginning

```python
import re

text = "Python 3.12"

# Match only at the start of string
match = re.match(r"Python", text)
print(match.group(0) if match else "No match")  # Output: Python

# Won't match if pattern not at start
match = re.match(r"3\.12", text)
print(match.group(0) if match else "No match")  # Output: No match
```

### re.fullmatch() - Match Entire String

```python
import re

# Validate entire string matches pattern
text = "12345"

match = re.fullmatch(r"\d+", text)
print("Valid" if match else "Invalid")  # Output: Valid

text = "123abc"
match = re.fullmatch(r"\d+", text)
print("Valid" if match else "Invalid")  # Output: Invalid
```

### re.findall() - Find All Matches

```python
import re

text = "Email: john@example.com, jane@test.org"

# Find all email addresses
emails = re.findall(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)
print(emails)  # Output: ['john@example.com', 'jane@test.org']

# With groups - returns tuples
text = "ID: 123, Name: John; ID: 456, Name: Jane"
pattern = r"ID: (\d+), Name: (\w+)"
results = re.findall(pattern, text)
print(results)  # Output: [('123', 'John'), ('456', 'Jane')]
```

### re.finditer() - Iterator of Matches

```python
import re

text = "Phone: 555-1234, 555-5678"
pattern = r"(\d{3})-(\d{4})"

# Iterate over all matches
for match in re.finditer(pattern, text):
    print(f"Full: {match.group(0)}")
    print(f"Area: {match.group(1)}, Number: {match.group(2)}")
    print(f"Position: {match.start()}-{match.end()}")
    print()
# Output:
# Full: 555-1234
# Area: 555, Number: 1234
# Position: 7-15
# 
# Full: 555-5678
# Area: 555, Number: 5678
# Position: 17-25
```

### re.sub() - Replace Matches

```python
import re

text = "Hello World, Hello Universe"

# Simple replacement
result = re.sub(r"Hello", "Hi", text)
print(result)  # Output: Hi World, Hi Universe

# Replace with count limit
result = re.sub(r"Hello", "Hi", text, count=1)
print(result)  # Output: Hi World, Hello Universe

# Using backreferences
text = "John Doe, Jane Smith"
result = re.sub(r"(\w+) (\w+)", r"\2, \1", text)
print(result)  # Output: Doe, John, Smith, Jane

# Using function for replacement
def uppercase_match(match):
    return match.group(0).upper()

text = "hello world"
result = re.sub(r"\w+", uppercase_match, text)
print(result)  # Output: HELLO WORLD

# Remove extra whitespace
text = "Too   many    spaces"
result = re.sub(r"\s+", " ", text)
print(result)  # Output: Too many spaces
```

### re.subn() - Replace with Count

```python
import re

text = "cat bat rat mat"

# Returns tuple (new_string, number_of_substitutions)
result, count = re.subn(r"[cbr]at", "hat", text)
print(f"Result: {result}")  # Output: Result: hat hat hat mat
print(f"Replacements: {count}")  # Output: Replacements: 3
```

### re.split() - Split by Pattern

```python
import re

# Split on whitespace
text = "apple banana cherry"
parts = re.split(r"\s+", text)
print(parts)  # Output: ['apple', 'banana', 'cherry']

# Split on multiple delimiters
text = "apple,banana;cherry:orange"
parts = re.split(r"[,;:]", text)
print(parts)  # Output: ['apple', 'banana', 'cherry', 'orange']

# Split with limit
text = "one two three four five"
parts = re.split(r"\s+", text, maxsplit=2)
print(parts)  # Output: ['one', 'two', 'three four five']

# Keep delimiters using groups
text = "apple,banana;cherry"
parts = re.split(r"([,;])", text)
print(parts)  # Output: ['apple', ',', 'banana', ';', 'cherry']
```

### re.compile() - Compile Pattern

```python
import re

# Compile for reuse (more efficient)
pattern = re.compile(r"\b\w+@\w+\.\w+\b", re.IGNORECASE)

text1 = "Contact: john@example.com"
text2 = "Email: JANE@TEST.ORG"

match1 = pattern.search(text1)
match2 = pattern.search(text2)

print(match1.group(0) if match1 else "No match")  # Output: john@example.com
print(match2.group(0) if match2 else "No match")  # Output: JANE@TEST.ORG

# Access compiled pattern properties
print(pattern.pattern)  # Output: \b\w+@\w+\.\w+\b
print(pattern.flags)    # Output: re.IGNORECASE value
```

### re.escape() - Escape Special Characters

```python
import re

# Escape special regex characters
user_input = "What is $100 + $50?"
escaped = re.escape(user_input)
print(escaped)  # Output: What\ is\ \$100\ \+\ \$50\?

# Safe pattern from user input
text = "The price is $100"
user_search = "$100"
pattern = re.escape(user_search)
match = re.search(pattern, text)
print(match.group(0) if match else "No match")  # Output: $100
```

### Match Object Methods

```python
import re

text = "Contact: John Doe (john@example.com)"
pattern = r"(\w+) (\w+) \(([\w@.]+)\)"
match = re.search(pattern, text)

if match:
    # Get matched string
    print(match.group(0))   # Output: John Doe (john@example.com)
    print(match.group(1))   # Output: John
    print(match.group(2))   # Output: Doe
    print(match.group(3))   # Output: john@example.com
    
    # Get all groups
    print(match.groups())   # Output: ('John', 'Doe', 'john@example.com')
    
    # Get positions
    print(match.start())    # Output: 9
    print(match.end())      # Output: 38
    print(match.span())     # Output: (9, 38)
    
    # Get span of specific group
    print(match.span(1))    # Output: (9, 13)
    
    # Named groups
    pattern = r"(?P<first>\w+) (?P<last>\w+)"
    match = re.search(pattern, "John Doe")
    print(match.group('first'))  # Output: John
    print(match.groupdict())     # Output: {'first': 'John', 'last': 'Doe'}
```

## Common RegEx Engines and Differences

*   **PCRE (Perl Compatible Regular Expressions):** Widely used, feature-rich.
*   **JavaScript:** Good support, but lookbehind assertions were limited (now widely supported).
*   **Python (`re` module):** Excellent support, including Unicode properties.
*   **.NET:** Powerful and feature-rich.
*   **Java:** Good support, some syntax differences.
*   **POSIX:** Basic and Extended Regular Expressions (BRE and ERE). Limited features.

## Best Practices

```
    RegEx Best Practices:
    
    ┌────────────────┐
    │  1. Be Specific │
    └───────┬────────┘
           ↓
    ┌────────────────┐
    │  2. Optimize   │
    └───────┬────────┘
           ↓
    ┌────────────────┐
    │  3. Test      │
    └───────┬────────┘
           ↓
    ┌────────────────┐
    │  4. Document  │
    └────────────────┘
```

### 1. Be Specific and Precise

*   **Avoid overly broad patterns:** Use `\d{4}` instead of `.{4}` for 4 digits.
*   **Use anchors:** Add `^` and `$` for exact matches.
*   **Be explicit:** `[0-9]` is clearer than `\d` when you want only ASCII digits.

```python
import re

# Bad: Too broad
pattern_bad = r".+@.+\..+"

# Good: Specific
pattern_good = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
```

### 2. Optimize Performance

*   **Use character classes:** `\d` is more efficient than `[0-9]`.
*   **Use non-capturing groups:** `(?:...)` when you don't need captured text.
*   **Compile patterns:** Use `re.compile()` for repeated use.
*   **Be aware of greediness:** Use lazy quantifiers (`*?`, `+?`) when appropriate.

```python
import re

# Bad: Capturing unnecessary groups
pattern_bad = r"(http|https)://([\w.]+)/(.+)"

# Good: Non-capturing groups
pattern_good = r"(?:http|https)://[\w.]+/.+"

# Compile for reuse
compiled = re.compile(pattern_good)
for url in urls:
    compiled.match(url)
```

### 3. Avoid Catastrophic Backtracking

*   **Avoid nested quantifiers:** `(a+)+` or `(a*)*` can cause exponential backtracking.
*   **Use atomic groups or possessive quantifiers:** (if available in your engine).
*   **Be careful with alternation:** `(a|ab)+` can be slow.

```python
import re

# Dangerous: Can cause catastrophic backtracking
# pattern_dangerous = r"(a+)+b"

# Safe alternatives:
pattern_safe1 = r"a+b"
pattern_safe2 = r"(?:a+)b"
```

### 4. Test Thoroughly

*   **Use online tools:** regex101.com, regexr.com, pythex.org
*   **Test edge cases:** Empty strings, special characters, very long strings.
*   **Use unit tests:** Validate against known inputs/outputs.

```python
import re

def validate_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))

# Test cases
assert validate_email("test@example.com") == True
assert validate_email("invalid@") == False
assert validate_email("@invalid.com") == False
assert validate_email("") == False
```

### 5. Document Complex Patterns

*   **Use verbose mode:** `re.VERBOSE` or `re.X` flag.
*   **Add comments:** Explain what each part does.
*   **Break into parts:** Define subpatterns separately.

```python
import re

# Bad: Hard to read
pattern_bad = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"

# Good: Well documented
pattern_good = re.compile(r"""
    ^                      # Start of string
    (?=.*[a-z])           # At least one lowercase letter
    (?=.*[A-Z])           # At least one uppercase letter
    (?=.*\d)              # At least one digit
    (?=.*[@$!%*?&])       # At least one special character
    [A-Za-z\d@$!%*?&]{8,} # At least 8 characters from allowed set
    $                      # End of string
""", re.VERBOSE)
```

### 6. Use Raw Strings in Python

*   **Always use `r"..."` for patterns:** Prevents double-escaping issues.
*   **Escape special characters properly:** Use `\.` for literal dots.

```python
import re

# Bad: Need double escaping
pattern_bad = "\\d+\\.\\d+"  # Confusing!

# Good: Raw string
pattern_good = r"\d+\.\d+"  # Clear and correct
```

### 7. Consider Alternatives

*   **String methods:** For simple cases, use `str.startswith()`, `str.endswith()`, `str.split()`.
*   **Parsing libraries:** For structured data (JSON, XML, CSV).
*   **Avoid regex for:** HTML/XML parsing, programming language parsing.

```python
# Bad: Using regex for simple check
import re
if re.match(r"^Hello", text):
    pass

# Good: Using string method
if text.startswith("Hello"):
    pass
```

### 8. Know Your Engine

*   **Python `re`:** Good standard support, lacks some advanced features.
*   **Python `regex`:** Extended features (Unicode properties, recursion).
*   **PCRE:** Full-featured, used in many languages.
*   **JavaScript:** Good support, some differences in lookbehind.

### 9. Security Considerations

*   **Validate input length:** Prevent DoS from long strings.
*   **Sanitize user input:** Escape special characters with `re.escape()`.
*   **Set timeout limits:** For untrusted patterns.

```python
import re

# Safe: Escape user input
user_input = "$100"
escaped = re.escape(user_input)
pattern = re.compile(escaped)

# Validate input length
if len(text) > 10000:
    raise ValueError("Input too long")
```

### 10. Common Pitfalls to Avoid

*   **Forgetting anchors:** `\d+` matches "abc123" at position 3.
*   **Not escaping special chars:** `.` matches any character, not just dot.
*   **Greedy vs lazy confusion:** `.*` vs `.*?`
*   **Character class mistakes:** `[A-z]` includes more than `[A-Za-z]`.
*   **Word boundaries:** `\b` doesn't work with non-ASCII characters as expected.

```python
import re

# Common mistakes:
text = "test@example.com"

# Wrong: Doesn't escape dot
pattern_wrong = r".+@.+..+"  # Matches too much

# Correct: Escapes dot
pattern_correct = r".+@.+\..+"

# Wrong: Character class includes extra chars
pattern_wrong2 = r"[A-z]+"  # Includes [\]^_`

# Correct: Explicit ranges
pattern_correct2 = r"[A-Za-z]+"
```

## Quick Reference Summary

```
    ┌────────────────────────────────────────────┐
    │           REGEX QUICK REFERENCE          │
    ├────────────────────────────────────────────┤
    │ BASIC                                    │
    ├────────────────────────────────────────────┤
    │ .         Any character except \n       │
    │ \d        Digit [0-9]                  │
    │ \w        Word char [a-zA-Z0-9_]       │
    │ \s        Whitespace [ \t\n\r\f\v]      │
    │ \D \W \S   Negated versions             │
    ├────────────────────────────────────────────┤
    │ QUANTIFIERS                              │
    ├────────────────────────────────────────────┤
    │ *         0 or more (greedy)            │
    │ +         1 or more (greedy)            │
    │ ?         0 or 1                        │
    │ {n}       Exactly n times               │
    │ {n,m}     Between n and m times         │
    │ *? +? ??  Lazy versions                 │
    ├────────────────────────────────────────────┤
    │ ANCHORS                                  │
    ├────────────────────────────────────────────┤
    │ ^         Start of string/line           │
    │ $         End of string/line             │
    │ \b        Word boundary                  │
    │ \B        Not word boundary              │
    ├────────────────────────────────────────────┤
    │ GROUPS                                   │
    ├────────────────────────────────────────────┤
    │ (...)     Capturing group                │
    │ (?:...)   Non-capturing group            │
    │ (?P<n>...) Named group                   │
    │ \1 \2     Backreference                  │
    │ (?P=n)    Named backreference            │
    ├────────────────────────────────────────────┤
    │ LOOKAROUND                               │
    ├────────────────────────────────────────────┤
    │ (?=...)   Positive lookahead             │
    │ (?!...)   Negative lookahead             │
    │ (?<=...)  Positive lookbehind            │
    │ (?<!...)  Negative lookbehind            │
    ├────────────────────────────────────────────┤
    │ FLAGS                                    │
    ├────────────────────────────────────────────┤
    │ re.I      Case-insensitive               │
    │ re.M      Multiline ^ $ anchors          │
    │ re.S      Dot matches newline            │
    │ re.X      Verbose mode (comments)        │
    └────────────────────────────────────────────┘
```

### Essential Python Methods

| Method | Description | Returns |
|--------|-------------|----------|
| `re.search(pattern, string)` | Find first match anywhere | Match object or None |
| `re.match(pattern, string)` | Match at start of string | Match object or None |
| `re.fullmatch(pattern, string)` | Match entire string | Match object or None |
| `re.findall(pattern, string)` | Find all matches | List of strings |
| `re.finditer(pattern, string)` | Iterator of matches | Iterator of Match objects |
| `re.sub(pattern, repl, string)` | Replace matches | New string |
| `re.split(pattern, string)` | Split by pattern | List of strings |
| `re.compile(pattern)` | Compile pattern | Compiled pattern object |

### Common Patterns Quick Guide

```python
import re

# Email
r"^[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}$"

# URL
r"^https?://[\w.-]+\.[a-zA-Z]{2,}(/.*)?$"

# Phone (US)
r"^\d{3}-\d{3}-\d{4}$"

# IP Address
r"^(?:\d{1,3}\.){3}\d{1,3}$"

# Date YYYY-MM-DD
r"^\d{4}-\d{2}-\d{2}$"

# Hex Color
r"^#[a-fA-F0-9]{6}$"

# Username (3-16 alphanumeric)
r"^[a-zA-Z0-9_]{3,16}$"

# Strong Password
r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
```