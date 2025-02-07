
---
title: Regular Expressions (RegEx) Cheat Sheet
description: A comprehensive reference guide for regular expressions, covering syntax, character classes, quantifiers, anchors, groups, and more, with Python examples.
---

# Regular Expressions (RegEx) Cheat Sheet

[TOC]

This cheat sheet provides an exhaustive overview of regular expressions (RegEx), covering syntax, character classes, quantifiers, anchors, groups, flags, and advanced techniques. It aims to be a complete reference for using regular expressions, with a focus on Python examples.

## Basic Syntax

### Literals

Characters match themselves literally, except for special characters.

*   `abc`: Matches the literal string "abc".

```python
import re
text = "abc def ghi"
pattern = r"abc"
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match")  # Output: abc
```

### Special Characters

These characters have special meanings in RegEx:

*   `. ^ $ * + ? { } [ ] \ | ( )`

To match these characters literally, escape them with a backslash (`\`):

*   `\.`: Matches a literal dot (`.`).
*   `\\`: Matches a literal backslash (`\`).

```python
import re
text = "123.456"
pattern = r"\."  # Matches a literal dot
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match")  # Output: .

text = "path\\to\\file"
pattern = r"\\" # Matches a literal backslash
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match") # Output: \
```

### Character Classes

*   `.`: Matches any character except a newline (unless the `s` flag is used).
*   `[abc]`: Matches any character inside the brackets (a, b, or c).
*   `[^abc]`: Matches any character *not* inside the brackets.
*   `[a-z]`: Matches any character in the range a to z (lowercase).
*   `[A-Z]`: Matches any character in the range A to Z (uppercase).
*   `[0-9]`: Matches any digit.
*   `[a-zA-Z0-9]`: Matches any alphanumeric character.

```python
import re

text = "apple banana cherry"
pattern = r"[abc]"  # Matches 'a', 'b', or 'c'
matches = re.findall(pattern, text)
print(matches)  # Output: ['a', 'a', 'b', 'a', 'a', 'a', 'a', 'c']

text = "apple banana cherry"
pattern = r"[^abc]"  # Matches any character except 'a', 'b', or 'c'
matches = re.findall(pattern, text)
print(matches)  # Output: ['p', 'p', 'l', 'e', ' ', 'n', 'n', ' ', 'h', 'e', 'r', 'r', 'y']

text = "apple1 banana2 cherry3"
pattern = r"[0-9]"  # Matches any digit
matches = re.findall(pattern, text)
print(matches)  # Output: ['1', '2', '3']

text = "Hello. World!"
pattern = r"." # Matches any character except newline
matches = re.findall(pattern, text)
print(matches) # Output: ['H', 'e', 'l', 'l', 'o', '.', ' ', 'W', 'o', 'r', 'l', 'd', '!']
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

```python
import re

text = "123 abc 456"
pattern = r"\d+"  # Matches one or more digits
matches = re.findall(pattern, text)
print(matches)  # Output: ['123', '456']

text = "Hello World"
pattern = r"\bWorld\b"  # Matches "World" at a word boundary
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match")  # Output: World

text = "Hello\tWorld\n"
pattern = r"\s+"  # Matches one or more whitespace characters
matches = re.findall(pattern, text)
print(matches)  # Output: ['\t', '\n']
```

### Quantifiers

*   `*`: Matches 0 or more occurrences.
*   `+`: Matches 1 or more occurrences.
*   `?`: Matches 0 or 1 occurrence.
*   `{n}`: Matches exactly `n` occurrences.
*   `{n,}`: Matches `n` or more occurrences.
*   `{n,m}`: Matches between `n` and `m` occurrences (inclusive).
*   `*?`, `+?`, `??`, `{n,}?`, `{n,m}?`: Non-greedy (lazy) versions.

```python
import re

text = "aaabbbccc"
pattern = r"a+"  # Matches one or more 'a's
matches = re.findall(pattern, text)
print(matches)  # Output: ['aaa']

text = "ab abb abbb"
pattern = r"ab{2,4}"  # Matches 'ab' with 2 to 4 'b's
matches = re.findall(pattern, text)
print(matches)  # Output: ['abb', 'abbb']

text = "color colour"
pattern = r"colou?r"  # Matches 'color' or 'colour'
matches = re.findall(pattern, text)
print(matches)  # Output: ['color', 'colour']

text = "aaaa"
pattern = r"a*?" # Matches 0 or more 'a', non-greedy
matches = re.findall(pattern, text)
print(matches) # Output: ['', 'a', '', 'a', '', 'a', '', 'a', '']
```

### Anchors

*   `^`: Matches the beginning of the string (or line).
*   `$`: Matches the end of the string (or line).
*   `\A`: Matches the beginning of the string.
*   `\Z`: Matches the end of the string, or before a newline at the end.
*   `\z`: Matches the end of the string.

```python
import re

text = "hello world"
pattern = r"^hello"  # Matches 'hello' at the beginning
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match")  # Output: hello

text = "world\nhello"
pattern = r"hello$"  # Matches 'hello' at the end
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match") # Output: hello

text = "hello world\n"
pattern = r"\Ahello"
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match") # Output: hello

text = "hello world\n"
pattern = r"world\Z"
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match") # Output: world

text = "hello world"
pattern = r"world\z"
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match") # Output: world
```

### Alternation

*   `|`: Matches either the expression before or the expression after the `|`.

```python
import re

text = "cat and dog"
pattern = r"cat|dog"  # Matches 'cat' or 'dog'
matches = re.findall(pattern, text)
print(matches)  # Output: ['cat', 'dog']
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

*   `(?= )`: Positive lookahead.
*   `(?! )`: Negative lookahead.
*   `(?<= )`: Positive lookbehind.
*   `(?<! )`: Negative lookbehind.

```python
import re

text = "apple123 banana456"
pattern = r"\w+(?=\d+)"  # Matches words followed by digits (positive lookahead)
matches = re.findall(pattern, text)
print(matches)  # Output: ['apple', 'banana']

text = "apple banana cherry"
pattern = r"\w+(?!e)"  # Matches words NOT followed by 'e' (negative lookahead)
matches = re.findall(pattern, text)
print(matches)  # Output: ['appl', 'banan', 'banan', 'cherr', 'cherr']

text = "123apple 456banana"
pattern = r"(?<=\d+)\w+"  # Matches words preceded by digits (positive lookbehind)
matches = re.findall(pattern, text)
print(matches)  # Output: ['apple', 'banana']

text = "apple banana cherry"
pattern = r"(?<!a)\w+"  # Matches words NOT preceded by 'a' (negative lookbehind)
matches = re.findall(pattern, text)
print(matches)  # Output: ['pple', 'banana', 'cherry']
```

### Flags (Modifiers)

*   `i`: Case-insensitive.
*   `m`: Multiline.
*   `s`: Dotall (single-line).
*   `g`: Global (find all matches).
*   `x`: Extended (allow whitespace and comments).
*   `u`: Unicode.

```python
import re

text = "Hello World"
pattern = r"world"
match = re.search(pattern, text, re.IGNORECASE)  # Case-insensitive
print(match.group(0)) if match else print("No match")  # Output: World

text = "Line 1\nLine 2\nLine 3"
pattern = r"^Line"
matches = re.findall(pattern, text, re.MULTILINE)  # Multiline
print(matches)  # Output: ['Line', 'Line', 'Line']

text = "Hello\nWorld"
pattern = r"Hello.World"
match = re.search(pattern, text, re.DOTALL)  # Dotall
print(match.group(0)) if match else print("No match")  # Output: Hello\nWorld
```

### Character Properties (Unicode)

*   `\p{Property}`: Matches a character with the specified Unicode property.
*   `\P{Property}`: Matches a character *without* the specified Unicode property.

```python
import re

text = "Hello 123 こんにちは"
pattern = r"\p{L}+"  # Matches one or more letters
matches = re.findall(pattern, text)
print(matches)  # Output: ['Hello', 'こんにちは']

text = "Hello 123 こんにちは"
pattern = r"\p{N}+"  # Matches one or more numbers
matches = re.findall(pattern, text)
print(matches)  # Output: ['123']

text = "Hello 123 こんにちは"
pattern = r"\P{L}+"  # Matches one or more characters that are NOT letters
matches = re.findall(pattern, text)
print(matches) # Output: [' ', '123 ', ' ']
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

*   `(?> )`: Atomic group.

```python
import re

text = "aaaaaaaaab"
pattern = r"a+b"  # Regular +
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match")  # Output: ab

text = "aaaaaaaaab"
pattern = r"a+?b"  # Non-greedy +?
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match")  # Output: ab

text = "aaaaaaaaab"
pattern = r"(?>a+)b"  # Atomic group
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match")  # Output: No match (because a+ consumed all 'a's)
```

### Recursive Patterns

*   `(?R)` or `(?0)`: Recursively matches the entire pattern.
*   `(?1)`, `(?2)`, etc.: Recursively matches a specific capturing group.

```python
import re

text = "((abc)def(ghi))"
pattern = r"\(([^()]|(?R))*\)"  # Matches balanced parentheses
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match")  # Output: ((abc)def(ghi))
```

### Conditional Expressions

*   `(?(condition)yes-pattern|no-pattern)`

```python
import re

text = "ab"
pattern = r"(?(?<=a)b|c)"  # If preceded by 'a', match 'b'; otherwise, match 'c'.
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match")  # Output: b

text = "cb"
pattern = r"(?(?<=a)b|c)"  # If preceded by 'a', match 'b'; otherwise, match 'c'.
match = re.search(pattern, text)
print(match.group(0)) if match else print("No match")  # Output: c
```

### Named Capture Groups

*   `(?P<name> )`: Creates a named capturing group.
*   `(?P=name)`: Backreference to a named capturing group.

```python
import re

text = "apple apple"
pattern = r"(?P<word>\w+) (?P=word)"  # Matches repeated words
match = re.search(pattern, text)
print(match.group('word')) if match else print("No match")  # Output: apple
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

*   `(?| )`: Resets the capture group numbering within each alternative.

```python
import re

text = "abc"
pattern = r"(?|(a)|(b)|(c))"  # All alternatives capture to group 1
match = re.search(pattern, text)
print(match.group(1)) if match else print("No match")  # Output: a

text = "def"
pattern = r"(?|(a)|(b)|(c))"  # All alternatives capture to group 1
match = re.search(pattern, text)
print(match.group(1)) if match else print("No match")  # Output: None
```

### Subroutine Calls

*   `(?&name)`: Calls a named subroutine.

```python
import re
text = "((abc)def(ghi))"
pattern = r"""
(?(DEFINE)
  (?P<paren>\(([^()]|(?&paren))*\))  # Define a named subroutine 'paren'
)
^(?&paren)$  # Call the subroutine
"""
match = re.search(pattern, text, re.VERBOSE)
print(match.group(0)) if match else print("No match")  # Output: ((abc)def(ghi))
```

## Common RegEx Engines and Differences

*   **PCRE (Perl Compatible Regular Expressions):** Widely used, feature-rich.
*   **JavaScript:** Good support, but lookbehind assertions were limited (now widely supported).
*   **Python (`re` module):** Excellent support, including Unicode properties.
*   **.NET:** Powerful and feature-rich.
*   **Java:** Good support, some syntax differences.
*   **POSIX:** Basic and Extended Regular Expressions (BRE and ERE). Limited features.

## Best Practices

*   **Be specific:** Avoid overly broad patterns.
*   **Use character classes:** `\d` is more efficient than `[0-9]`.
*   **Use non-capturing groups `(?:...)` when you don't need the captured text.**
*   **Be aware of greediness:** Use non-greedy quantifiers (`*?`, `+?`, etc.).
*   **Test your regex:** Use online tools (regex101.com, regexr.com) or Python's `re` module interactively.
*   **Comment complex regexes:** Use the `x` flag (extended mode) for readability.
*   **Avoid catastrophic backtracking:** Be careful with nested quantifiers.
*   **Escape special characters:** Always escape special characters.
*   **Use raw strings in Python:** Use `r"\d+"` to avoid escaping backslashes.
*   **Consider alternatives:** Sometimes, regular expressions are not the best tool.
*   **Know your engine:** Be aware of the features and limitations of your RegEx engine.