
---
title: Data Structures and Algorithms (DSA)
description: Data Structures and Algorithms for cracking interviews
---

# Data Structures and Algorithms (DSA)

![Total Questions](https://img.shields.io/badge/Total%20Questions-25-blue?style=flat&labelColor=black&color=blue)
![Unanswered Questions](https://img.shields.io/badge/Unanswered%20Questions-0-blue?style=flat&labelColor=black&color=yellow)
![Answered Questions](https://img.shields.io/badge/Answered%20Questions-25-blue?style=flat&labelColor=black&color=success)

[TOC]

This document provides a curated list of Data Structures and Algorithms (DSA) questions commonly asked in technical interviews, along with solutions and explanations.  It covers a wide range of difficulty levels and topics.

---

## 😁 Easy

---

### Two Number Sum

!!! question

	Write a function that takes in a non-empty array of distinct integers and an
	integer representing a target sum.  If any two numbers in the input array sum
	up to the target sum, the function should return them in an array, in any
	order.  If no two numbers sum up to the target sum, the function should return
	an empty array.

	??? example "Try it!"
		- LeetCode: https://leetcode.com/problems/two-sum/

	??? success "Answer"
		```python
		# O(n) time | O(n) space
		def twoNumberSum(array, targetSum):
			seen = set()
			for num in array:
				complement = targetSum - num
				if complement in seen:
					return [complement, num]
				seen.add(num)
			return []
		```

		```python
		# O(nlog(n)) time | O(1) space
		def twoNumberSum(array, targetSum):
			array.sort()
			left = 0
			right = len(array) - 1
			while left < right:
				current_sum = array[left] + array[right]
				if current_sum == targetSum:
					return [array[left], array[right]]
				elif current_sum < targetSum:
					left += 1
				else:
					right -= 1
			return []
		```

		```python
		# O(n^2) time | O(1) space
		def twoNumberSum(array, targetSum):
			for i in range(len(array)):
				for j in range(i + 1, len(array)):
					if array[i] + array[j] == targetSum:
						return [array[i], array[j]]
			return []
		```

---

### Validate Subsequence

!!! question

	Given two non-empty arrays of integers, write a function that determines
	whether the second array is a subsequence of the first one.

	A subsequence of an array is a set of numbers that aren't necessarily adjacent
	in the array but that are in the same order as they appear in the array. For
	instance, the numbers `[1, 3, 4]` form a subsequence of the array `[1, 2, 3, 4]`,
	and so do the numbers `[2, 4]`. Note that a single number in an array and the
	array itself are both valid subsequences of the array.

	??? example "Try it!"
		- GeeksforGeeks: https://www.geeksforgeeks.org/problems/array-subset-of-another-array2317/1

	??? success "Answer"
		```python
		# O(n) time | O(1) space - where n is the length of the array
		def isValidSubsequence(array, sequence):
			arr_idx = 0
			seq_idx = 0
			while arr_idx < len(array) and seq_idx < len(sequence):
				if array[arr_idx] == sequence[seq_idx]:
					seq_idx += 1
				arr_idx += 1
			return seq_idx == len(sequence)
		```

		```python
		# Alternative, slightly more concise implementation
		def isValidSubsequence(array, sequence):
		    seqIdx = 0
		    for value in array:
		        if seqIdx == len(sequence):
		            return True
		        if sequence[seqIdx] == value:
		            seqIdx += 1
		    return seqIdx == len(sequence)
		```

---

### Nth Fibonacci

!!! question

	The Fibonacci sequence is defined as follows: the first number of the sequence
	is `0`, the second number is `1`, and the nth number is the sum of the (n - 1)th
	and (n - 2)th numbers. Write a function that takes in an integer `n` and returns
	the nth Fibonacci number.

	??? example "Try it!"
		- LeetCode: https://leetcode.com/problems/fibonacci-number/
		- GeeksforGeeks: https://www.geeksforgeeks.org/problems/nth-fibonacci-number1335/1

	??? success "Answer"
		```python
		# O(n) time | O(n) space - Recursive with memoization
		def getNthFib(n, memo={1: 0, 2: 1}):
		    if n in memo:
		        return memo[n]
		    else:
		        memo[n] = getNthFib(n - 1, memo) + getNthFib(n - 2, memo)
		        return memo[n]
		```

		```python
		# O(n) time | O(1) space - Iterative
		def getNthFib(n):
			lastTwo = [0, 1]
			counter = 3
			while counter <= n:
				nextFib = lastTwo[0] + lastTwo[1]
				lastTwo[0] = lastTwo[1]
				lastTwo[1] = nextFib
				counter += 1
			return lastTwo[1] if n > 1 else lastTwo[0]
		```

		```python
		# O(2^n) time | O(n) space - Simple recursive (very inefficient)
		def getNthFib(n):
		    if n == 2:
		        return 1
		    elif n == 1:
		        return 0
		    else:
		        return getNthFib(n - 1) + getNthFib(n - 2)
		```

---

### Product Sum

!!! question

	Write a function that takes in a "special" array and returns its product sum. A "special" array
	is a non-empty array that contains either integers or other "special" arrays. The product sum of a "special" array is the sum of its
	elements, where "special" arrays inside it are summed themselves and then multiplied by their level of depth.

	For example, the product sum of `[x, y]` is `x + y`; the product sum of `[x, [y, z]]` is `x + 2y + 2z`.

	Eg:
	Input: `[5, 2, [7, -1], 3, [6, [-13, 8], 4]]`
	Output: `12 # calculated as: 5 + 2 + 2 * (7 - 1) + 3 + 2 * (6 + 3 * (-13 + 8) + 4)`

	??? success "Answer"

		```python
		# O(n) time | O(d) space - where n is the total number of elements in the array,
		# including sub-elements, and d is the greatest depth of "special" arrays in the array
		def productSum(array, depth = 1):
			sum = 0
			for element in array:
				if type(element) is list:
					sum += productSum(element, depth + 1)
				else:
					sum += element
			return sum * depth
		```

---

### Find Three Largest Numbers

!!! question

    Write a function that takes in an array of at least three integers and, *without sorting the input array*, returns a *sorted* array of the three largest integers in the input array.  The function should return duplicate integers if necessary; for example, it should return `[10, 10, 12]` for an input array of `[10, 5, 9, 10, 12]`.

    ??? success "Answer"
        ```python
        # O(n) time | O(1) space
        def findThreeLargestNumbers(array):
            threeLargest = [None, None, None]
            for num in array:
                updateLargest(threeLargest, num)
            return threeLargest

        def updateLargest(threeLargest, num):
            if threeLargest[2] is None or num > threeLargest[2]:
                shiftAndUpdate(threeLargest, num, 2)
            elif threeLargest[1] is None or num > threeLargest[1]:
                shiftAndUpdate(threeLargest, num, 1)
            elif threeLargest[0] is None or num > threeLargest[0]:
                shiftAndUpdate(threeLargest, num, 0)

        def shiftAndUpdate(array, num, idx):
            for i in range(idx + 1):
                if i == idx:
                    array[i] = num
                else:
                    array[i] = array[i + 1]
        ```

---

### Palindrome Check

!!! question

    Write a function that takes in a non-empty string and that returns a boolean representing whether the string is a palindrome. A string is defined as a palindrome if it's written the same forward and backward.  Single-character strings are palindromes.

    ??? example "Try it!"
        - LeetCode: https://leetcode.com/problems/valid-palindrome/ (Similar, but allows ignoring non-alphanumeric characters)

    ??? success "Answer"
        ```python
        # O(n) time | O(1) space
        def isPalindrome(string):
            left = 0
            right = len(string) - 1
            while left < right:
                if string[left] != string[right]:
                    return False
                left += 1
                right -= 1
            return True
        ```

        ```python
        # O(n) time | O(n) space - Recursive
        def isPalindrome(string, i=0):
            j = len(string) - 1 - i
            return True if i >= j else string[i] == string[j] and isPalindrome(string, i + 1)
        ```

        ```python
        # O(n) time | O(n) space - Pythonic (creates a reversed string)
        def isPalindrome(string):
            return string == string[::-1]
        ```

---

### Caesar Cipher Encryptor

!!! question

    Given a non-empty string of lowercase letters and a non-negative integer representing a key, write a function that returns a new string obtained by shifting every letter in the input string by k positions in the alphabet, where k is the key.  Letters should "wrap" around the alphabet; in other words, the letter `z` shifted by one returns the letter `a`.

    ??? success "Answer"
        ```python
        # O(n) time | O(n) space
        def caesarCipherEncryptor(string, key):
            newLetters = []
            newKey = key % 26
            for letter in string:
                newLetters.append(getNewLetter(letter, newKey))
            return "".join(newLetters)

        def getNewLetter(letter, key):
            newLetterCode = ord(letter) + key
            return chr(newLetterCode) if newLetterCode <= 122 else chr(96 + newLetterCode % 122)
        ```
        ```python
        # O(n) time | O(n) space - using an alphabet string
        def caesarCipherEncryptor(string, key):
            newLetters = []
            newKey = key % 26
            alphabet = list("abcdefghijklmnopqrstuvwxyz")
            for letter in string:
                newLetters.append(getNewLetter(letter, newKey, alphabet))
            return "".join(newLetters)


        def getNewLetter(letter, key, alphabet):
            newLetterCode = alphabet.index(letter) + key
            return alphabet[newLetterCode % 26]
        ```

---
## 🙂 Medium

---

### Smallest Difference

!!! question

    Write a function that takes in two non-empty arrays of integers, finds the
    pair of numbers (one from each array) whose absolute difference is closest to zero,
    and returns an array containing these two numbers, with the number from the
    first array in the first position.

    You can assume that there will only be one pair of numbers with the smallest
    difference.

    ??? success "Answer"
        ```python
        # O(nlog(n) + mlog(m)) time | O(1) space - where n is the length of the first array
        # and m is the length of the second array
        def smallestDifference(arrayOne, arrayTwo):
            arrayOne.sort()
            arrayTwo.sort()
            idxOne = 0
            idxTwo = 0
            smallest = float("inf")
            current = float("inf")
            smallestPair = []
            while idxOne < len(arrayOne) and idxTwo < len(arrayTwo):
                firstNum = arrayOne[idxOne]
                secondNum = arrayTwo[idxTwo]
                if firstNum < secondNum:
                    current = secondNum - firstNum
                    idxOne += 1
                elif secondNum < firstNum:
                    current = firstNum - secondNum
                    idxTwo += 1
                else:
                    return [firstNum, secondNum]
                if smallest > current:
                    smallest = current
                    smallestPair = [firstNum, secondNum]
            return smallestPair
        ```

---

### Monotonic Array

!!! question

    Write a function that takes in an array of integers and returns a boolean
    representing whether the array is monotonic.

    An array is said to be monotonic if its elements, from left to right, are
    entirely non-increasing or entirely non-decreasing.

    Non-increasing elements aren't necessarily exclusively decreasing; they simply
    don't increase.  Similarly, non-decreasing elements aren't necessarily
    exclusively increasing; they simply don't decrease.

    Note that empty arrays and arrays of one element are monotonic.

    ??? success "Answer"
        ```python
        # O(n) time | O(1) space - where n is the length of the array
        def isMonotonic(array):
            isNonDecreasing = True
            isNonIncreasing = True
            for i in range(1, len(array)):
                if array[i] < array[i - 1]:
                    isNonDecreasing = False
                if array[i] > array[i - 1]:
                    isNonIncreasing = False
            return isNonDecreasing or isNonIncreasing
        ```

        ```python
        # Slightly more verbose, but potentially easier to understand version
        def isMonotonic(array):
            if len(array) <= 2:
                return True

            direction = array[1] - array[0]
            for i in range(2, len(array)):
                if direction == 0:
                    direction = array[i] - array[i - 1]
                    continue
                if breaksDirection(direction, array[i - 1], array[i]):
                    return False

            return True

        def breaksDirection(direction, previousInt, currentInt):
            difference = currentInt - previousInt
            if direction > 0:
                return difference < 0
            return difference > 0
        ```

---

### Move Element To End

!!! question

    You're given an array of integers and an integer. Write a function that moves
    all instances of that integer in the array to the end of the array and returns
    the array.

    The function should perform this in place (i.e., it should mutate the input
    array) and doesn't need to maintain the order of the other integers.

    ??? success "Answer"
        ```python
        # O(n) time | O(1) space - where n is the length of the array
        def moveElementToEnd(array, toMove):
            i = 0
            j = len(array) - 1
            while i < j:
                while i < j and array[j] == toMove:
                    j -= 1
                if array[i] == toMove:
                    array[i], array[j] = array[j], array[i]
                i += 1
            return array
        ```
---

### Spiral Traverse

!!! question

    Write a function that takes in an n x m two-dimensional array (that can be
    square-shaped when n == m) and returns a one-dimensional array of all the
    array's elements in spiral order.

    Spiral order starts at the top left corner of the two-dimensional array, goes
    to the right, and proceeds in a spiral pattern all the way until every element
    has been visited.

    ??? success "Answer"
        ```python
        # O(n) time | O(n) space - where n is the total number of elements in the array
        def spiralTraverse(array):
            result = []
            startRow, endRow = 0, len(array) - 1
            startCol, endCol = 0, len(array[0]) - 1

            while startRow <= endRow and startCol <= endCol:
                # Traverse top border
                for col in range(startCol, endCol + 1):
                    result.append(array[startRow][col])

                # Traverse right border
                for row in range(startRow + 1, endRow + 1):
                    result.append(array[row][endCol])

                # Traverse bottom border
                for col in reversed(range(startCol, endCol)):
                    # Handle the edge case when there's a single row
                    # in the middle of the matrix. In this case, we don't
                    # want to double-count the values in this row,
                    # which we've already counted in the first for loop
                    # above. See test case 8 for an example of this.
                    if startRow == endRow:
                        break
                    result.append(array[endRow][col])

                # Traverse left border
                for row in reversed(range(startRow + 1, endRow)):
                    # Handle the edge case when there's a single column
                    # in the middle of the matrix. In this case, we don't
                    # want to double-count the values in this column,
                    # which we've already counted in the second for loop
                    # above. See test case 9 for an example of this.
                    if startCol == endCol:
                        break
                    result.append(array[row][startCol])

                startRow += 1
                endRow -= 1
                startCol += 1
                endCol -= 1

            return result
        ```

        ```python
        # Recursive solution - O(n) time | O(n) space
        def spiralTraverse(array):
            result = []
            spiralFill(array, 0, len(array) - 1, 0, len(array[0]) - 1, result)
            return result

        def spiralFill(array, startRow, endRow, startCol, endCol, result):
            if startRow > endRow or startCol > endCol:
                return

            for col in range(startCol, endCol + 1):
                result.append(array[startRow][col])

            for row in range(startRow + 1, endRow + 1):
                result.append(array[row][endCol])

            for col in reversed(range(startCol, endCol)):
                if startRow == endRow:
                    break
                result.append(array[endRow][col])

            for row in reversed(range(startRow + 1, endRow)):
                if startCol == endCol:
                    break
                result.append(array[row][startCol])

            spiralFill(array, startRow + 1, endRow - 1, startCol + 1, endCol - 1, result)
        ```

---

### Longest Peak

!!! question

    Write a function that takes in an array of integers and returns the length of
    the longest peak in the array.

    A peak is defined as adjacent integers in the array that are *strictly*
    increasing until they reach a tip (the highest value in the peak), at which
    point they become *strictly* decreasing. At least three integers are required to
    form a peak.

    For example, the integers `1, 4, 10, 2` form a peak, but the integers `4, 0, 10`
    don't and neither do the integers `1, 2, 2, 0`. Similarly, the integers `1, 2, 3`
    don't form a peak because there aren't any strictly decreasing integers after
    the `3`.

    ??? success "Answer"
        ```python
        # O(n) time | O(1) space - where n is the length of the input array
        def longestPeak(array):
            longestPeakLength = 0
            i = 1
            while i < len(array) - 1:
                isPeak = array[i - 1] < array[i] and array[i] > array[i + 1]
                if not isPeak:
                    i += 1
                    continue

                leftIdx = i - 2
                while leftIdx >= 0 and array[leftIdx] < array[leftIdx + 1]:
                    leftIdx -= 1
                rightIdx = i + 2
                while rightIdx < len(array) and array[rightIdx] < array[rightIdx - 1]:
                    rightIdx += 1

                currentPeakLength = rightIdx - leftIdx - 1
                longestPeakLength = max(longestPeakLength, currentPeakLength)
                i = rightIdx
            return longestPeakLength
        ```

---

###  Array of Products

!!! question
    Write a function that takes in a non-empty array of integers and returns an
    array of the same length, where each element in the output array is equal to
    the product of every other number in the input array.

    In other words, the value at `output[i]` is equal to the product of every number
    in the input array other than `input[i]`.

    Note that you're expected to solve this problem without using division.

    ??? success "Answer"
        ```python
        # O(n) time | O(n) space - where n is the length of the input array
        def arrayOfProducts(array):
            products = [1 for _ in range(len(array))]

            leftRunningProduct = 1
            for i in range(len(array)):
                products[i] = leftRunningProduct
                leftRunningProduct *= array[i]

            rightRunningProduct = 1
            for i in reversed(range(len(array))):
                products[i] *= rightRunningProduct
                rightRunningProduct *= array[i]

            return products
        ```

        ```python
        # Another solution, more verbose, but easier to understand for some.
        # O(n) time | O(n) space - where n is the length of the input array
        def arrayOfProducts(array):
            leftProducts = [1 for _ in range(len(array))]
            rightProducts = [1 for _ in range(len(array))]
            products = [1 for _ in range(len(array))]

            leftRunningProduct = 1
            for i in range(len(array)):
                leftProducts[i] = leftRunningProduct
                leftRunningProduct *= array[i]

            rightRunningProduct = 1
            for i in reversed(range(len(array))):
                rightProducts[i] = rightRunningProduct
                rightRunningProduct *= array[i]

            for i in range(len(array)):
                products[i] = leftProducts[i] * rightProducts[i]

            return products
        ```

        ```python
        # Brute force - O(n^2) time | O(n) space
        def arrayOfProducts(array):
            products = [1 for _ in range(len(array))]

            for i in range(len(array)):
                runningProduct = 1
                for j in range(len(array)):
                    if i != j:
                        runningProduct *= array[j]
                products[i] = runningProduct
            return products
        ```

---

### First Duplicate Value

!!! question

    Given an array of integers between `1` and `n`, inclusive, where `n` is the
    length of the array, write a function that returns the first integer that appears
    more than once (when the array is read from left to right).

    In other words, out of all the integers that might occur more than once in the
    input array, your function should return the one whose first duplicate value
    has the minimum index.

    If no integer appears more than once, your function should return `-1`.

    Note that you're allowed to mutate the input array.

    ??? success "Answer"
        ```python
        # O(n) time | O(n) space - where n is the length of the input array
        def firstDuplicateValue(array):
            seen = set()
            for value in array:
                if value in seen:
                    return value
                seen.add(value)
            return -1
        ```

        ```python
        # O(n) time | O(1) space - where n is the length of the input array
        # This solution utilizes the fact that all values are between 1 and n (inclusive)
        def firstDuplicateValue(array):
            for value in array:
                absValue = abs(value)
                if array[absValue - 1] < 0:
                    return absValue
                array[absValue - 1] *= -1
            return -1
        ```

        ```python
        # Brute Force: O(n^2) time | O(1) space
        def firstDuplicateValue(array):
            minimumSecondIndex = len(array)
            for i in range(len(array)):
                value = array[i]
                for j in range(i + 1, len(array)):
                    valueToCompare = array[j]
                    if value == valueToCompare:
                        minimumSecondIndex = min(minimumSecondIndex, j)

            if minimumSecondIndex == len(array):
                return -1

            return array[minimumSecondIndex]
        ```

---

### Merge Overlapping Intervals

!!! question

    Write a function that takes in a non-empty array of arbitrary intervals,
    merges any overlapping intervals, and returns the new intervals in no
    particular order.

    Each interval `interval` is an array of two integers, with
    `interval[0]` as the start of the interval and `interval[1]` as the end
    of the interval.

    Note that back-to-back intervals aren't considered to be overlapping. For
    example, `[1, 5]` and `[6, 7]` aren't overlapping; however,
    `[1, 6]` and `[6, 7]` are indeed overlapping.

    Also note that the input intervals might be unsorted.

    ??? success "Answer"
        ```python
        # O(nlog(n)) time | O(n) space - where n is the length of the input array
        def mergeOverlappingIntervals(intervals):
            # Sort the intervals by their starting values
            sortedIntervals = sorted(intervals, key=lambda x: x[0])

            mergedIntervals = []
            currentInterval = sortedIntervals[0]
            mergedIntervals.append(currentInterval)

            for nextInterval in sortedIntervals:
                _, currentIntervalEnd = currentInterval
                nextIntervalStart, nextIntervalEnd = nextInterval

                if currentIntervalEnd >= nextIntervalStart:
                    currentInterval[1] = max(currentIntervalEnd, nextIntervalEnd)
                else:
                    currentInterval = nextInterval
                    mergedIntervals.append(currentInterval)

            return mergedIntervals
        ```

---

## 🤨 Hard

---

### Four Number Sum

!!! question

    Write a function that takes in a non-empty array of distinct integers and an
    integer representing a target sum. The function should find all quadruplets in
    the array that sum up to the target sum and return a two-dimensional array of
    all these quadruplets in no particular order.

    If no four numbers sum up to the target sum, the function should return an
    empty array.

    ??? success "Answer"
        ```python
        # Average: O(n^2) time | O(n^2) space
        # Worst: O(n^3) time | O(n^2) space
        def fourNumberSum(array, targetSum):
            allPairSums = {}
            quadruplets = []
            for i in range(1, len(array) - 1):
                for j in range(i + 1, len(array)):
                    currentSum = array[i] + array[j]
                    difference = targetSum - currentSum
                    if difference in allPairSums:
                        for pair in allPairSums[difference]:
                            quadruplets.append(pair + [array[i], array[j]])
                for k in range(0, i):
                    currentSum = array[i] + array[k]
                    if currentSum not in allPairSums:
                        allPairSums[currentSum] = [[array[k], array[i]]]
                    else:
                        allPairSums[currentSum].append([array[k], array[i]])
            return quadruplets
        ```

        ```python
        # O(n^3) time | O(n) space - Iterative
        def fourNumberSum(array, targetSum):
            array.sort()
            quadruplets = []
            for i in range(len(array) - 3):
                for j in range(i + 1, len(array) - 2):
                    left = j + 1
                    right = len(array) - 1
                    while left < right:
                        currentSum = array[i] + array[j] + array[left] + array[right]
                        if currentSum == targetSum:
                            quadruplets.append([array[i], array[j], array[left], array[right]])
                            left += 1
                            right -= 1
                        elif currentSum < targetSum:
                            left += 1
                        else:
                            right -= 1
            return quadruplets
        ```

---

### Subarray Sort

!!! question

    Write a function that takes in an array of at least two integers and that
    returns an array of the starting and ending indices of the smallest subarray
    in the input array that needs to be sorted in place in order for the entire
    input array to be sorted (in ascending order).

    If the input array is already sorted, the function should return `[-1, -1]`.

    ??? success "Answer"
        ```python
        # O(n) time | O(1) space - where n is the length of the input array
        def subarraySort(array):
            minOutOfOrder = float("inf")
            maxOutOfOrder = float("-inf")
            for i in range(len(array)):
                num = array[i]
                if isOutOfOrder(i, num, array):
                    minOutOfOrder = min(minOutOfOrder, num)
                    maxOutOfOrder = max(maxOutOfOrder, num)
            if minOutOfOrder == float("inf"):
                return [-1, -1]
            subarrayLeftIdx = 0
            while minOutOfOrder >= array[subarrayLeftIdx]:
                subarrayLeftIdx += 1
            subarrayRightIdx = len(array) - 1
            while maxOutOfOrder <= array[subarrayRightIdx]:
                subarrayRightIdx -= 1
            return [subarrayLeftIdx, subarrayRightIdx]

        def isOutOfOrder(i, num, array):
            if i == 0:
                return num > array[i + 1]
            if i == len(array) - 1:
                return num < array[i - 1]
            return num > array[i + 1] or num < array[i - 1]
        ```

---

### Largest Range

!!! question

    Write a function that takes in an array of integers and returns an array of
    length 2 representing the largest range of integers contained in that array.

    The first number in the output array should be the first number in the range,
    while the second number should be the last number in the range.

    A range of numbers is defined as a set of numbers that come right after each
    other in the set of real integers. For instance, the output array `[2, 6]`
    represents the range `{2, 3, 4, 5, 6}`, which is a range of length 5. Note that
    numbers don't need to be sorted or adjacent in the input array in order to form
    a range.

    You can assume that there will only be one largest range.

    ??? success "Answer"
        ```python
        # O(n) time | O(n) space - where n is the length of the input array
        def largestRange(array):
            bestRange = []
            longestLength = 0
            nums = {}
            for num in array:
                nums[num] = True
            for num in array:
                if not nums[num]:
                    continue
                nums[num] = False
                currentLength = 1
                left = num - 1
                right = num + 1
                while left in nums:
                    nums[left] = False
                    currentLength += 1
                    left -= 1
                while right in nums:
                    nums[right] = False
                    currentLength += 1
                    right += 1
                if currentLength > longestLength:
                    longestLength = currentLength
                    bestRange = [left + 1, right - 1]
            return bestRange
        ```

---

### Min Rewards

!!! question
    Imagine that you're a teacher who's just graded the final exam in a class.
    You have a list of student scores on the final exam in a particular order
    (not necessarily sorted), and you want to reward your students. You decide to
    do so fairly by giving them arbitrary rewards following two rules:

    1.  All students must receive at least one reward.
    2.  Any given student must receive strictly more rewards than an adjacent
        student (a student immediately to the left or to the right) with a lower
        score and must receive strictly fewer rewards than an adjacent student with
        a higher score.

    Write a function that takes in a list of scores and returns the minimum number
    of rewards that you must give out to students to satisfy the two rules.

    You can assume that all students have different scores; in other words, the
    scores are all unique.

    ??? success "Answer"
        ```python
        # O(n) time | O(n) space - where n is the length of the input array
        def minRewards(scores):
            rewards = [1 for _ in scores]
            for i in range(1, len(scores)):
                if scores[i] > scores[i - 1]:
                    rewards[i] = rewards[i - 1] + 1
            for i in reversed(range(len(scores) - 1)):
                if scores[i] > scores[i + 1]:
                    rewards[i] = max(rewards[i], rewards[i + 1] + 1)
            return sum(rewards)
        ```

        ```python
        # O(n) time | O(n) space - where n is the length of the input array
        # This solution finds local mins and expands from them.
        def minRewards(scores):
            rewards = [1 for _ in scores]
            localMinIdxs = getLocalMinIdxs(scores)
            for localMinIdx in localMinIdxs:
                expandFromLocalMinIdx(localMinIdx, scores, rewards)
            return sum(rewards)


        def getLocalMinIdxs(array):
            if len(array) == 1:
                return [0]
            localMinIdxs = []
            for i in range(len(array)):
                if i == 0 and array[i] < array[i + 1]:
                    localMinIdxs.append(i)
                if i == len(array) - 1 and array[i] < array[i - 1]:
                    localMinIdxs.append(i)
                if i == 0 or i == len(array) - 1:
                    continue
                if array[i] < array[i + 1] and array[i] < array[i - 1]:
                    localMinIdxs.append(i)
            return localMinIdxs


        def expandFromLocalMinIdx(localMinIdx, scores, rewards):
            leftIdx = localMinIdx - 1
            while leftIdx >= 0 and scores[leftIdx] > scores[leftIdx + 1]:
                rewards[leftIdx] = max(rewards[leftIdx], rewards[leftIdx + 1] + 1)
                leftIdx -= 1
            rightIdx = localMinIdx + 1
            while rightIdx < len(scores) and scores[rightIdx] > scores[rightIdx - 1]:
                rewards[rightIdx] = rewards[rightIdx - 1] + 1
                rightIdx += 1
        ```

        ```python
        # Brute Force, O(n^2) time | O(n) space
        def minRewards(scores):
            rewards = [1 for _ in scores]
            for i in range(1, len(scores)):
                j = i - 1
                if scores[i] > scores[j]:
                    rewards[i] = rewards[j] + 1
                else:
                    while j >= 0 and scores[j] > scores[j + 1]:
                        rewards[j] = max(rewards[j], rewards[j + 1] + 1)
                        j -= 1
            return sum(rewards)
        ```

---

### Zigzag Traverse

!!! question
    Write a function that takes in an n x m two-dimensional array (that can be
    square-shaped when n == m) and returns a one-dimensional array of all the
    array's elements in zigzag order.

    Zigzag order starts at the top left corner of the two-dimensional array, goes
    down by one element, and proceeds in a zigzag pattern all the way to the
    bottom right corner.

    ??? success "Answer"
        ```python
        # O(n) time | O(n) space - where n is the total number of elements in the array
        def zigzagTraverse(array):
            height = len(array) - 1
            width = len(array[0]) - 1
            result = []
            row, col = 0, 0
            goingDown = True
            while not isOutOfBounds(row, col, height, width):
                result.append(array[row][col])
                if goingDown:
                    if col == 0 or row == height:
                        goingDown = False
                        if row == height:
                            col += 1
                        else:
                            row += 1
                    else:
                        row += 1
                        col -= 1
                else:
                    if row == 0 or col == width:
                        goingDown = True
                        if col == width:
                            row += 1
                        else:
                            col += 1
                    else:
                        row -= 1
                        col += 1
            return result

        def isOutOfBounds(row, col, height, width):
            return row < 0 or row > height or col < 0 or col > width
        ```

---

## 😲 Very Hard

---

### Max Sum Submatrix

!!! question

    You're given a two-dimensional array (a matrix) of potentially unequal height
    and width that's filled with integers. You're also given a positive integer
    `size`. Write a function that returns the maximum sum that can be generated
    from a submatrix with dimensions `size * size`.

    For example, consider the following matrix:

    ```
    [
      [2, 4],
      [5, 6],
      [-3, 2],
    ]
    ```

    If `size = 2`, then the 2x2 submatrices to consider are:

    ```
    [2, 4]
    [5, 6]
    ```

    ```
    [5, 6]
    [-3, 2]
    ```

    The sum of the elements in the first submatrix is 17, and the sum of the
    elements in the second submatrix is 10. Thus, your function should return 17.

    **Note:** `size` will always be greater than or equal to 1, and the
    dimensions of the input matrix will always be at least `size * size`.

    ??? success "Answer"
        ```python
        # O(w * h) time | O(w * h) space - where w and h
        # are the width and height of the input matrix
        def maximumSumSubmatrix(matrix, size):
            sums = createSumMatrix(matrix)
            maxSubMatrixSum = float("-inf")

            for row in range(size - 1, len(matrix)):
                for col in range(size - 1, len(matrix[row])):
                    total = sums[row][col]

                    touchesTopBorder = row - size < 0
                    if not touchesTopBorder:
                        total -= sums[row - size][col]

                    touchesLeftBorder = col - size < 0
                    if not touchesLeftBorder:
                        total -= sums[row][col - size]

                    touchesTopOrLeftBorder = touchesTopBorder or touchesLeftBorder
                    if not touchesTopOrLeftBorder:
                        total += sums[row - size][col - size]

                    maxSubMatrixSum = max(maxSubMatrixSum, total)

            return maxSubMatrixSum

        def createSumMatrix(matrix):
            sums = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
            sums[0][0] = matrix[0][0]
            # Fill the first row
            for idx in range(1, len(matrix[0])):
                sums[0][idx] = sums[0][idx - 1] + matrix[0][idx]
            # Fill the first column
            for idx in range(1, len(matrix)):
                sums[idx][0] = sums[idx - 1][0] + matrix[idx][0]
            # Fill the rest of the matrix
            for row in range(1, len(matrix)):
                for col in range(1, len(matrix[row])):
                    sums[row][col] = (
                        sums[row - 1][col]
                        + sums[row][col - 1]
                        - sums[row - 1][col - 1]
                        + matrix[row][col]
                    )
            return sums
        ```

        ```python
        # Brute Force: O(w * h * size^2) time | O(1) space
        def maximumSumSubmatrix(matrix, size):
            max_sum = float('-inf')
            for row in range(len(matrix) - size + 1):
                for col in range(len(matrix[0]) - size + 1):
                    current_sum = 0
                    for i in range(size):
                        for j in range(size):
                            current_sum += matrix[row + i][col + j]
                    max_sum = max(max_sum, current_sum)
            return max_sum
        ```

---

### Maximize Expression

!!! question

    Write a function that takes in an array of integers and returns the largest
    possible value for the expression `array[a] - array[b] + array[c] - array[d]`,
    where `a`, `b`, `c`, and `d` are indices of the array and `a < b < c < d`.

    If the input array has fewer than 4 elements, your function should return 0.

    ??? success "Answer"
        ```python
        # O(n) time | O(n) space - where n is the length of the array
        def maximizeExpression(array):
            if len(array) < 4:
                return 0

            maxOfA = [array[0]]
            maxOfAMinusB = [float("-inf")]
            maxOfAMinusBPlusC = [float("-inf")] * 2
            maxOfAMinusBPlusCMinusD = [float("-inf")] * 3

            for idx in range(1, len(array)):
                currentMax = max(maxOfA[idx - 1], array[idx])
                maxOfA.append(currentMax)

            for idx in range(1, len(array)):
                currentMax = max(maxOfAMinusB[idx - 1], maxOfA[idx - 1] - array[idx])
                maxOfAMinusB.append(currentMax)

            for idx in range(2, len(array)):
                currentMax = max(maxOfAMinusBPlusC[idx - 1], maxOfAMinusB[idx - 1] + array[idx])
                maxOfAMinusBPlusC.append(currentMax)

            for idx in range(3, len(array)):
                currentMax = max(maxOfAMinusBPlusCMinusD[idx - 1], maxOfAMinusBPlusC[idx - 1] - array[idx])
                maxOfAMinusBPlusCMinusD.append(currentMax)

            return maxOfAMinusBPlusCMinusD[-1]
        ```

        ```python
        # Another O(n) time | O(n) space solution
        def maximizeExpression(array):
            if len(array) < 4:
                return 0

            maxOfA = [array[0]] * len(array)
            maxOfAMinusB = [float("-inf")] * len(array)
            maxOfAMinusBPlusC = [float("-inf")] * len(array)
            maxOfAMinusBPlusCMinusD = [float("-inf")] * len(array)

            for idx in range(1, len(array)):
                maxOfA[idx] = max(maxOfA[idx - 1], array[idx])

            for idx in range(1, len(array)):
                maxOfAMinusB[idx] = max(maxOfAMinusB[idx - 1], maxOfA[idx - 1] - array[idx])

            for idx in range(2, len(array)):
                maxOfAMinusBPlusC[idx] = max(maxOfAMinusBPlusC[idx - 1], maxOfAMinusB[idx - 1] + array[idx])

            for idx in range(3, len(array)):
                maxOfAMinusBPlusCMinusD[idx] = max(
                    maxOfAMinusBPlusCMinusD[idx - 1], maxOfAMinusBPlusC[idx - 1] - array[idx]
                )

            return maxOfAMinusBPlusCMinusD[len(array) - 1]
        ```

        ```python
        # Brute Force: O(n^4) time | O(1) space
        def maximizeExpression(array):
            if len(array) < 4:
                return 0

            maximumValueFound = float("-inf")

            for a in range(len(array)):
                aValue = array[a]
                for b in range(a + 1, len(array)):
                    bValue = array[b]
                    for c in range(b + 1, len(array)):
                        cValue = array[c]
                        for d in range(c + 1, len(array)):
                            dValue = array[d]
                            expressionValue = aValue - bValue + cValue - dValue
                            maximumValueFound = max(expressionValue, maximumValueFound)

            return maximumValueFound
        ```

---

### Max Subset Sum With No Adjacent Elements

!!! question

    Write a function that takes in an array of positive integers and returns the
    maximum sum of non-adjacent elements in the array.

    If the input array is empty, the function should return `0`.

    ??? success "Answer"

        ```python
        # O(n) time | O(1) space - where n is the length of the input array
        def maxSubsetSumNoAdjacent(array):
            if not len(array):
                return 0
            elif len(array) == 1:
                return array[0]
            second = array[0]
            first = max(array[0], array[1])
            for i in range(2, len(array)):
                current = max(first, second + array[i])
                second = first
                first = current
            return first
        ```

        ```python
        # O(n) time | O(n) space
        def maxSubsetSumNoAdjacent(array):
            if not len(array):
                return 0
            elif len(array) == 1:
                return array[0]
            maxSums = array[:]
            maxSums[1] = max(array[0], array[1])
            for i in range(2, len(array)):
                maxSums[i] = max(maxSums[i - 1], maxSums[i - 2] + array[i])
            return maxSums[-1]
        ```

---

### Number Of Ways To Make Change

!!! question

    Given an array of distinct positive integers representing coin denominations and a
    single non-negative integer `n` representing a target amount of
    money, write a function that returns the number of ways to make change for
    that target amount using the given coin denominations.

    Note that an unlimited amount of coins is at your disposal.

    ??? success "Answer"
        ```python
        # O(nd) time | O(n) space - where n is the target amount and d is the number of coin denominations
        def numberOfWaysToMakeChange(n, denoms):
            ways = [0 for amount in range(n + 1)]
            ways[0] = 1
            for denom in denoms:
                for amount in range(1, n + 1):
                    if denom <= amount:
                        ways[amount] += ways[amount - denom]
            return ways[n]
        ```

---

### Number of Ways to Traverse Graph

!!! question

    You're given two positive integers representing the width and height of a grid-shaped, rectangular graph. Write a function that returns the number of ways to
    reach the bottom right corner of the graph when starting at the top left corner. Each move you take must either go down or right. In other words, you can
    never move up or left in the graph.

    For example, given the graph illustrated below, with `width = 2` and `height = 3`, there are three ways to reach the bottom right corner when starting at the top left corner:

    ```
     _ _
    |_|_|
    |_|_|
    |_|_|
    ```

    1.  Down, Down, Right
    2.  Right, Down, Down
    3.  Down, Right, Down

    Note: you may assume that `width * height >= 2`. In other words, the graph will
    never be a 1x1 grid.

    ??? success "Answer"
        ```python
        # O(n + m) time | O(1) space - where n is the width and m is the height
        # This solution uses the formula for combinations: nCr = n! / (r! * (n - r)!)
        def numberOfWaysToTraverseGraph(width, height):
            xDistanceToCorner = width - 1
            yDistanceToCorner = height - 1

            # The number of permutations of right and down movements
            # is the number of ways to reach the bottom right corner.
            numerator = factorial(xDistanceToCorner + yDistanceToCorner)
            denominator = factorial(xDistanceToCorner) * factorial(yDistanceToCorner)
            return numerator // denominator

        def factorial(num):
            result = 1
            for n in range(2, num + 1):
                result *= n
            return result
        ```

        ```python
        # O(n * m) time | O(n * m) space
        def numberOfWaysToTraverseGraph(width, height):
            numberOfWays = [[0 for _ in range(width)] for _ in range(height)]

            for widthIdx in range(width):
                for heightIdx in range(height):
                    if widthIdx == 0 or heightIdx == 0:
                        numberOfWays[heightIdx][widthIdx] = 1
                    else:
                        waysLeft = numberOfWays[heightIdx][widthIdx - 1]
                        waysUp = numberOfWays[heightIdx - 1][widthIdx]
                        numberOfWays[heightIdx][widthIdx] = waysLeft + waysUp

            return numberOfWays[height - 1][width - 1]
        ```

        ```python
        # Recursive solution - O(2^(n + m)) time | O(n + m) space
        def numberOfWaysToTraverseGraph(width, height):
            if width == 1 or height == 1:
                return 1
            return numberOfWaysToTraverseGraph(width - 1, height) + numberOfWaysToTraverseGraph(width, height - 1)
        ```

---

### Levenshtein Distance

!!! question

    Write a function that takes in two strings and returns the minimum number of
    edit operations that need to be performed on the first string to obtain the
    second string.

    There are three edit operations: insertion of a character, deletion of a
    character, and substitution of a character for another.

    ??? success "Answer"
        ```python
        # O(nm) time | O(nm) space - where n and m are the lengths of the two input strings
        def levenshteinDistance(str1, str2):
            edits = [[x for x in range(len(str1) + 1)] for y in range(len(str2) + 1)]
            for i in range(1, len(str2) + 1):
                edits[i][0] = edits[i - 1][0] + 1
            for i in range(1, len(str2) + 1):
                for j in range(1, len(str1) + 1):
                    if str2[i - 1] == str1[j - 1]:
                        edits[i][j] = edits[i - 1][j - 1]
                    else:
                        edits[i][j] = 1 + min(edits[i - 1][j - 1], edits[i][j - 1], edits[i - 1][j])
            return edits[-1][-1]
        ```

        ```python
        # O(nm) time | O(min(n, m)) space - where n and m are the lengths of the two input strings
        def levenshteinDistance(str1, str2):
            small = str1 if len(str1) < len(str2) else str2
            big = str1 if len(str1) >= len(str2) else str2
            evenEdits = [x for x in range(len(small) + 1)]
            oddEdits = [None for x in range(len(small) + 1)]
            for i in range(1, len(big) + 1):
                if i % 2 == 1:
                    currentEdits = oddEdits
                    previousEdits = evenEdits
                else:
                    currentEdits = evenEdits
                    previousEdits = oddEdits
                currentEdits[0] = i
                for j in range(1, len(small) + 1):
                    if big[i - 1] == small[j - 1]:
                        currentEdits[j] = previousEdits[j - 1]
                    else:
                        currentEdits[j] = 1 + min(previousEdits[j - 1], previousEdits[j], currentEdits[j - 1])
            return evenEdits[-1] if len(big) % 2 == 0 else oddEdits[-1]
        ```

---

### Kadane's Algorithm

!!! question

    Write a function that takes in a non-empty array of integers and returns the
    maximum sum that can be obtained by summing up all of the integers in a
    non-empty subarray of the input array. A subarray must only contain adjacent
    numbers (numbers next to each other in the input array).

    ??? success "Answer"
        ```python
        # O(n) time | O(1) space - where n is the length of the input array
        def kadanesAlgorithm(array):
            maxEndingHere = array[0]
            maxSoFar = array[0]
            for num in array[1:]:
                maxEndingHere = max(num, maxEndingHere + num)
                maxSoFar = max(maxSoFar, maxEndingHere)
            return maxSoFar
        ```

---

### Single Cycle Check

!!! question

    You're given an array of integers where each integer represents a jump of its
    value in the array. For instance, the integer `2` represents a jump of two indices
    forward in the array; the integer `-3` represents a jump of three indices
    backward in the array.

    If a jump spills past the array's bounds, it wraps over to the other side. For
    instance, a jump of `-1` at index `0` brings us to the last index in the
    array. Similarly, a jump of `1` at the last index in the array brings us to
    index `0`.

    Write a function that returns a boolean representing whether the jumps in the
    array form a single cycle. A single cycle occurs if, starting at any index in
    the array and following the jumps, every element in the array is visited
    exactly once before landing back on the starting index.

    ??? success "Answer"
        ```python
        # O(n) time | O(1) space - where n is the length of the input array
        def hasSingleCycle(array):
            numElementsVisited = 0
            currentIdx = 0
            while numElementsVisited < len(array):
                if numElementsVisited > 0 and currentIdx == 0:
                    return False
                numElementsVisited += 1
                currentIdx = getNextIdx(currentIdx, array)
            return currentIdx == 0

        def getNextIdx(currentIdx, array):
            jump = array[currentIdx]
            nextIdx = (currentIdx + jump) % len(array)
            return nextIdx if nextIdx >= 0 else nextIdx + len(array)
        ```

---

### Breadth-first Search

!!! question

    You're given a `Node` class that has a `name` and an array of optional
    `children` nodes. When put together, nodes form an acyclic tree-like
    structure.

    Implement the `breadthFirstSearch` method on the `Node` class, which
    takes in an empty array, traverses the tree using the Breadth-first Search
    approach (specifically navigating the tree from left to right), stores all of
    the nodes' names in the input array, and returns it.

    ??? success "Answer"
        ```python
        # O(v + e) time | O(v) space - where v is the number of vertices
        # and e is the number of edges in the input graph
        class Node:
            def __init__(self, name):
                self.children = []
                self.name = name

            def addChild(self, name):
                self.children.append(Node(name))
                return self

            def breadthFirstSearch(self, array):
                queue = [self]
                while len(queue) > 0:
                    current = queue.pop(0)
                    array.append(current.name)
                    for child in current.children:
                        queue.append(child)
                return array
        ```

---

### River Sizes

!!! question

    You're given a two-dimensional array (a matrix) of potentially unequal height
    and width containing only `0`s and `1`s. Each `0` represents land, and each `1`
    represents part of a river. A river consists of any number of `1`s that are
    either horizontally or vertically adjacent (but not diagonally adjacent). The
    number of adjacent `1`s forming a river determine its size.

    Note that a river can twist. In other words, it doesn't have to be a straight
    vertical line or a straight horizontal line; it can be L-shaped, for example.

    Write a function that returns an array of the sizes of all rivers represented
    in the input matrix. The sizes don't need to be in any particular order.

    ??? success "Answer"
        ```python
        # O(wh) time | O(wh) space - where w and h are the width and height of the input matrix
        def riverSizes(matrix):
            sizes = []
            visited = [[False for value in row] for row in matrix]
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    if visited[i][j]:
                        continue
                    traverseNode(i, j, matrix, visited, sizes)
            return sizes

        def traverseNode(i, j, matrix, visited, sizes):
            currentRiverSize = 0
            nodesToExplore = [[i, j]]
            while len(nodesToExplore):
                currentNode = nodesToExplore.pop()
                i = currentNode[0]
                j = currentNode[1]
                if visited[i][j]:
                    continue
                visited[i][j] = True
                if matrix[i][j] == 0:
                    continue
                currentRiverSize += 1
                unvisitedNeighbors = getUnvisitedNeighbors(i, j, matrix, visited)
                for neighbor in unvisitedNeighbors:
                    nodesToExplore.append(neighbor)
            if currentRiverSize > 0:
                sizes.append(currentRiverSize)

        def getUnvisitedNeighbors(i, j, matrix, visited):
            unvisitedNeighbors = []
            if i > 0 and not visited[i - 1][j]:
                unvisitedNeighbors.append([i - 1, j])
            if i < len(matrix) - 1 and not visited[i + 1][j]:
                unvisitedNeighbors.append([i + 1, j])
            if j > 0 and not visited[i][j - 1]:
                unvisitedNeighbors.append([i, j - 1])
            if j < len(matrix[0]) - 1 and not visited[i][j + 1]:
                unvisitedNeighbors.append([i, j + 1])
            return unvisitedNeighbors
        ```

---

### Youngest Common Ancestor

!!! question
    You're given three inputs, all of which are instances of an `AncestralTree`
    class that have an `ancestor` property pointing to their youngest ancestor.
    The first input is the top ancestor in an ancestral tree (i.e., the only instance
    that has no ancestor--its `ancestor` property points to `None` / `null`), and
    the other two inputs are descendants in the ancestral tree.

    Write a function that returns the youngest common ancestor to the two
    descendants.

    Note that a descendant is considered its own ancestor. So in the simple
    ancestral tree below, the youngest common ancestor to nodes A and B is node A.

    ```
    // The youngest common ancestor to nodes A and B is node A.
    // The youngest common ancestor to nodes A and C is node A.
    // The youngest common ancestor to nodes A and I is node A.
    // The youngest common ancestor to nodes E and I is node B.
    // The youngest common ancestor to nodes H and G is node G.
    // The youngest common ancestor to nodes D and G is node A.
    // This tree is:
    //        A
    //     /    \
    //    B      C
    //  /   \  /  \
    // D    E F    G
    // |       \
    // H        I
    ```

    ??? success "Answer"
        ```python
        # O(d) time | O(1) space - where d is the depth of the deeper node
        def youngestCommonAncestor(topAncestor, descendantOne, descendantTwo):
            depthOne = getDepth(descendantOne, topAncestor)
            depthTwo = getDepth(descendantTwo, topAncestor)

            if depthOne > depthTwo:
                return backtrackAncestralTree(descendantOne, descendantTwo, depthOne - depthTwo)
            else:
                return backtrackAncestralTree(descendantTwo, descendantOne, depthTwo - depthOne)

        def getDepth(node, topAncestor):
            depth = 0
            while node != topAncestor:
                depth += 1
                node = node.ancestor
            return depth

        def backtrackAncestralTree(lowerDescendant, higherDescendant, diff):
            while diff > 0:
                lowerDescendant = lowerDescendant.ancestor
                diff -= 1
            while lowerDescendant != higherDescendant:
                lowerDescendant = lowerDescendant.ancestor
                higherDescendant = higherDescendant.ancestor
            return lowerDescendant
        ```

        ```python
        # O(d) time | O(1) space - where d is the depth of the deeper node
        def youngestCommonAncestor(topAncestor, descendantOne, descendantTwo):
            ancestorOne = descendantOne
            ancestorTwo = descendantTwo

            while ancestorOne != ancestorTwo:
                ancestorOne = topAncestor if ancestorOne == topAncestor else ancestorOne.ancestor
                ancestorTwo = topAncestor if ancestorTwo == topAncestor else ancestorTwo.ancestor

            return ancestorOne
        ```

---

### BST Construction

!!! question

    Write a `BinarySearchTree` class. The class should have a `value` property,
    and left and right node properties, each of which can either be a `BinarySearchTree`
    node or `None`. A `BinarySearchTree` node is said to be a valid BST node if and
    only if it satisfies the BST property: its `value` is strictly greater than the
    values of every node to its left; its `value` is less than or equal to the values
    of every node to its right; and both of its children nodes are either valid BST
    nodes themselves or `None`.

    The `BinarySearchTree` class should support:

    *   Inserting values: The `insert` method takes in an integer value and
        inserts it into the BST.
    *   Removing values: The `remove` method takes in an integer value and
        removes it from the BST. If the value isn't in the BST, the method
        should do nothing. If there are multiple instances of the value in the
        BST, the method should remove only the first instance of the value.
    *   Searching for values: The `contains` method takes in an integer value and
        returns a boolean representing whether or not the value is contained in
        the BST.

    ??? success "Answer"
        ```python
        # Average: O(log(n)) time | O(1) space
        # Worst: O(n) time | O(1) space
        class BST:
            def __init__(self, value):
                self.value = value
                self.left = None
                self.right = None

            def insert(self, value):
                currentNode = self
                while True:
                    if value < currentNode.value:
                        if currentNode.left is None:
                            currentNode.left = BST(value)
                            break
                        else:
                            currentNode = currentNode.left
                    else:
                        if currentNode.right is None:
                            currentNode.right = BST(value)
                            break
                        else:
                            currentNode = currentNode.right
                return self

            def contains(self, value):
                currentNode = self
                while currentNode is not None:
                    if value < currentNode.value:
                        currentNode = currentNode.left
                    elif value > currentNode.value:
                        currentNode = currentNode.right
                    else:
                        return True
                return False

            def remove(self, value, parentNode=None):
                currentNode = self
                while currentNode is not None:
                    if value < currentNode.value:
                        parentNode = currentNode
                        currentNode = currentNode.left
                    elif value > currentNode.value:
                        parentNode = currentNode
                        currentNode = currentNode.right
                    else:
                        # Case 1: Node to remove has two children
                        if currentNode.left is not None and currentNode.right is not None:
                            currentNode.value = currentNode.right.getMinValue()
                            currentNode.right.remove(currentNode.value, currentNode)
                        # Case 2: Node to remove is the root node
                        elif parentNode is None:
                            if currentNode.left is not None:
                                currentNode.value = currentNode.left.value
                                currentNode.left = currentNode.left.left
                                currentNode.right = currentNode.left.right
                            elif currentNode.right is not None:
                                currentNode.value = currentNode.right.value
                                currentNode.left = currentNode.right.left
                                currentNode.right = currentNode.right.right
                            else:
                                # This is a single node tree, do nothing
                                pass
                        # Case 3: Node to remove has one child
                        elif parentNode.left == currentNode:
                            parentNode.left = currentNode.left if currentNode.left is not None else currentNode.right
                        elif parentNode.right == currentNode:
                            parentNode.right = currentNode.left if currentNode.left is not None else currentNode.right
                        break
                return self

            def getMinValue(self):
                currentNode = self
                while currentNode.left is not None:
                    currentNode = currentNode.left
                return currentNode.value
        ```

---

### Invert Binary Tree

!!! question

    Write a function that takes in a Binary Tree, inverts it, and returns the
    inverted tree.

    In other words, the function should swap every left node in the tree for its
    corresponding right node.

    Each `BinaryTree` node has an integer `value`, a
    `left` child node, and a `right` child node. Children nodes can either be
    `BinaryTree` nodes themselves or `None`.

    ??? success "Answer"
        ```python
        # O(n) time | O(d) space - where n is the number of nodes in the tree
        # and d is the depth of the tree
        class BinaryTree:
            def __init__(self, value):
                self.value = value
                self.left = None
                self.right = None

        def invertBinaryTree(tree):
            if tree is None:
                return
            tree.left, tree.right = tree.right, tree.left
            invertBinaryTree(tree.left)
            invertBinaryTree(tree.right)
        ```

        ```python
        # Iterative solution - O(n) time | O(n) space
        def invertBinaryTree(tree):
            queue = [tree]
            while len(queue):
                current = queue.pop(0)
                if current is None:
                    continue
                current.left, current.right = current.right, current.left
                queue.append(current.left)
                queue.append(current.right)
        ```

---

### Validate BST

!!! question

    Write a function that takes in a potentially invalid Binary Search Tree (BST)
    and returns a boolean representing whether the BST is valid.

    Each `BST` node has an integer `value`, a
    `left` child node, and a `right` child node. A node is said to be a valid
    BST node if and only if it satisfies the BST property: its `value` is strictly
    greater than the values of every node to its left; its `value` is less than or
    equal to the values of every node to its right; and both of its children nodes
    are either valid BST nodes themselves or `None`.

    ??? success "Answer"
        ```python
        # O(n) time | O(d) space - where n is the number of nodes in the tree
        # and d is the depth of the tree
        class BST:
            def __init__(self, value):
                self.value = value
                self.left = None
                self.right = None

        def validateBst(tree):
            return validateBstHelper(tree, float("-inf"), float("inf"))

        def validateBstHelper(tree, minValue, maxValue):
            if tree is None:
                return True
            if tree.value < minValue or tree.value >= maxValue:
                return False
            leftIsValid = validateBstHelper(tree.left, minValue, tree.value)
            return leftIsValid and validateBstHelper(tree.right, tree.value, maxValue)
        ```

---

### Node Depths

!!! question

    The distance between a node in a Binary Tree and the tree's root is called the
    node's depth.

    Write a function that takes in a Binary Tree and returns the sum of its nodes'
    depths.

    Each `BinaryTree` node has an integer `value`, a
    `left` child node, and a `right` child node. Children nodes can either be
    `BinaryTree` nodes themselves or `None`.

    ??? success "Answer"
        ```python
        # O(n) time | O(d) space - where n is the number of nodes in the tree
        # and d is the depth of the tree
        class BinaryTree:
            def __init__(self, value):
                self.value = value
                self.left = None
                self.right = None

        def nodeDepths(root):
            return nodeDepthsHelper(root, 0)

        def nodeDepthsHelper(node, depth):
            if node is None:
                return 0
            return depth + nodeDepthsHelper(node.left, depth + 1) + nodeDepthsHelper(node.right, depth + 1)
        ```

        ```python
        # Iterative solution - O(n) time | O(d) space
        def nodeDepths(root):
            stack = [{"node": root, "depth": 0}]
            sumOfDepths = 0
            while len(stack) > 0:
                nodeInfo = stack.pop()
                node, depth = nodeInfo["node"], nodeInfo["depth"]
                if node is None:
                    continue
                sumOfDepths += depth
                stack.append({"node": node.left, "depth": depth + 1})
                stack.append({"node": node.right, "depth": depth + 1})
            return sumOfDepths
        ```

---

### Branch Sums

!!! question

    Write a function that takes in a Binary Tree and returns a list of its branch
    sums ordered from leftmost branch sum to rightmost branch sum.

    A branch sum is the sum of all values in a Binary Tree branch. A Binary Tree
    branch is a path of nodes in a tree that starts at the root node and ends at
    any leaf node.

    Each `BinaryTree` node has an integer `value`, a
    `left` child node, and a `right` child node. Children nodes can either be
    `BinaryTree` nodes themselves or `None`.

    ??? success "Answer"
        ```python
        # O(n) time | O(n) space - where n is the number of nodes in the tree
        class BinaryTree:
            def __init__(self, value):
                self.value = value
                self.left = None
                self.right = None

        def branchSums(root):
            sums = []
            calculateBranchSums(root, 0, sums)
            return sums

        def calculateBranchSums(node, runningSum, sums):
            if node is None:
                return
            runningSum += node.value
            if node.left is None and node.right is None:
                sums.append(runningSum)
                return
            calculateBranchSums(node.left, runningSum, sums)
            calculateBranchSums(node.right, runningSum, sums)
        ```

---

### Find Successor

!!! question

    Write a function that takes in a Binary Tree (where nodes have an additional
    `parent` property) as well as a node contained in that tree and returns the
    given node's successor.

    A node's successor is the next node to be visited (in an in-order traversal)
    when traversing the tree. If a node has no successor, your function should
    return `None`.

    Each `BinaryTree` node has an integer `value`, a
    `parent` node, a `left` child node, and a `right` child node. Children nodes
    can either be `BinaryTree` nodes themselves or `None`.

    ??? success "Answer"
        ```python
        # O(h) time | O(1) space - where h is the height of the tree
        class BinaryTree:
            def __init__(self, value):
                self.value = value
                self.left = None
                self.right = None
                self.parent = None

        def findSuccessor(tree, node):
            if node.right is not None:
                return getLeftmostChild(node.right)
            return getRightmostParent(node)

        def getLeftmostChild(node):
            currentNode = node
            while currentNode.left is not None:
                currentNode = currentNode.left
            return currentNode

        def getRightmostParent(node):
            currentNode = node
            while currentNode.parent is not None and currentNode == currentNode.parent.right:
                currentNode = currentNode.parent
            return currentNode.parent
        ```

---

### Binary Tree Diameter

!!! question

    Write a function that takes in a Binary Tree and returns its diameter. The
    diameter of a Binary Tree is defined as the length of its longest path, even
    if that path doesn't pass through the root of the tree.

    A path is a collection of connected nodes in a tree, where no node is
    connected to more than two other nodes. The length of a path is the number of
    edges between the path's end nodes.

    Each `BinaryTree` node has an integer `value`, a
    `left` child node, and a `right` child node. Children nodes can either be
    `BinaryTree` nodes themselves or `None`.

    ??? success "Answer"
        ```python
        # O(n) time | O(h) space - where n is the number of nodes in the tree
        # and h is the height of the tree
        class BinaryTree:
            def __init__(self, value):
                self.value = value
                self.left = None
                self.right = None

        def binaryTreeDiameter(tree):
            return getTreeInfo(tree)[1]

        def getTreeInfo(tree):
            if tree is None:
                return [0, 0]  # [height, diameter]

            leftTreeInfo = getTreeInfo(tree.left)
            rightTreeInfo = getTreeInfo(tree.right)

            longestPathThroughRoot = leftTreeInfo[0] + rightTreeInfo[0]
            maxDiameterSoFar = max(leftTreeInfo[1], rightTreeInfo[1])
            currentDiameter = max(longestPathThroughRoot, maxDiameterSoFar)
            currentHeight = 1 + max(leftTreeInfo[0], rightTreeInfo[0])

            return [currentHeight, currentDiameter]
        ```

---

### Lowest Common Ancestor

!!! question

    Write a function that takes in a Binary Tree and two nodes, and returns the
    lowest common ancestor (LCA) of those two nodes.

    A node is considered an ancestor of itself.

    You can assume that all Binary Tree nodes have a `parent` property, and that
    all nodes in the tree are unique.

    Each `BinaryTree` node has an integer `value`, a
    `parent` node, a `left` child node, and a `right` child node. Children nodes
    can either be `BinaryTree` nodes themselves or `None`.

    ??? success "Answer"
        ```python
        # O(d) time | O(1) space - where d is the depth of the deeper node
        class BinaryTree:
            def __init__(self, value, parent=None):
                self.value = value
                self.parent = parent
                self.left = None
                self.right = None

        def lowestCommonAncestor(p, q):
            depthP = getDepth(p)
            depthQ = getDepth(q)

            if depthP > depthQ:
                return backtrackAncestralTree(p, q, depthP - depthQ)
            else:
                return backtrackAncestralTree(q, p, depthQ - depthP)

        def getDepth(node):
            depth = 0
            while node is not None:
                depth += 1
                node = node.parent
            return depth

        def backtrackAncestralTree(lowerNode, higherNode, diff):
            while diff > 0:
                lowerNode = lowerNode.parent
                diff -= 1
            while lowerNode != higherNode:
                lowerNode = lowerNode.parent
                higherNode = higherNode.parent
            return lowerNode
        ```

---

### Invert Binary Tree

!!! question

    Write a function that takes in a Binary Tree, inverts it, and returns the
    inverted tree.

    In other words, the function should swap every left node in the tree for its
    corresponding right node.

    Each `BinaryTree` node has an integer `value`, a
    `left` child node, and a `right` child node. Children nodes can either be
    `BinaryTree` nodes themselves or `None`.

    ??? success "Answer"
        ```python
        # O(n) time | O(d) space - where n is the number of nodes in the tree
        # and d is the depth of the tree
        class BinaryTree:
            def __init__(self, value):
                self.value = value
                self.left = None
                self.right = None

        def invertBinaryTree(tree):
            if tree is None:
                return
            tree.left, tree.right = tree.right, tree.left
            invertBinaryTree(tree.left)
            invertBinaryTree(tree.right)
            return tree
        ```

        ```python
        # Iterative solution - O(n) time | O(n) space
        def invertBinaryTree(tree):
            queue = [tree]
            while len(queue) > 0:
                current = queue.pop(0)
                if current is None:
                    continue
                current.left, current.right = current.right, current.left
                queue.append(current.left)
                queue.append(current.right)
            return tree
        ```

---

### Max Subset Sum With No Adjacent Elements

!!! question

    Write a function that takes in an array of positive integers and returns the
    maximum sum of non-adjacent elements in the array.

    If the input array is empty, the function should return `0`.

    ??? success "Answer"

        ```python
        # O(n) time | O(1) space - where n is the length of the input array
        def maxSubsetSumNoAdjacent(array):
            if not len(array):
                return 0
            elif len(array) == 1:
                return array[0]
            second = array[0]
            first = max(array[0], array[1])
            for i in range(2, len(array)):
                current = max(first, second + array[i])
                second = first
                first = current
            return first
        ```

        ```python
        # O(n) time | O(n) space
        def maxSubsetSumNoAdjacent(array):
            if not len(array):
                return 0
            elif len(array) == 1:
                return array[0]
            maxSums = array[:]
            maxSums[1] = max(array[0], array[1])
            for i in range(2, len(array)):
                maxSums[i] = max(maxSums[i - 1], maxSums[i - 2] + array[i])
            return maxSums[-1]
        ```

---

### Number Of Ways To Make Change

!!! question

    Given an array of distinct positive integers representing coin denominations and a
    single non-negative integer `n` representing a target amount of
    money, write a function that returns the number of ways to make change for
    that target amount using the given coin denominations.

    Note that an unlimited amount of coins is at your disposal.

    ??? success "Answer"
        ```python
        # O(nd) time | O(n) space - where n is the target amount and d is the number of coin denominations
        def numberOfWaysToMakeChange(n, denoms):
            ways = [0 for amount in range(n + 1)]
            ways[0] = 1
            for denom in denoms:
                for amount in range(1, n + 1):
                    if denom <= amount:
                        ways[amount] += ways[amount - denom]
            return ways[n]
        ```

---

### Min Heap Construction

!!! question

    Implement a `MinHeap` class.  The class should support:

    *   Inserting integers.
    *   Removing the minimum integer.
    *   Peeking at the minimum integer.
    *   Sifting integers up and down the heap.

    ??? success "Answer"
        ```python
        # Average and Worst: O(log(n)) time | O(1) space for insertion, removal, and sifting
        # Average and Worst: O(1) time | O(1) space for peeking
        class MinHeap:
            def __init__(self, array):
                self.heap = self.buildHeap(array)

            def buildHeap(self, array):
                firstParentIdx = (len(array) - 2) // 2
                for currentIdx in reversed(range(firstParentIdx + 1)):
                    self.siftDown(currentIdx, len(array) - 1, array)
                return array

            def siftDown(self, currentIdx, endIdx, heap):
                childOneIdx = currentIdx * 2 + 1
                while childOneIdx <= endIdx:
                    childTwoIdx = currentIdx * 2 + 2 if currentIdx * 2 + 2 <= endIdx else -1
                    if childTwoIdx != -1 and heap[childTwoIdx] < heap[childOneIdx]:
                        idxToSwap = childTwoIdx
                    else:
                        idxToSwap = childOneIdx
                    if heap[idxToSwap] < heap[currentIdx]:
                        self.swap(currentIdx, idxToSwap, heap)
                        currentIdx = idxToSwap
                        childOneIdx = currentIdx * 2 + 1
                    else:
                        return

            def siftUp(self, currentIdx, heap):
                parentIdx = (currentIdx - 1) // 2
                while currentIdx > 0 and heap[currentIdx] < heap[parentIdx]:
                    self.swap(currentIdx, parentIdx, heap)
                    currentIdx = parentIdx
                    parentIdx = (currentIdx - 1) // 2

            def peek(self):
                return self.heap[0]

            def remove(self):
                self.swap(0, len(self.heap) - 1, self.heap)
                valueToRemove = self.heap.pop()
                self.siftDown(0, len(self.heap) - 1, self.heap)
                return valueToRemove

            def insert(self, value):
                self.heap.append(value)
                self.siftUp(len(self.heap) - 1, self.heap)

            def swap(self, i, j, heap):
                heap[i], heap[j] = heap[j], heap[i]
        ```

---

### Dijkstra's Algorithm

!!! question

    Implement Dijkstra's algorithm.

    Given a graph represented as an adjacency list (where each key is a node, and the value is a list of edges, where each edge is a tuple of (neighbor_node, weight)), and a starting node, write a function that returns a dictionary containing the shortest distances from the start node to all other nodes in the graph. If a node is unreachable, its distance should be `float('inf')`.

    ??? success "Answer"
        ```python
        # O(v^2 + e) time | O(v) space - where v is the number of vertices
        # and e is the number of edges in the graph
        import heapq

        def dijkstrasAlgorithm(start, graph):
            numberOfNodes = len(graph)
            minDistances = {node: float("inf") for node in range(numberOfNodes)}
            minDistances[start] = 0
            visited = set()

            pq = [(0, start)] # (distance, node)
            while pq:
                currentDistance, currentNode = heapq.heappop(pq)
                if currentNode in visited:
                    continue
                visited.add(currentNode)

                for edge in graph[currentNode]:
                    neighbor, weight = edge
                    distanceToNeighbor = currentDistance + weight
                    if distanceToNeighbor < minDistances[neighbor]:
                        minDistances[neighbor] = distanceToNeighbor
                        heapq.heappush(pq, (distanceToNeighbor, neighbor))

            return minDistances
        ```

---

### Topological Sort

!!! question

    Write a function that takes in a list of arbitrary jobs and a list of dependencies.
    A job is represented by a single character. A dependency is represented as a pair
    of jobs, where the first job is dependent on the second job. In other words, the
    second job must be completed before the first job can be started.

    The function should return a list of the jobs in a valid order. If no valid
    order exists (because the dependencies create a cycle), the function should
    return an empty array.

    ??? success "Answer"
        ```python
        # O(j + d) time | O(j + d) space - where j is the number of jobs
        # and d is the number of dependencies
        def topologicalSort(jobs, deps):
            jobGraph = createJobGraph(jobs, deps)
            return getOrderedJobs(jobGraph)

        def createJobGraph(jobs, deps):
            graph = JobGraph(jobs)
            for job, dep in deps:
                graph.addPrereq(job, dep)
            return graph

        def getOrderedJobs(graph):
            orderedJobs = []
            nodes = list(graph.nodes.values())
            for node in nodes:
                if node.state == "B":
                    continue
                isCyclic = depthFirstTraverse(node, orderedJobs)
                if isCyclic:
                    return []
            return orderedJobs

        def depthFirstTraverse(node, orderedJobs):
            if node.state == "V":
                return False  # Cycle detected
            if node.state == "P":
                return True
            node.state = "V"
            for prereqNode in node.prereqs:
                isCyclic = depthFirstTraverse(prereqNode, orderedJobs)
                if isCyclic:
                    return True
            node.state = "P"
            orderedJobs.append(node.job)
            return False

        class JobGraph:
            def __init__(self, jobs):
                self.nodes = {}
                for job in jobs:
                    self.addNode(job)

            def addNode(self, job):
                if job not in self.nodes:
                    self.nodes[job] = JobNode(job)

            def addPrereq(self, job, prereq):
                jobNode = self.getNode(job)
                prereqNode = self.getNode(prereq)
                jobNode.prereqs.append(prereqNode)

            def getNode(self, job):
                if job not in self.nodes:
                    self.addNode(job)
                return self.nodes[job]

        class JobNode:
            def __init__(self, job):
                self.job = job
                self.prereqs = []
                self.state = "U"  # U = Unvisited, V = Visiting, P = Visited
        ```

---

### Knapsack Problem

!!! question

    Given a set of items, each with a weight and a value, and a knapsack with a
    maximum weight capacity, write a function that determines the maximum total
    value of items that can be placed in the knapsack.

    You can assume that the input items are represented as a list of tuples, where
    each tuple contains the weight and value of an item.

    ??? success "Answer"
        ```python
        # O(nc) time | O(nc) space - where n is the number of items and c is the capacity of the knapsack
        def knapsackProblem(items, capacity):
            knapsackValues = [[0 for _ in range(capacity + 1)] for _ in range(len(items) + 1)]

            for i in range(1, len(items) + 1):
                currentWeight, currentValue = items[i - 1]
                for c in range(capacity + 1):
                    if currentWeight > c:
                        knapsackValues[i][c] = knapsackValues[i - 1][c]
                    else:
                        knapsackValues[i][c] = max(
                            knapsackValues[i - 1][c],
                            knapsackValues[i - 1][c - currentWeight] + currentValue
                        )

            return [knapsackValues[-1][-1], getKnapsackItems(knapsackValues, items)]

        def getKnapsackItems(knapsackValues, items):
            sequence = []
            i = len(knapsackValues) - 1
            c = len(knapsackValues[0]) - 1
            while i > 0:
                if knapsackValues[i][c] == knapsackValues[i - 1][c]:
                    i -= 1
                else:
                    sequence.append(i - 1)
                    c -= items[i - 1][0]
                    i -= 1
                if c == 0:
                    break
            return list(reversed(sequence))
        ```

---

### Dijkstra's Algorithm

!!! question

    Implement Dijkstra's algorithm.

    Given a graph represented as an adjacency list (where each key is a node, and the value is a list of edges, where each edge is a tuple of (neighbor_node, weight)), and a starting node, write a function that returns a dictionary containing the shortest distances from the start node to all other nodes in the graph. If a node is unreachable, its distance should be `float('inf')`.

    ??? success "Answer"
        ```python
        # O(v^2 + e) time | O(v) space - where v is the number of vertices
        # and e is the number of edges in the graph
        import heapq

        def dijkstrasAlgorithm(start, graph):
            numberOfNodes = len(graph)
            minDistances = {node: float("inf") for node in range(numberOfNodes)}
            minDistances[start] = 0
            visited = set()

            pq = [(0, start)] # (distance, node)
            while pq:
                currentDistance, currentNode = heapq.heappop(pq)
                if currentNode in visited:
                    continue
                visited.add(currentNode)

                for edge in graph[currentNode]:
                    neighbor, weight = edge
                    distanceToNeighbor = currentDistance + weight
                    if distanceToNeighbor < minDistances[neighbor]:
                        minDistances[neighbor] = distanceToNeighbor
                        heapq.heappush(pq, (distanceToNeighbor, neighbor))

            return minDistances
        ```

---

### Topological Sort

!!! question

    Write a function that takes in a list of arbitrary jobs and a list of dependencies.
    A job is represented by a single character. A dependency is represented as a pair
    of jobs, where the first job is dependent on the second job. In other words, the
    second job must be completed before the first job can be started.

    The function should return a list of the jobs in a valid order. If no valid
    order exists (because the dependencies create a cycle), the function should
    return an empty array.

    ??? success "Answer"
        ```python
        # O(j + d) time | O(j + d) space - where j is the number of jobs
        # and d is the number of dependencies
        def topologicalSort(jobs, deps):
            jobGraph = createJobGraph(jobs, deps)
            return getOrderedJobs(jobGraph)

        def createJobGraph(jobs, deps):
            graph = JobGraph(jobs)
            for job, dep in deps:
                graph.addPrereq(job, dep)
            return graph

        def getOrderedJobs(graph):
            orderedJobs = []
            nodes = list(graph.nodes.values())
            for node in nodes:
                if node.state == "B":
                    continue
                isCyclic = depthFirstTraverse(node, orderedJobs)
                if isCyclic:
                    return []
            return orderedJobs

        def depthFirstTraverse(node, orderedJobs):
            if node.state == "V":
                return False  # Cycle detected
            if node.state == "P":
                return True
            node.state = "V"
            for prereqNode in node.prereqs:
                isCyclic = depthFirstTraverse(prereqNode, orderedJobs)
                if isCyclic:
                    return True
            node.state = "P"
            orderedJobs.append(node.job)
            return False

        class JobGraph:
            def __init__(self, jobs):
                self.nodes = {}
                for job in jobs:
                    self.addNode(job)

            def addNode(self, job):
                if job not in self.nodes:
                    self.nodes[job] = JobNode(job)

            def addPrereq(self, job, prereq):
                jobNode = self.getNode(job)
                prereqNode = self.getNode(prereq)
                jobNode.prereqs.append(prereqNode)

            def getNode(self, job):
                if job not in self.nodes:
                    self.addNode(job)
                return self.nodes[job]

        class JobNode:
            def __init__(self, job):
                self.job = job
                self.prereqs = []
                self.state = "U"  # U = Unvisited, V = Visiting, P = Visited
        ```

---

### Knapsack Problem

!!! question

    Given a set of items, each with a weight and a value, and a knapsack with a
    maximum weight capacity, write a function that determines the maximum total
    value of items that can be placed in the knapsack.

    You can assume that the input items are represented as a list of tuples, where
    each tuple contains the weight and value of an item.

    ??? success "Answer"
        ```python
        # O(nc) time | O(nc) space - where n is the number of items and c is the capacity of the knapsack
        def knapsackProblem(items, capacity):
            knapsackValues = [[0 for _ in range(capacity + 1)] for _ in range(len(items) + 

```markdown
1)]

            for i in range(1, len(items) + 1):
                currentWeight, currentValue = items[i - 1]
                for c in range(capacity + 1):
                    if currentWeight > c:
                        knapsackValues[i][c] = knapsackValues[i - 1][c]
                    else:
                        knapsackValues[i][c] = max(
                            knapsackValues[i - 1][c],
                            knapsackValues[i - 1][c - currentWeight] + currentValue
                        )

            return [knapsackValues[-1][-1], getKnapsackItems(knapsackValues, items)]

        def getKnapsackItems(knapsackValues, items):
            sequence = []
            i = len(knapsackValues) - 1
            c = len(knapsackValues[0]) - 1
            while i > 0:
                if knapsackValues[i][c] == knapsackValues[i - 1][c]:
                    i -= 1
                else:
                    sequence.append(i - 1)
                    c -= items[i - 1][0]
                    i -= 1
                if c == 0:
                    break
            return list(reversed(sequence))
        ```

        ```python
        # Recursive solution with memoization - O(nc) time | O(nc) space
        def knapsackProblem(items, capacity):

            def knapsackHelper(i, c, memo):
                if (i, c) in memo:
                    return memo[(i, c)]

                if i == 0 or c == 0:
                    result = [0, []]
                elif items[i - 1][0] > c:
                    result = knapsackHelper(i - 1, c, memo)
                else:
                    val_without_current, items_without_current = knapsackHelper(i - 1, c, memo)
                    val_with_current, items_with_current = knapsackHelper(i - 1, c - items[i - 1][0], memo)
                    val_with_current += items[i - 1][1]

                    if val_with_current > val_without_current:
                        result = [val_with_current, items_with_current + [i - 1]]
                    else:
                        result = [val_without_current, items_without_current]

                memo[(i, c)] = result
                return result

            memo = {}
            value, item_indices = knapsackHelper(len(items), capacity, memo)
            return [value, item_indices]
        ```

---

### Disk Stacking

!!! question

    You're given a non-empty array of arrays where each subarray holds three
    integers and represents a disk. These integers denote each disk's width,
    depth, and height, respectively. Your goal is to stack up the disks and to
    maximize the total height of the stack. A disk must have a strictly smaller
    width, depth, and height than any other disk below it.

    Write a function that returns an array of the disks in the final stack,
    starting with the top disk and ending with the bottom disk. Note that you
    can't rotate disks; in other words, the integers in each subarray must
    represent [width, depth, height] at all times.

    You can assume that there will only be one stack with the greatest total
    height.

    ??? success "Answer"
        ```python
        # O(n^2) time | O(n) space - where n is the number of disks
        def diskStacking(disks):
            disks.sort(key=lambda disk: disk[2])  # Sort by height
            heights = [disk[2] for disk in disks]
            sequences = [None for _ in disks]
            maxHeightIdx = 0
            for i in range(1, len(disks)):
                currentDisk = disks[i]
                for j in range(0, i):
                    otherDisk = disks[j]
                    if areValidDimensions(otherDisk, currentDisk):
                        if heights[i] <= currentDisk[2] + heights[j]:
                            heights[i] = currentDisk[2] + heights[j]
                            sequences[i] = j
                if heights[i] >= heights[maxHeightIdx]:
                    maxHeightIdx = i
            return buildSequence(disks, sequences, maxHeightIdx)

        def areValidDimensions(o, c):
            return o[0] < c[0] and o[1] < c[1] and o[2] < c[2]

        def buildSequence(array, sequences, currentIdx):
            sequence = []
            while currentIdx is not None:
                sequence.append(array[currentIdx])
                currentIdx = sequences[currentIdx]
            return list(reversed(sequence))
        ```

---

### Numbers In Pi

!!! question

    Given a string representation of the first `n` digits of Pi and a list of
    positive integers (all in string format), write a function that returns the
    smallest number of spaces that can be added to the n digits of Pi such that
    all resulting numbers are found in the list of positive integers.

    Note that a single number can appear multiple times in the resulting numbers.
    For example, if Pi is "3141592" and the numbers are ["314", "49", "9001", "15926", "5", "3141592"],
    the number "3141592" is in the list and requires 0 spaces to be added.
    If Pi is "3141592653589793238462643383279", and the numbers are ["314159265358979323846", "26433", "8", "3279", "314159265", "35897932384626433832", "79"],
    the numbers "314159265", "35897932384626433832", and "79" are in the list, and they require 2 spaces to be added.

    If no number of spaces to be added exists, such that all resulting numbers are
    found in the list of integers, the function should return -1.

    ??? success "Answer"
        ```python
        # O(n^3 + m) time | O(n + m) space - where n is the number of digits in Pi and m is the number of favorite numbers
        def numbersInPi(pi, numbers):
            numbersTable = {number: True for number in numbers}
            minSpaces = getMinSpaces(pi, numbersTable, {}, 0)
            return -1 if minSpaces == float("inf") else minSpaces

        def getMinSpaces(pi, numbersTable, cache, idx):
            if idx == len(pi):
                return -1
            if idx in cache:
                return cache[idx]
            minSpaces = float("inf")
            for i in range(idx, len(pi)):
                prefix = pi[idx : i + 1]
                if prefix in numbersTable:
                    minSpacesInSuffix = getMinSpaces(pi, numbersTable, cache, i + 1)
                    if minSpacesInSuffix == float("inf"):
                        minSpaces = min(minSpaces, minSpacesInSuffix)
                    else:
                        minSpaces = min(minSpaces, minSpacesInSuffix + 1)
            cache[idx] = minSpaces
            return cache[idx]
        ```

---

### Maximum Sum Submatrix

!!! question

    You're given a two-dimensional array (a matrix) of potentially unequal height
    and width that's filled with integers. You're also given a positive integer
    `size`. Write a function that returns the maximum sum that can be generated
    from a submatrix with dimensions `size * size`.

    For example, consider the following matrix:

    ```
    [
      [2, 4],
      [5, 6],
      [-3, 2],
    ]
    ```

    If `size = 2`, then the 2x2 submatrices to consider are:

    ```
    [2, 4]
    [5, 6]
    ```

    ```
    [5, 6]
    [-3, 2]
    ```

    The sum of the elements in the first submatrix is 17, and the sum of the
    elements in the second submatrix is 10. Thus, your function should return 17.

    **Note:** `size` will always be greater than or equal to 1, and the
    dimensions of the input matrix will always be at least `size * size`.

    ??? success "Answer"
        ```python
        # O(w * h) time | O(w * h) space - where w and h
        # are the width and height of the input matrix
        def maximumSumSubmatrix(matrix, size):
            sums = createSumMatrix(matrix)
            maxSubMatrixSum = float("-inf")

            for row in range(size - 1, len(matrix)):
                for col in range(size - 1, len(matrix[row])):
                    total = sums[row][col]

                    touchesTopBorder = row - size < 0
                    if not touchesTopBorder:
                        total -= sums[row - size][col]

                    touchesLeftBorder = col - size < 0
                    if not touchesLeftBorder:
                        total -= sums[row][col - size]

                    touchesTopOrLeftBorder = touchesTopBorder or touchesLeftBorder
                    if not touchesTopOrLeftBorder:
                        total += sums[row - size][col - size]

                    maxSubMatrixSum = max(maxSubMatrixSum, total)

            return maxSubMatrixSum

        def createSumMatrix(matrix):
            sums = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
            sums[0][0] = matrix[0][0]
            # Fill the first row
            for idx in range(1, len(matrix[0])):
                sums[0][idx] = sums[0][idx - 1] + matrix[0][idx]
            # Fill the first column
            for idx in range(1, len(matrix)):
                sums[idx][0] = sums[idx - 1][0] + matrix[idx][0]
            # Fill the rest of the matrix
            for row in range(1, len(matrix)):
                for col in range(1, len(matrix[row])):
                    sums[row][col] = (
                        sums[row - 1][col]
                        + sums[row][col - 1]
                        - sums[row - 1][col - 1]
                        + matrix[row][col]
                    )
            return sums
        ```

        ```python
        # Brute Force: O(w * h * size^2) time | O(1) space
        def maximumSumSubmatrix(matrix, size):
            max_sum = float('-inf')
            for row in range(len(matrix) - size + 1):
                for col in range(len(matrix[0]) - size + 1):
                    current_sum = 0
                    for i in range(size):
                        for j in range(size):
                            current_sum += matrix[row + i][col + j]
                    max_sum = max(max_sum, current_sum)
            return max_sum
        ```

---

### Maximize Expression

!!! question

    Write a function that takes in an array of integers and returns the largest
    possible value for the expression `array[a] - array[b] + array[c] - array[d]`,
    where `a`, `b`, `c`, and `d` are indices of the array and `a < b < c < d`.

    If the input array has fewer than 4 elements, your function should return 0.

    ??? success "Answer"
        ```python
        # O(n) time | O(n) space - where n is the length of the array
        def maximizeExpression(array):
            if len(array) < 4:
                return 0

            maxOfA = [array[0]]
            maxOfAMinusB = [float("-inf")]
            maxOfAMinusBPlusC = [float("-inf")] * 2
            maxOfAMinusBPlusCMinusD = [float("-inf")] * 3

            for idx in range(1, len(array)):
                currentMax = max(maxOfA[idx - 1], array[idx])
                maxOfA.append(currentMax)

            for idx in range(1, len(array)):
                currentMax = max(maxOfAMinusB[idx - 1], maxOfA[idx - 1] - array[idx])
                maxOfAMinusB.append(currentMax)

            for idx in range(2, len(array)):
                currentMax = max(maxOfAMinusBPlusC[idx - 1], maxOfAMinusB[idx - 1] + array[idx])
                maxOfAMinusBPlusC.append(currentMax)

            for idx in range(3, len(array)):
                currentMax = max(maxOfAMinusBPlusCMinusD[idx - 1], maxOfAMinusBPlusC[idx - 1] - array[idx])
                maxOfAMinusBPlusCMinusD.append(currentMax)

            return maxOfAMinusBPlusCMinusD[-1]
        ```

        ```python
        # Another O(n) time | O(n) space solution
        def maximizeExpression(array):
            if len(array) < 4:
                return 0

            maxOfA = [array[0]] * len(array)
            maxOfAMinusB = [float("-inf")] * len(array)
            maxOfAMinusBPlusC = [float("-inf")] * len(array)
            maxOfAMinusBPlusCMinusD = [float("-inf")] * len(array)

            for idx in range(1, len(array)):
                maxOfA[idx] = max(maxOfA[idx - 1], array[idx])

            for idx in range(1, len(array)):
                maxOfAMinusB[idx] = max(maxOfAMinusB[idx - 1], maxOfA[idx - 1] - array[idx])

            for idx in range(2, len(array)):
                maxOfAMinusBPlusC[idx] = max(maxOfAMinusBPlusC[idx - 1], maxOfAMinusB[idx - 1] + array[idx])

            for idx in range(3, len(array)):
                maxOfAMinusBPlusCMinusD[idx] = max(
                    maxOfAMinusBPlusCMinusD[idx - 1], maxOfAMinusBPlusC[idx - 1] - array[idx]
                )

            return maxOfAMinusBPlusCMinusD[len(array) - 1]
        ```

        ```python
        # Brute Force: O(n^4) time | O(1) space
        def maximizeExpression(array):
            if len(array) < 4:
                return 0

            maximumValueFound = float("-inf")

            for a in range(len(array)):
                aValue = array[a]
                for b in range(a + 1, len(array)):
                    bValue = array[b]
                    for c in range(b + 1, len(array)):
                        cValue = array[c]
                        for d in range(c + 1, len(array)):
                            dValue = array[d]
                            expressionValue = aValue - bValue + cValue - dValue
                            maximumValueFound = max(expressionValue, maximumValueFound)

            return maximumValueFound
        ```

---

### Dijkstra's Algorithm

!!! question

    Implement Dijkstra's algorithm.

    Given a graph represented as an adjacency list (where each key is a node, and the value is a list of edges, where each edge is a tuple of (neighbor_node, weight)), and a starting node, write a function that returns a dictionary containing the shortest distances from the start node to all other nodes in the graph. If a node is unreachable, its distance should be `float('inf')`.

    ??? success "Answer"
        ```python
        # O(v^2 + e) time | O(v) space - where v is the number of vertices
        # and e is the number of edges in the graph
        import heapq

        def dijkstrasAlgorithm(start, graph):
            numberOfNodes = len(graph)
            minDistances = {node: float("inf") for node in range(numberOfNodes)}
            minDistances[start] = 0
            visited = set()

            pq = [(0, start)] # (distance, node)
            while pq:
                currentDistance, currentNode = heapq.heappop(pq)
                if currentNode in visited:
                    continue
                visited.add(currentNode)

                for edge in graph[currentNode]:
                    neighbor, weight = edge
                    distanceToNeighbor = currentDistance + weight
                    if distanceToNeighbor < minDistances[neighbor]:
                        minDistances[neighbor] = distanceToNeighbor
                        heapq.heappush(pq, (distanceToNeighbor, neighbor))

            return minDistances
        ```
        ```python
        # Using adjacency matrix: O(V^2)
        def dijkstrasAlgorithm(graph, start):
            V = len(graph)
            dist = [float('inf')] * V
            dist[start] = 0
            visited = [False] * V

            for _ in range(V):
                # Find the vertex with the minimum distance value, from the set of vertices not yet included in shortest path tree
                min_dist = float('inf')
                min_index = -1
                for v in range(V):
                    if dist[v] < min_dist and not visited[v]:
                        min_dist = dist[v]
                        min_index = v

                if min_index == -1: # No more reachable nodes
                    break

                visited[min_index] = True

                # Update dist value of the adjacent vertices of the picked vertex.
                for v in range(V):
                    if (graph[min_index][v] > 0 and  # There's an edge
                        not visited[v] and           # The neighbor hasn't been visited
                        dist[v] > dist[min_index] + graph[min_index][v]): # Shorter path found
                        dist[v] = dist[min_index] + graph[min_index][v]

            return dist
        ```

---

### Topological Sort

!!! question

    Write a function that takes in a list of arbitrary jobs and a list of dependencies.
    A job is represented by a single character. A dependency is represented as a pair
    of jobs, where the first job is dependent on the second job. In other words, the
    second job must be completed before the first job can be started.

    The function should return a list of the jobs in a valid order. If no valid
    order exists (because the dependencies create a cycle), the function should
    return an empty array.

    ??? success "Answer"
        ```python
        # O(j + d) time | O(j + d) space - where j is the number of jobs
        # and d is the number of dependencies
        def topologicalSort(jobs, deps):
            jobGraph = createJobGraph(jobs, deps)
            return getOrderedJobs(jobGraph)

        def createJobGraph(jobs, deps):
            graph = JobGraph(jobs)
            for job, dep in deps:
                graph.addPrereq(job, dep)
            return graph

        def getOrderedJobs(graph):
            orderedJobs = []
            nodes = list(graph.nodes.values())
            for node in nodes:
                if node.state == "B": # B = Blank (unvisited)
                    continue
                isCyclic = depthFirstTraverse(node, orderedJobs)
                if isCyclic:
                    return []
            return orderedJobs

        def depthFirstTraverse(node, orderedJobs):
            if node.state == "V": # V = Visiting (currently in the call stack)
                return True  # Cycle detected
            if node.state == "P": # P = Processed (already visited)
                return False
            node.state = "V"
            for prereqNode in node.prereqs:
                isCyclic = depthFirstTraverse(prereqNode, orderedJobs)
                if isCyclic:
                    return True
            node.state = "P"
            orderedJobs.append(node.job)
            return False

        class JobGraph:
            def __init__(self, jobs):
                self.nodes = {}
                for job in jobs:
                    self.addNode(job)

            def addNode(self, job):
                if job not in self.nodes:
                    self.nodes[job] = JobNode(job)

            def addPrereq(self, job, prereq):
                jobNode = self.getNode(job)
                prereqNode = self.getNode(prereq)
                jobNode.prereqs.append(prereqNode)

            def getNode(self, job):
                if job not in self.nodes:
                    self.addNode(job)
                return self.nodes[job]

        class JobNode:
            def __init__(self, job):
                self.job = job
                self.prereqs = []
                self.state = "U"  # U = Unvisited, V = Visiting, P = Visited
        ```

        ```python
        # Kahn's Algorithm (BFS-based) - O(j + d) time | O(j + d) space
        def topologicalSort(jobs, deps):
            jobGraph = createJobGraph(jobs, deps)
            return getOrderedJobs(jobGraph)

        def createJobGraph(jobs, deps):
            graph = JobGraph(jobs)
            for job, dep in deps:
                graph.addDep(job, dep)  # Note: addDep instead of addPrereq
            return graph

        def getOrderedJobs(graph):
            orderedJobs = []
            nodesWithNoPrereqs = [node for node in graph.nodes.values() if node.numOfPrereqs == 0]
            while len(nodesWithNoPrereqs):
                node = nodesWithNoPrereqs.pop()
                orderedJobs.append(node.job)
                removeDeps(node, nodesWithNoPrereqs)
            # If there are still nodes with prereqs, there's a cycle
            for node in graph.nodes.values():
                if node.numOfPrereqs > 0:
                    return []
            return orderedJobs

        def removeDeps(node, nodesWithNoPrereqs):
            while len(node.dependants):
                dependant = node.dependants.pop()
                dependant.numOfPrereqs -= 1
                if dependant.numOfPrereqs == 0:
                    nodesWithNoPrereqs.append(dependant)

        class JobGraph:
            def __init__(self, jobs):
                self.nodes = {}
                for job in jobs:
                    self.addNode(job)

            def addNode(self, job):
                if job not in self.nodes:
                    self.nodes[job] = JobNode(job)

            def addDep(self, job, dep): # Note: addDep
                jobNode = self.getNode(job)
                depNode = self.getNode(dep)
                jobNode.dependants.append(depNode) # job depends on dep
                depNode.numOfPrereqs += 1

            def getNode(self, job):
                if job not in self.nodes:
                    self.addNode(job)
                return self.nodes[job]

        class JobNode:
            def __init__(self, job):
                self.job = job
                self.dependants = [] # List of nodes that depend on this node
                self.numOfPrereqs = 0
        ```

---

### Knapsack Problem

!!! question

    Given a set of items, each with a weight and a value, and a knapsack with a
    maximum weight capacity, write a function that determines the maximum total
    value of items that can be placed in the knapsack.

    You can assume that the input items are represented as a list of tuples, where
    each tuple contains the weight and value of an item.

    ??? success "Answer"
        ```python
        # O(nc) time | O(nc) space - where n is the number of items and c is the capacity of the knapsack
        def knapsackProblem(items, capacity):
            knapsackValues = [[0 for _ in range(capacity + 1)] for _ in range(len(items) + 1)]

            for i in range(1, len(items) + 1):
                currentWeight, currentValue = items[i - 1]
                for c in range(capacity + 1):
                    if currentWeight > c:
                        knapsackValues[i][c] = knapsackValues[i - 1][c]
                    else:
                        knapsackValues[i][c] = max(
                            knapsackValues[i - 1][c],
                            knapsackValues[i - 1][c - currentWeight] + currentValue
                        )

            return [knapsackValues[-1][-1], getKnapsackItems(knapsackValues, items)]

        def getKnapsackItems(knapsackValues, items):
            sequence = []
            i = len(knapsackValues) - 1
            c = len(knapsackValues[0]) - 1
            while i > 0:
                if knapsackValues[i][c] == knapsackValues[i - 1][c]:
                    i -= 1
                else:
                    sequence.append(i - 1)
                    c -= items[i - 1][0]
                    i -= 1
                if c == 0:
                    break
            return list(reversed(sequence))
        ```

        ```python
        # Recursive solution with memoization - O(nc) time | O(nc) space
        def knapsackProblem(items, capacity):

            def knapsackHelper(i, c, memo):
                if (i, c) in memo:
                    return memo[(i, c)]

                if i == 0 or c == 0:
                    result = [0, []]
                elif items[i - 1][0] > c:
                    result = knapsackHelper(i - 1, c, memo)
                else:
                    val_without_current, items_without_current = knapsackHelper(i - 1, c, memo)
                    val_with_current, items_with_current = knapsackHelper(i - 1, c - items[i - 1][0], memo)
                    val_with_current += items[i - 1][1]

                    if val_with_current > val_without_current:
                        result = [val_with_current, items_with_current + [i - 1]]
                    else:
                        result = [val_without_current, items_without_current]

                memo[(i, c)] = result
                return result

            memo = {}
            value, item_indices = knapsackHelper(len(items), capacity, memo)
            return [value, item_indices]
        ```

---

### Disk Stacking

!!! question

    You're given a non-empty array of arrays where each subarray holds three
    integers and represents a disk. These integers denote each disk's width,
    depth, and height, respectively. Your goal is to stack up the disks and to
    maximize the total height of the stack. A disk must have a strictly smaller
    width, depth, and height than any other disk below it.

    Write a function that returns an array of the disks in the final stack,
    starting with the top disk and ending with the bottom disk. Note that you
    can't rotate disks; in other words, the integers in each subarray must
    represent [width, depth, height] at all times.

    You can assume that there will only be one stack with the greatest total
    height.

    ??? success "Answer"
        ```python
        # O(n^2) time | O(n) space - where n is the number of disks
        def diskStacking(disks):
            disks.sort(key=lambda disk: disk[2])  # Sort by height
            heights = [disk[2] for disk in disks]
            sequences = [None for _ in disks]
            maxHeightIdx = 0
            for i in range(1, len(disks)):
                currentDisk = disks[i]
                for j in range(0, i):
                    otherDisk = disks[j]
                    if areValidDimensions(otherDisk, currentDisk):
                        if heights[i] <= currentDisk[2] + heights[j]:
                            heights[i] = currentDisk[2] + heights[j]
                            sequences[i] = j
                if heights[i] >= heights[maxHeightIdx]:
                    maxHeightIdx = i
            return buildSequence(disks, sequences, maxHeightIdx)

        def areValidDimensions(o, c):
            return o[0] < c[0] and o[1] < c[1] and o[2] < c[2]

        def buildSequence(array, sequences, currentIdx):
            sequence = []
            while currentIdx is not None:
                sequence.append(array[currentIdx])
                currentIdx = sequences[currentIdx]
            return list(reversed(sequence))
        ```

---

### Numbers In Pi

!!! question

    Given a string representation of the first `n` digits of Pi and a list of
    positive integers (all in string format), write a function that returns the
    smallest number of spaces that can be added to the n digits of Pi such that
    all resulting numbers are found in the list of positive integers.

    Note that a single number can appear multiple times in the resulting numbers.
    For example, if Pi is "3141592" and the numbers are ["314", "49", "9001", "15926", "5", "3141592"],
    the number "3141592" is in the list and requires 0 spaces to be added.
    If Pi is "3141592653589793238462643383279", and the numbers are ["314159265358979323846", "26433", "8", "3279", "314159265", "35897932384626433832", "79"],
    the numbers "314159265", "35897932384626433832", and "79" are in the list, and they require 2 spaces to be added.

    If no number of spaces to be added exists, such that all resulting numbers are
    found in the list of integers, the function should return -1.

    ??? success "Answer"
        ```python
        # O(n^3 + m) time | O(n + m) space - where n is the number of digits in Pi and m is the number of favorite numbers
        def numbersInPi(pi, numbers):
            numbersTable = {number: True for number in numbers}
            minSpaces = getMinSpaces(pi, numbersTable, {}, 0)
            return -1 if minSpaces == float("inf") else minSpaces

        def getMinSpaces(pi, numbersTable, cache, idx):
            if idx == len(pi):
                return -1
            if idx in cache:
                return cache[idx]
            minSpaces = float("inf")
            for i in range(idx, len(pi)):
                prefix = pi[idx : i + 1]
                if prefix in numbersTable:
                    minSpacesInSuffix = getMinSpaces(pi, numbersTable, cache, i + 1)
                    if minSpacesInSuffix == float("inf"):
                        minSpaces = min(minSpaces, minSpacesInSuffix)
                    else:
                        minSpaces = min(minSpaces, minSpacesInSuffix + 1)
            cache[idx] = minSpaces
            return cache[idx]
        ```

---
### Longest Common Subsequence

!!! question
    Write a function that takes in two strings and returns their longest common
    subsequence.

    A subsequence of a string is a set of characters that aren't necessarily
    adjacent in the string but that are in the same order as they appear in the
    string. For instance, the characters `["a", "c", "d"]` form a subsequence of
    the string `"abcd"`, and so do the characters `["b", "d"]`. Note that a single
    character in a string and the string itself are both valid subsequences of the
    string.

    You can assume that there will only be one longest common subsequence.

    ??? success "Answer"
        ```python
        # O(nm) time | O(nm) space - where n and m are the lengths of the two input strings
        def longestCommonSubsequence(str1, str2):
            lcs = [[[] for _ in range(len(str1) + 1)] for _ in range(len(str2) + 1)]
            for i in range(1, len(str2) + 1):
                for j in range(1, len(str1) + 1):
                    if str2[i - 1] == str1[j - 1]:
                        lcs[i][j] = lcs[i - 1][j - 1] + [str2[i - 1]]
                    else:
                        lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1], key=len)
            return lcs[-1][-1]
        ```

        ```python
        # O(nm) time | O(nm) space - where n and m are the lengths of the two input strings
        # This solution builds up the lengths of the LCS instead of the actual subsequences.
        def longestCommonSubsequence(str1, str2):
            lengths = [[0 for _ in range(len(str1) + 1)] for _ in range(len(str2) + 1)]
            for i in range(1, len(str2) + 1):
                for j in range(1, len(str1) + 1):
                    if str2[i - 1] == str1[j - 1]:
                        lengths[i][j] = lengths[i - 1][j - 1] + 1
                    else:
                        lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
            return buildSequence(lengths, str1)

        def buildSequence(lengths, string):
            sequence = []
            i = len(lengths) - 1
            j = len(lengths[0]) - 1
            while i != 0 and j != 0:
                if lengths[i][j] == lengths[i - 1][j]:
                    i -= 1
                elif lengths[i][j] == lengths[i][j - 1]:
                    j -= 1
                else:
                    sequence.append(string[j - 1])
                    i -= 1
                    j -= 1
            return list(reversed(sequence))
        ```
        ```python
        # O(nm*min(n,m)) time | O(nm*min(n,m)) space
        def longestCommonSubsequence(str1, str2):
            small = str1 if len(str1) < len(str2) else str2
            big = str1 if len(str1) >= len(str2) else str2
            evenLCS = [[] for _ in range(len(small) + 1)]
            oddLCS = [[] for _ in range(len(small) + 1)]
            for i in range(1, len(big) + 1):
                if i % 2 == 1:
                    currentLCS = oddLCS
                    previousLCS = evenLCS
                else:
                    currentLCS = evenLCS
                    previousLCS = oddLCS
                for j in range(1, len(small) + 1):
                    if big[i - 1] == small[j - 1]:
                        currentLCS[j] = previousLCS[j - 1] + [big[i - 1]]
                    else:
                        currentLCS[j] = max(previousLCS[j], currentLCS[j - 1], key=len)
            return evenLCS[-1] if len(big) % 2 == 0 else oddLCS[-1]
        ```

---
### Min Number of Jumps

!!! question

    You're given a non-empty array of positive integers where each integer
    represents the maximum number of steps you can take forward in the array. For
    example, if the element at index `1` is `3`, you can go from index `1` to index
    `2`, `3`, or `4`.

    Write a function that returns the minimum number of jumps needed to reach the
    final index.

    Note that jumping from index `i` to index `i + x` always constitutes one jump,
    no matter how large `x` is.

    ??? success "Answer"
        ```python
        # O(n) time | O(1) space - where n is the length of the array
        def minNumberOfJumps(array):
            if len(array) == 1:
                return 0
            jumps = 0
            maxReach = array[0]
            steps = array[0]
            for i in range(1, len(array) - 1):
                maxReach = max(maxReach, i + array[i])
                steps -= 1
                if steps == 0:
                    jumps += 1
                    steps = maxReach - i
            return jumps + 1
        ```

        ```python
        # O(n^2) time | O(n) space
        def minNumberOfJumps(array):
            jumps = [float("inf") for x in array]
            jumps[0] = 0
            for i in range(1, len(array)):
                for j in range(0, i):
                    if array[j] >= i - j:
                        jumps[i] = min(jumps[j] + 1, jumps[i])
            return jumps[-1]
        ```

---

### Water Area (Trapping Rain Water)

!!! question

    You're given an array of non-negative integers representing the heights of
    bars in a histogram.  Assume that each bar has a width of 1.  Write a
    function that returns the amount of water that would be trapped between the
    bars after it rains.

    For example, given the input array `[0, 8, 0, 0, 5, 0, 0, 10, 0, 0, 1, 1, 0, 3]`,
    the function should return `48`.

    ??? example "Try it!"
        - LeetCode: https://leetcode.com/problems/trapping-rain-water/

    ??? success "Answer"
        ```python
        # O(n) time | O(1) space - where n is the length of the input array
        def waterArea(heights):
            if len(heights) == 0:
                return 0

            leftIdx = 0
            rightIdx = len(heights) - 1
            leftMax = heights[leftIdx]
            rightMax = heights[rightIdx]
            surfaceArea = 0

            while leftIdx < rightIdx:
                if heights[leftIdx] < heights[rightIdx]:
                    leftIdx += 1
                    leftMax = max(leftMax, heights[leftIdx])
                    surfaceArea += leftMax - heights[leftIdx]
                else:
                    rightIdx -= 1
                    rightMax = max(rightMax, heights[rightIdx])
                    surfaceArea += rightMax - heights[rightIdx]

            return surfaceArea
        ```

        ```python
        # O(n) time | O(n) space
        def waterArea(heights):
            maxes = [0 for x in heights]
            leftMax = 0
            for i in range(len(heights)):
                height = heights[i]
                maxes[i] = leftMax
                leftMax = max(leftMax, height)
            rightMax = 0
            for i in reversed(range(len(heights))):
                height = heights[i]
                minHeight = min(rightMax, maxes[i])
                if height < minHeight:
                    maxes[i] = minHeight - height
                else:
                    maxes[i] = 0
                rightMax = max(rightMax, height)
            return sum(maxes)
        ```

---

### Minimum Characters For Palindrome

!!! question

    Write a function that takes in a non-empty string and returns the minimum
    number of characters that must be added to the front of the string to make
    the resulting string a palindrome.

    A palindrome is defined as a string that's written the same forward and
    backward.

    ??? success "Answer"
        ```python
        # O(n^2) time | O(n) space - where n is the length of the input string
        def minimumCharactersForPalindrome(string):
            if string == string[::-1]:
                return 0

            for i in range(len(string) - 1, -1, -1):
                if string[:i] == string[:i][::-1]:
                    return len(string) - i
        ```

        ```python
        # KMP Algorithm - O(n) time | O(n) space
        def minimumCharactersForPalindrome(string):
            temp = string + "?" + string[::-1]  # '?' prevents false positives
            lps = [0] * len(temp)
            j = 0
            for i in range(1, len(temp)):
                while j > 0 and temp[i] != temp[j]:
                    j = lps[j - 1]
                if temp[i] == temp[j]:
                    j += 1
                lps[i] = j
            return len(string) - lps[-1]
        ```

---

        