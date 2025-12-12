
---
title: Data Structures and Algorithms (DSA) Interview Questions
description: 100+ DSA interview questions with Python solutions - arrays, trees, graphs, dynamic programming, sorting algorithms, and time complexity analysis for coding interviews.
---

# Data Structures and Algorithms (DSA)

<!-- ![Total Questions](https://img.shields.io/badge/Total%20Questions-25-blue?style=flat&labelColor=black&color=blue)
![Unanswered Questions](https://img.shields.io/badge/Unanswered%20Questions-0-blue?style=flat&labelColor=black&color=yellow)
![Answered Questions](https://img.shields.io/badge/Answered%20Questions-25-blue?style=flat&labelColor=black&color=success) -->

<!-- [TOC] -->

This document provides a curated list of Data Structures and Algorithms (DSA) questions commonly asked in technical interviews.  It covers a wide range of difficulty levels and topics.

This is updated frequently but right now this is the most exhaustive list of type of questions being asked.

---

## Premium Interview Questions

### Two Sum - Find Pairs with Target Sum - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Array`, `Hash Table`, `Two Pointers` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Problem:** Find two numbers in array that sum to target.
    
    **Approach 1: Hash Map O(n)**
    
    ```python
    def two_sum(nums, target):
        """Return indices of two numbers that add to target"""
        seen = {}  # value -> index
        
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        
        return []
    
    # Example
    nums = [2, 7, 11, 15]
    target = 9
    print(two_sum(nums, target))  # [0, 1]
    ```
    
    **Approach 2: Two Pointers O(n log n)**
    
    ```python
    def two_sum_sorted(nums, target):
        """For sorted array or when indices don't matter"""
        nums_sorted = sorted(enumerate(nums), key=lambda x: x[1])
        left, right = 0, len(nums) - 1
        
        while left < right:
            curr_sum = nums_sorted[left][1] + nums_sorted[right][1]
            if curr_sum == target:
                return [nums_sorted[left][0], nums_sorted[right][0]]
            elif curr_sum < target:
                left += 1
            else:
                right -= 1
        return []
    ```
    
    **Time Complexity:**
    
    | Approach | Time | Space |
    |----------|------|-------|
    | Hash Map | O(n) | O(n) |
    | Two Pointers | O(n log n) | O(1) |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Hash table usage, handling duplicates.
        
        **Strong answer signals:**
        
        - Uses hash map for O(n) solution
        - Handles edge cases (no solution, duplicates)
        - Discusses trade-offs between approaches
        - Can extend to 3Sum, kSum problems

---

### Reverse a Linked List - Amazon, Meta Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Linked List`, `Pointers`, `Recursion` | **Asked by:** Amazon, Meta, Microsoft

??? success "View Answer"

    **Iterative Approach:**
    
    ```python
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
    
    def reverse_list(head):
        """Reverse linked list iteratively - O(n) time, O(1) space"""
        prev = None
        current = head
        
        while current:
            next_node = current.next  # Save next
            current.next = prev       # Reverse pointer
            prev = current            # Move prev forward
            current = next_node       # Move current forward
        
        return prev
    ```
    
    **Recursive Approach:**
    
    ```python
    def reverse_list_recursive(head):
        """Reverse linked list recursively - O(n) time, O(n) space"""
        if not head or not head.next:
            return head
        
        new_head = reverse_list_recursive(head.next)
        head.next.next = head  # Reverse the link
        head.next = None       # Set original next to None
        
        return new_head
    ```
    
    **Visual:**
    
    ```
    Before: 1 -> 2 -> 3 -> 4 -> None
    After:  4 -> 3 -> 2 -> 1 -> None
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Pointer manipulation, edge cases.
        
        **Strong answer signals:**
        
        - Shows both iterative and recursive
        - Draws diagram to explain
        - Handles empty list and single node
        - Mentions space complexity difference

---

### Valid Parentheses - Using Stack - Amazon, Google Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Stack`, `String`, `Matching` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Solution:**
    
    ```python
    def is_valid(s):
        """Check if parentheses are balanced"""
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        
        for char in s:
            if char in mapping:  # Closing bracket
                if not stack or stack[-1] != mapping[char]:
                    return False
                stack.pop()
            else:  # Opening bracket
                stack.append(char)
        
        return len(stack) == 0
    
    # Examples
    print(is_valid("()[]{}"))  # True
    print(is_valid("([)]"))    # False
    print(is_valid("{[]}"))    # True
    ```
    
    **Time & Space:** O(n), O(n)
    
    **Edge Cases:**
    
    - Empty string: True
    - Single bracket: False
    - Nested `{[()]}`): True
    - Wrong order `([)]`: False

    !!! tip "Interviewer's Insight"
        **What they're testing:** Stack usage, matching logic.
        
        **Strong answer signals:**
        
        - Uses dictionary for bracket pairs
        - Checks stack not empty before pop
        - Returns `len(stack) == 0` not just True
        - Discusses extensions (longest valid, generate valid)

---

### Binary Search - Iterative and Recursive - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Binary Search`, `Array`, `Divide & Conquer` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Iterative (Preferred):**
    
    ```python
    def binary_search(nums, target):
        """Returns index of target or -1 if not found"""
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2  # Avoid overflow
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    ```
    
    **Recursive:**
    
    ```python
    def binary_search_recursive(nums, target, left=0, right=None):
        if right is None:
            right = len(nums) - 1
        
        if left > right:
            return -1
        
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            return binary_search_recursive(nums, target, mid + 1, right)
        else:
            return binary_search_recursive(nums, target, left, mid - 1)
    ```
    
    **Variants:**
    
    | Variant | Use Case |
    |---------|----------|
    | Lower bound | First >= target |
    | Upper bound | Last <= target |
    | Rotated array | Find pivot first |
    
    **Time:** O(log n), **Space:** O(1) iterative, O(log n) recursive

    !!! tip "Interviewer's Insight"
        **What they're testing:** Binary search correctness, off-by-one errors.
        
        **Strong answer signals:**
        
        - Uses `left <= right` (not `<`)
        - Avoids overflow with `left + (right - left) // 2`
        - Handle empty array
        - Knows when to use `< target` vs `<= target`

---

### Maximum Subarray - Kadane's Algorithm - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Array`, `Dynamic Programming`, `Greedy` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Kadane's Algorithm:**
    
    ```python
    def max_subarray(nums):
        """Find contiguous subarray with largest sum"""
        max_sum = nums[0]
        current_sum = nums[0]
        
        for i in range(1, len(nums)):
            # Either extend previous subarray or start new one
            current_sum = max(nums[i], current_sum + nums[i])
            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    # Example
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(max_subarray(nums))  # 6 (subarray [4, -1, 2, 1])
    ```
    
    **With Indices:**
    
    ```python
    def max_subarray_with_indices(nums):
        max_sum = current_sum = nums[0]
        start = end = temp_start = 0
        
        for i in range(1, len(nums)):
            if nums[i] > current_sum + nums[i]:
                current_sum = nums[i]
                temp_start = i
            else:
                current_sum += nums[i]
            
            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i
        
        return max_sum, (start, end)
    ```
    
    **Time:** O(n), **Space:** O(1)

    !!! tip "Interviewer's Insight"
        **What they're testing:** DP intuition, handling negative numbers.
        
        **Strong answer signals:**
        
        - Explains "extend or restart" decision
        - Handles all negative array
        - Can return actual subarray indices
        - Mentions divide & conquer alternative O(n log n)

---

### Merge Two Sorted Lists - Meta, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Linked List`, `Two Pointers`, `Recursion` | **Asked by:** Meta, Amazon, Google

??? success "View Answer"

    **Iterative with Dummy Node:**
    
    ```python
    def merge_two_lists(l1, l2):
        """Merge two sorted linked lists"""
        dummy = ListNode(0)
        current = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
        
        # Attach remaining nodes
        current.next = l1 if l1 else l2
        
        return dummy.next
    ```
    
    **Recursive:**
    
    ```python
    def merge_two_lists_recursive(l1, l2):
        if not l1:
            return l2
        if not l2:
            return l1
        
        if l1.val <= l2.val:
            l1.next = merge_two_lists_recursive(l1.next, l2)
            return l1
        else:
            l2.next = merge_two_lists_recursive(l1, l2.next)
            return l2
    ```
    
    **Time:** O(n + m), **Space:** O(1) iterative, O(n + m) recursive

    !!! tip "Interviewer's Insight"
        **What they're testing:** Pointer manipulation, merge logic.
        
        **Strong answer signals:**
        
        - Uses dummy node to simplify edge cases
        - Handles when one list is exhausted
        - Extends to merge K sorted lists (heap approach)
        - Discusses in-place vs new list

---

### LRU Cache Design - Amazon, Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Hash Table`, `Doubly Linked List`, `Design` | **Asked by:** Amazon, Google, Meta, Microsoft

??? success "View Answer"

    **Implementation:**
    
    ```python
    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self.cache = {}  # key -> node
            
            # Doubly linked list with dummy head/tail
            self.head = Node(0, 0)
            self.tail = Node(0, 0)
            self.head.next = self.tail
            self.tail.prev = self.head
        
        def get(self, key):
            if key in self.cache:
                node = self.cache[key]
                self._remove(node)
                self._add(node)
                return node.val
            return -1
        
        def put(self, key, value):
            if key in self.cache:
                self._remove(self.cache[key])
            
            node = Node(key, value)
            self._add(node)
            self.cache[key] = node
            
            if len(self.cache) > self.capacity:
                # Remove LRU (just after head)
                lru = self.head.next
                self._remove(lru)
                del self.cache[lru.key]
        
        def _remove(self, node):
            node.prev.next = node.next
            node.next.prev = node.prev
        
        def _add(self, node):
            # Add to end (before tail = most recently used)
            prev = self.tail.prev
            prev.next = node
            node.prev = prev
            node.next = self.tail
            self.tail.prev = node
    
    class Node:
        def __init__(self, key, val):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None
    ```
    
    **Alternative:** Use `OrderedDict`:
    
    ```python
    from collections import OrderedDict
    
    class LRUCache:
        def __init__(self, capacity):
            self.cache = OrderedDict()
            self.capacity = capacity
        
        def get(self, key):
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return -1
        
        def put(self, key, value):
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Hash + doubly linked list combo.
        
        **Strong answer signals:**
        
        - Uses dummy nodes for cleaner code
        - O(1) for both get and put
        - Explains why doubly linked (O(1) removal)
        - Can discuss LFU cache as follow-up

---

### Longest Substring Without Repeating Characters - Sliding Window - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Sliding Window`, `Hash Set`, `String` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Sliding Window Solution:**
    
    ```python
    def length_of_longest_substring(s):
        """Find length of longest substring without repeating chars"""
        char_index = {}  # Character -> last index
        max_length = 0
        start = 0
        
        for end, char in enumerate(s):
            if char in char_index and char_index[char] >= start:
                start = char_index[char] + 1
            
            char_index[char] = end
            max_length = max(max_length, end - start + 1)
        
        return max_length
    
    # Examples
    print(length_of_longest_substring("abcabcbb"))  # 3 ("abc")
    print(length_of_longest_substring("bbbbb"))     # 1 ("b")
    print(length_of_longest_substring("pwwkew"))    # 3 ("wke")
    ```
    
    **Alternative with Set:**
    
    ```python
    def length_of_longest_substring_set(s):
        seen = set()
        start = max_length = 0
        
        for end, char in enumerate(s):
            while char in seen:
                seen.remove(s[start])
                start += 1
            seen.add(char)
            max_length = max(max_length, end - start + 1)
        
        return max_length
    ```
    
    **Time:** O(n), **Space:** O(min(n, alphabet size))

    !!! tip "Interviewer's Insight"
        **What they're testing:** Sliding window technique.
        
        **Strong answer signals:**
        
        - Uses hash map for O(1) lookup
        - Handles the `char_index[char] >= start` check
        - Can extend to "at most K distinct characters"
        - Explains window expansion/contraction

---

### Climbing Stairs - Dynamic Programming - Amazon, Google Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Dynamic Programming`, `Fibonacci`, `Memoization` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **DP Bottom-Up (O(1) space):**
    
    ```python
    def climb_stairs(n):
        """Ways to climb n stairs (1 or 2 steps at a time)"""
        if n <= 2:
            return n
        
        prev2, prev1 = 1, 2
        
        for i in range(3, n + 1):
            current = prev1 + prev2
            prev2 = prev1
            prev1 = current
        
        return prev1
    ```
    
    **With Memoization:**
    
    ```python
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def climb_stairs_memo(n):
        if n <= 2:
            return n
        return climb_stairs_memo(n - 1) + climb_stairs_memo(n - 2)
    ```
    
    **General K Steps:**
    
    ```python
    def climb_stairs_k(n, k):
        """n stairs, can take 1 to k steps"""
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, min(k, i) + 1):
                dp[i] += dp[i - j]
        
        return dp[n]
    ```
    
    **Pattern:** This is Fibonacci! f(n) = f(n-1) + f(n-2)

    !!! tip "Interviewer's Insight"
        **What they're testing:** DP fundamentals, space optimization.
        
        **Strong answer signals:**
        
        - Recognizes Fibonacci pattern
        - Optimizes to O(1) space
        - Generalizes to K steps
        - Discusses matrix exponentiation for O(log n)

---

### Binary Tree Level Order Traversal (BFS) - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Tree`, `BFS`, `Queue` | **Asked by:** Amazon, Google, Microsoft

??? success "View Answer"

    **BFS with Queue:**
    
    ```python
    from collections import deque
    
    def level_order(root):
        """Returns list of lists, each list is a level"""
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level)
        
        return result
    
    # Example output: [[3], [9, 20], [15, 7]]
    ```
    
    **DFS Alternative:**
    
    ```python
    def level_order_dfs(root):
        result = []
        
        def dfs(node, level):
            if not node:
                return
            
            if len(result) == level:
                result.append([])
            
            result[level].append(node.val)
            dfs(node.left, level + 1)
            dfs(node.right, level + 1)
        
        dfs(root, 0)
        return result
    ```
    
    **Variants:**
    
    - Zigzag: Alternate direction per level
    - Bottom-up: Reverse result
    - Right side view: Last element per level

    !!! tip "Interviewer's Insight"
        **What they're testing:** BFS implementation, tree traversal.
        
        **Strong answer signals:**
        
        - Uses deque for O(1) popleft
        - Correctly groups by level
        - Handles empty tree
        - Can modify for variants (zigzag, right view)

---

### Validate Binary Search Tree - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Tree`, `BST`, `Recursion`, `DFS` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Recursive with Bounds:**
    
    ```python
    def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
        """Check if tree is valid BST"""
        if not root:
            return True
        
        if not (min_val < root.val < max_val):
            return False
        
        return (is_valid_bst(root.left, min_val, root.val) and
                is_valid_bst(root.right, root.val, max_val))
    ```
    
    **Inorder Traversal (should be sorted):**
    
    ```python
    def is_valid_bst_inorder(root):
        """BST inorder traversal is sorted"""
        prev = float('-inf')
        
        def inorder(node):
            nonlocal prev
            if not node:
                return True
            
            if not inorder(node.left):
                return False
            
            if node.val <= prev:
                return False
            prev = node.val
            
            return inorder(node.right)
        
        return inorder(root)
    ```
    
    **Iterative Inorder:**
    
    ```python
    def is_valid_bst_iterative(root):
        stack = []
        prev = float('-inf')
        
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            
            root = stack.pop()
            if root.val <= prev:
                return False
            prev = root.val
            root = root.right
        
        return True
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** BST property, recursive thinking.
        
        **Strong answer signals:**
        
        - Uses bounds, not just left < root < right
        - Handles equal values correctly (typically invalid)
        - Knows inorder of BST is sorted
        - Can do iterative version

---

### Lowest Common Ancestor of Binary Tree - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Tree`, `Recursion`, `DFS` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Recursive Solution:**
    
    ```python
    def lowest_common_ancestor(root, p, q):
        """Find LCA of nodes p and q"""
        if not root or root == p or root == q:
            return root
        
        left = lowest_common_ancestor(root.left, p, q)
        right = lowest_common_ancestor(root.right, p, q)
        
        if left and right:
            return root  # p and q in different subtrees
        
        return left if left else right
    ```
    
    **For BST (optimized):**
    
    ```python
    def lca_bst(root, p, q):
        """LCA for Binary Search Tree"""
        while root:
            if p.val < root.val and q.val < root.val:
                root = root.left
            elif p.val > root.val and q.val > root.val:
                root = root.right
            else:
                return root
        return None
    ```
    
    **With Parent Pointers:**
    
    ```python
    def lca_with_parent(p, q):
        """When nodes have parent pointers"""
        ancestors = set()
        
        while p:
            ancestors.add(p)
            p = p.parent
        
        while q:
            if q in ancestors:
                return q
            q = q.parent
        
        return None
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Recursive tree thinking.
        
        **Strong answer signals:**
        
        - Explains the logic clearly
        - Optimizes for BST if given
        - Handles when p or q is root
        - Knows O(n) time, O(h) space

---

### Top K Frequent Elements - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Heap`, `Hash Table`, `Bucket Sort` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Using Heap O(n log k):**
    
    ```python
    from collections import Counter
    import heapq
    
    def top_k_frequent(nums, k):
        """Return k most frequent elements"""
        count = Counter(nums)
        
        # Use min heap of size k
        return heapq.nlargest(k, count.keys(), key=count.get)
    ```
    
    **Bucket Sort O(n):**
    
    ```python
    def top_k_frequent_bucket(nums, k):
        """O(n) using bucket sort"""
        count = Counter(nums)
        
        # Buckets indexed by frequency
        buckets = [[] for _ in range(len(nums) + 1)]
        for num, freq in count.items():
            buckets[freq].append(num)
        
        result = []
        for i in range(len(buckets) - 1, -1, -1):
            for num in buckets[i]:
                result.append(num)
                if len(result) == k:
                    return result
        
        return result
    ```
    
    **Quickselect O(n) average:**
    
    ```python
    def top_k_quickselect(nums, k):
        count = Counter(nums)
        unique = list(count.keys())
        
        def partition(left, right, pivot_idx):
            pivot_freq = count[unique[pivot_idx]]
            unique[pivot_idx], unique[right] = unique[right], unique[pivot_idx]
            store_idx = left
            
            for i in range(left, right):
                if count[unique[i]] < pivot_freq:
                    unique[store_idx], unique[i] = unique[i], unique[store_idx]
                    store_idx += 1
            
            unique[right], unique[store_idx] = unique[store_idx], unique[right]
            return store_idx
        
        # ... quickselect implementation
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Heap usage, optimization thinking.
        
        **Strong answer signals:**
        
        - Uses min-heap of size k (not max-heap of size n)
        - Knows bucket sort for O(n)
        - Counter for frequency counting
        - Discusses trade-offs of each approach

---

### Course Schedule - Topological Sort - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Graph`, `Topological Sort`, `DFS`, `BFS` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Problem:** Can finish all courses given prerequisites?
    
    **DFS - Cycle Detection:**
    
    ```python
    def can_finish(num_courses, prerequisites):
        """Return True if possible to finish all courses"""
        graph = {i: [] for i in range(num_courses)}
        for course, prereq in prerequisites:
            graph[prereq].append(course)
        
        # 0: unvisited, 1: visiting, 2: visited
        state = [0] * num_courses
        
        def has_cycle(node):
            if state[node] == 1:  # Currently visiting = cycle
                return True
            if state[node] == 2:  # Already processed
                return False
            
            state[node] = 1  # Mark as visiting
            
            for neighbor in graph[node]:
                if has_cycle(neighbor):
                    return True
            
            state[node] = 2  # Mark as visited
            return False
        
        for course in range(num_courses):
            if has_cycle(course):
                return False
        
        return True
    ```
    
    **BFS - Kahn's Algorithm:**
    
    ```python
    from collections import deque
    
    def can_finish_bfs(num_courses, prerequisites):
        """Kahn's algorithm for topological sort"""
        graph = {i: [] for i in range(num_courses)}
        in_degree = [0] * num_courses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
        completed = 0
        
        while queue:
            course = queue.popleft()
            completed += 1
            
            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        return completed == num_courses
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Graph representation, cycle detection.
        
        **Strong answer signals:**
        
        - Recognizes as cycle detection problem
        - Uses 3-state DFS coloring
        - Knows Kahn's algorithm alternative
        - Can extend to Course Schedule II (return order)

---

### Word Break - Dynamic Programming - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `DP`, `String`, `Memoization` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **DP Solution:**
    
    ```python
    def word_break(s, word_dict):
        """Can s be segmented into dictionary words?"""
        word_set = set(word_dict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True  # Empty string
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[n]
    
    # Example
    s = "leetcode"
    word_dict = ["leet", "code"]
    print(word_break(s, word_dict))  # True
    ```
    
    **With Memoization:**
    
    ```python
    from functools import lru_cache
    
    def word_break_memo(s, word_dict):
        word_set = set(word_dict)
        
        @lru_cache(maxsize=None)
        def can_break(start):
            if start == len(s):
                return True
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_set and can_break(end):
                    return True
            
            return False
        
        return can_break(0)
    ```
    
    **Optimized with Trie:**
    
    For very long strings with many dictionary words, use Trie for O(1) prefix lookup.

    !!! tip "Interviewer's Insight"
        **What they're testing:** DP formulation, string segmentation.
        
        **Strong answer signals:**
        
        - Uses set for O(1) word lookup
        - Explains dp[i] = can segment s[:i]
        - Knows Word Break II (return all ways) is harder
        - Mentions Trie optimization for large dictionaries

---

### Trapping Rain Water - Two Pointers - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Array`, `Two Pointers`, `Stack`, `DP` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Two Pointers O(n) time, O(1) space:**
    
    ```python
    def trap(height):
        """Calculate trapped rain water"""
        if not height:
            return 0
        
        left, right = 0, len(height) - 1
        left_max = right_max = 0
        water = 0
        
        while left < right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water += right_max - height[right]
                right -= 1
        
        return water
    
    # Example
    height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    print(trap(height))  # 6
    ```
    
    **DP Approach O(n) space:**
    
    ```python
    def trap_dp(height):
        n = len(height)
        if n == 0:
            return 0
        
        left_max = [0] * n
        right_max = [0] * n
        
        left_max[0] = height[0]
        for i in range(1, n):
            left_max[i] = max(left_max[i-1], height[i])
        
        right_max[n-1] = height[n-1]
        for i in range(n-2, -1, -1):
            right_max[i] = max(right_max[i+1], height[i])
        
        water = 0
        for i in range(n):
            water += min(left_max[i], right_max[i]) - height[i]
        
        return water
    ```
    
    **Key Insight:** Water at position i = min(left_max, right_max) - height[i]

    !!! tip "Interviewer's Insight"
        **What they're testing:** Array manipulation, space optimization.
        
        **Strong answer signals:**
        
        - Explains water level logic
        - Optimizes from O(n) to O(1) space
        - Handles edge cases (empty, no trap possible)
        - Can also solve with monotonic stack

---

### Merge K Sorted Lists - Heap - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Heap`, `Linked List`, `Divide & Conquer` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Using Min Heap:**
    
    ```python
    import heapq
    
    def merge_k_lists(lists):
        """Merge k sorted linked lists"""
        # Min heap: (value, index, node)
        heap = []
        
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst.val, i, lst))
        
        dummy = ListNode(0)
        current = dummy
        
        while heap:
            val, idx, node = heapq.heappop(heap)
            current.next = node
            current = current.next
            
            if node.next:
                heapq.heappush(heap, (node.next.val, idx, node.next))
        
        return dummy.next
    ```
    
    **Divide and Conquer:**
    
    ```python
    def merge_k_lists_dc(lists):
        """Merge using divide and conquer"""
        if not lists:
            return None
        
        while len(lists) > 1:
            merged = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if i + 1 < len(lists) else None
                merged.append(merge_two_lists(l1, l2))
            lists = merged
        
        return lists[0]
    ```
    
    **Complexity:**
    
    | Approach | Time | Space |
    |----------|------|-------|
    | Heap | O(n log k) | O(k) |
    | D&C | O(n log k) | O(log k) |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Heap usage, efficiency for k lists.
        
        **Strong answer signals:**
        
        - Uses index to handle equal values in heap
        - Knows both heap and D&C approaches
        - Handles empty lists
        - Explains O(n log k) vs O(nk) naive

---

### Find Median from Data Stream - Heap - Meta, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Heap`, `Design`, `Two Heaps` | **Asked by:** Meta, Amazon, Google

??? success "View Answer"

    **Two Heaps Solution:**
    
    ```python
    import heapq
    
    class MedianFinder:
        def __init__(self):
            self.small = []  # Max heap (negate values)
            self.large = []  # Min heap
        
        def addNum(self, num):
            # Add to max heap (small)
            heapq.heappush(self.small, -num)
            
            # Balance: largest of small <= smallest of large
            if self.small and self.large and (-self.small[0] > self.large[0]):
                val = -heapq.heappop(self.small)
                heapq.heappush(self.large, val)
            
            # Balance sizes (small can have at most 1 more)
            if len(self.small) > len(self.large) + 1:
                val = -heapq.heappop(self.small)
                heapq.heappush(self.large, val)
            
            if len(self.large) > len(self.small):
                val = heapq.heappop(self.large)
                heapq.heappush(self.small, -val)
        
        def findMedian(self):
            if len(self.small) > len(self.large):
                return -self.small[0]
            return (-self.small[0] + self.large[0]) / 2
    ```
    
    **Complexity:**
    
    - addNum: O(log n)
    - findMedian: O(1)
    - Space: O(n)
    
    **Key Insight:**
    
    - Max heap holds smaller half
    - Min heap holds larger half
    - Median is at the tops

    !!! tip "Interviewer's Insight"
        **What they're testing:** Two-heap pattern, design.
        
        **Strong answer signals:**
        
        - Uses negate for max heap in Python
        - Maintains size balance invariant
        - Handles odd/even count cases
        - Can discuss follow-ups (sliding window median)

---

### Number of Islands - DFS/BFS Graph - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Graph`, `DFS`, `BFS`, `Matrix` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **DFS Solution:**
    
    ```python
    def num_islands(grid):
        """Count number of islands in grid"""
        if not grid:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        count = 0
        
        def dfs(r, c):
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
                return
            
            grid[r][c] = '0'  # Mark as visited
            
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    count += 1
                    dfs(r, c)
        
        return count
    ```
    
    **BFS Solution:**
    
    ```python
    from collections import deque
    
    def num_islands_bfs(grid):
        rows, cols = len(grid), len(grid[0])
        count = 0
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    count += 1
                    queue = deque([(r, c)])
                    grid[r][c] = '0'
                    
                    while queue:
                        row, col = queue.popleft()
                        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                                queue.append((nr, nc))
                                grid[nr][nc] = '0'
        
        return count
    ```
    
    **Union-Find Alternative:**
    
    For follow-ups like "Number of Islands II" (dynamic additions), use Union-Find.

    !!! tip "Interviewer's Insight"
        **What they're testing:** Graph traversal on grid.
        
        **Strong answer signals:**
        
        - Marks visited in-place or uses set
        - Uses direction array for cleaner code
        - Handles empty grid
        - Knows Union-Find for dynamic version

---

### Serialize and Deserialize Binary Tree - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Tree`, `DFS`, `BFS`, `String` | **Asked by:** Amazon, Google, Microsoft

??? success "View Answer"

    **Preorder DFS Solution:**
    
    ```python
    class Codec:
        def serialize(self, root):
            """Encodes tree to string"""
            result = []
            
            def dfs(node):
                if not node:
                    result.append('N')
                    return
                result.append(str(node.val))
                dfs(node.left)
                dfs(node.right)
            
            dfs(root)
            return ','.join(result)
        
        def deserialize(self, data):
            """Decodes string to tree"""
            values = iter(data.split(','))
            
            def dfs():
                val = next(values)
                if val == 'N':
                    return None
                
                node = TreeNode(int(val))
                node.left = dfs()
                node.right = dfs()
                return node
            
            return dfs()
    ```
    
    **BFS Solution:**
    
    ```python
    from collections import deque
    
    class Codec:
        def serialize(self, root):
            if not root:
                return 'N'
            
            result = []
            queue = deque([root])
            
            while queue:
                node = queue.popleft()
                if node:
                    result.append(str(node.val))
                    queue.append(node.left)
                    queue.append(node.right)
                else:
                    result.append('N')
            
            return ','.join(result)
        
        def deserialize(self, data):
            if data == 'N':
                return None
            
            values = data.split(',')
            root = TreeNode(int(values[0]))
            queue = deque([root])
            i = 1
            
            while queue:
                node = queue.popleft()
                if values[i] != 'N':
                    node.left = TreeNode(int(values[i]))
                    queue.append(node.left)
                i += 1
                
                if values[i] != 'N':
                    node.right = TreeNode(int(values[i]))
                    queue.append(node.right)
                i += 1
            
            return root
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Tree representation, encoding.
        
        **Strong answer signals:**
        
        - Uses 'N' for null markers
        - Preorder is simpler for DFS
        - BFS gives level-by-level format
        - Handles negative numbers and large values

---

### Dijkstra's Algorithm - Shortest Path - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Graph`, `Heap`, `Shortest Path`, `Greedy` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Implementation with Min Heap:**
    
    ```python
    import heapq
    from collections import defaultdict
    
    def dijkstra(n, edges, start):
        """Find shortest paths from start to all nodes"""
        graph = defaultdict(list)
        for u, v, weight in edges:
            graph[u].append((v, weight))
            graph[v].append((u, weight))  # For undirected
        
        distances = {i: float('inf') for i in range(n)}
        distances[start] = 0
        
        # Min heap: (distance, node)
        heap = [(0, start)]
        
        while heap:
            dist, node = heapq.heappop(heap)
            
            if dist > distances[node]:
                continue  # Already found shorter path
            
            for neighbor, weight in graph[node]:
                new_dist = dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))
        
        return distances
    ```
    
    **Network Delay Time (LeetCode 743):**
    
    ```python
    def network_delay_time(times, n, k):
        """Time for signal to reach all nodes"""
        distances = dijkstra(n, times, k - 1)  # 1-indexed
        max_time = max(distances.values())
        return max_time if max_time < float('inf') else -1
    ```
    
    **Complexity:**
    
    - Time: O((V + E) log V) with binary heap
    - Space: O(V + E)

    !!! tip "Interviewer's Insight"
        **What they're testing:** Graph algorithms, heap usage.
        
        **Strong answer signals:**
        
        - Uses min-heap for efficiency
        - Skips stale entries correctly
        - Knows doesn't work with negative weights
        - Can compare to Bellman-Ford, A*

---

### Implement Trie (Prefix Tree) - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Trie`, `Design`, `String` | **Asked by:** Amazon, Google, Microsoft

??? success "View Answer"

    **Implementation:**
    
    ```python
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False
    
    class Trie:
        def __init__(self):
            self.root = TrieNode()
        
        def insert(self, word):
            """Insert word into trie - O(m)"""
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
        
        def search(self, word):
            """Check if word exists - O(m)"""
            node = self._find_node(word)
            return node is not None and node.is_end
        
        def startsWith(self, prefix):
            """Check if prefix exists - O(m)"""
            return self._find_node(prefix) is not None
        
        def _find_node(self, prefix):
            node = self.root
            for char in prefix:
                if char not in node.children:
                    return None
                node = node.children[char]
            return node
    ```
    
    **With Autocomplete:**
    
    ```python
    def autocomplete(self, prefix):
        """Return all words with given prefix"""
        node = self._find_node(prefix)
        if not node:
            return []
        
        results = []
        
        def dfs(node, path):
            if node.is_end:
                results.append(path)
            for char, child in node.children.items():
                dfs(child, path + char)
        
        dfs(node, prefix)
        return results
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Trie data structure, prefix operations.
        
        **Strong answer signals:**
        
        - Uses dictionary for flexible children
        - Distinguishes word end from prefix
        - Knows space-time trade-offs
        - Can extend for wildcard search

---

### Longest Increasing Subsequence - Dynamic Programming - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `DP`, `Binary Search`, `Array` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **DP Solution O(n²):**
    
    ```python
    def length_of_lis(nums):
        """Length of longest increasing subsequence"""
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n  # dp[i] = LIS ending at i
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    # Example
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    print(length_of_lis(nums))  # 4 (sequence: 2, 3, 7, 101)
    ```
    
    **Binary Search O(n log n):**
    
    ```python
    import bisect
    
    def length_of_lis_bs(nums):
        """O(n log n) using patience sort"""
        tails = []  # tails[i] = smallest tail of LIS of length i+1
        
        for num in nums:
            pos = bisect.bisect_left(tails, num)
            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num
        
        return len(tails)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** DP optimization, binary search.
        
        **Strong answer signals:**
        
        - Starts with O(n²), then optimizes
        - Explains tails array invariant
        - Can reconstruct actual subsequence
        - Knows related: Longest Common Subsequence

---

### Coin Change - Minimum Coins DP - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `DP`, `Unbounded Knapsack` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Bottom-Up DP:**
    
    ```python
    def coin_change(coins, amount):
        """Minimum coins to make amount"""
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    # Example
    coins = [1, 2, 5]
    amount = 11
    print(coin_change(coins, amount))  # 3 (5 + 5 + 1)
    ```
    
    **Top-Down with Memoization:**
    
    ```python
    from functools import lru_cache
    
    def coin_change_memo(coins, amount):
        @lru_cache(maxsize=None)
        def dp(remaining):
            if remaining == 0:
                return 0
            if remaining < 0:
                return float('inf')
            
            min_coins = float('inf')
            for coin in coins:
                min_coins = min(min_coins, dp(remaining - coin) + 1)
            return min_coins
        
        result = dp(amount)
        return result if result != float('inf') else -1
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** DP state definition, unbounded knapsack.
        
        **Strong answer signals:**
        
        - Identifies as unbounded knapsack variant
        - Uses infinity for impossible states
        - Can count number of ways (Coin Change 2)
        - O(amount × coins) time

---

### Subsets - Backtracking - Amazon, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Backtracking`, `Recursion`, `Bit Manipulation` | **Asked by:** Amazon, Meta, Google

??? success "View Answer"

    **Backtracking:**
    
    ```python
    def subsets(nums):
        """Generate all subsets (power set)"""
        result = []
        
        def backtrack(start, current):
            result.append(current[:])  # Add copy
            
            for i in range(start, len(nums)):
                current.append(nums[i])
                backtrack(i + 1, current)
                current.pop()  # Backtrack
        
        backtrack(0, [])
        return result
    
    # Example
    nums = [1, 2, 3]
    print(subsets(nums))
    # [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
    ```
    
    **Iterative (Cascading):**
    
    ```python
    def subsets_iterative(nums):
        result = [[]]
        
        for num in nums:
            result += [subset + [num] for subset in result]
        
        return result
    ```
    
    **Bit Manipulation:**
    
    ```python
    def subsets_bits(nums):
        n = len(nums)
        result = []
        
        for mask in range(1 << n):
            subset = [nums[i] for i in range(n) if mask & (1 << i)]
            result.append(subset)
        
        return result
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Backtracking template, enumeration.
        
        **Strong answer signals:**
        
        - Uses append/pop pattern for backtracking
        - Knows bit manipulation alternative
        - Can modify for Subsets II (with duplicates)
        - Understands 2^n subsets exist

---

### Permutations - Backtracking - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Backtracking`, `Recursion` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Backtracking:**
    
    ```python
    def permute(nums):
        """Generate all permutations"""
        result = []
        
        def backtrack(current, remaining):
            if not remaining:
                result.append(current[:])
                return
            
            for i in range(len(remaining)):
                current.append(remaining[i])
                backtrack(current, remaining[:i] + remaining[i+1:])
                current.pop()
        
        backtrack([], nums)
        return result
    ```
    
    **In-Place Swap:**
    
    ```python
    def permute_swap(nums):
        result = []
        
        def backtrack(start):
            if start == len(nums):
                result.append(nums[:])
                return
            
            for i in range(start, len(nums)):
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)
                nums[start], nums[i] = nums[i], nums[start]  # Backtrack
        
        backtrack(0)
        return result
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Backtracking, state management.
        
        **Strong answer signals:**
        
        - Knows swap-based optimization
        - Can handle duplicates (Permutations II)
        - Understands n! permutations
        - Uses visited set alternative

---

### Combination Sum - Backtracking - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Backtracking`, `DFS` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Backtracking Solution:**
    
    ```python
    def combination_sum(candidates, target):
        """Find combinations summing to target (can reuse)"""
        result = []
        
        def backtrack(start, current, remaining):
            if remaining == 0:
                result.append(current[:])
                return
            if remaining < 0:
                return
            
            for i in range(start, len(candidates)):
                current.append(candidates[i])
                backtrack(i, current, remaining - candidates[i])  # i not i+1
                current.pop()
        
        backtrack(0, [], target)
        return result
    
    # Example
    candidates = [2, 3, 6, 7]
    target = 7
    print(combination_sum(candidates, target))
    # [[2, 2, 3], [7]]
    ```
    
    **Variants:**
    
    | Problem | Key Difference |
    |---------|---------------|
    | Combination Sum I | Unlimited use |
    | Combination Sum II | Use once, has duplicates |
    | Combination Sum III | K numbers, 1-9 only |
    | Combination Sum IV | Count permutations (DP) |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Backtracking for combinations.
        
        **Strong answer signals:**
        
        - Uses start index to avoid duplicates
        - Knows when to use i vs i+1
        - Prunes early with remaining < 0
        - Can adapt for variations

---

### Rotate Array - In-Place - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Array`, `In-Place`, `Reverse` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Reverse Three Times O(1) space:**
    
    ```python
    def rotate(nums, k):
        """Rotate array right by k steps - in-place"""
        n = len(nums)
        k = k % n  # Handle k > n
        
        def reverse(start, end):
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1
        
        reverse(0, n - 1)      # Reverse all
        reverse(0, k - 1)      # Reverse first k
        reverse(k, n - 1)      # Reverse rest
    
    # Example
    nums = [1, 2, 3, 4, 5, 6, 7]
    rotate(nums, 3)
    print(nums)  # [5, 6, 7, 1, 2, 3, 4]
    ```
    
    **Cyclic Replacements:**
    
    ```python
    def rotate_cyclic(nums, k):
        n = len(nums)
        k = k % n
        count = 0
        start = 0
        
        while count < n:
            current = start
            prev = nums[start]
            
            while True:
                next_idx = (current + k) % n
                nums[next_idx], prev = prev, nums[next_idx]
                current = next_idx
                count += 1
                
                if start == current:
                    break
            
            start += 1
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** In-place array manipulation.
        
        **Strong answer signals:**
        
        - Uses k % n for edge cases
        - Explains reverse approach intuition
        - O(1) space, O(n) time
        - Knows left rotation variant

---

### Product of Array Except Self - Prefix/Suffix - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Array`, `Prefix`, `Suffix` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **O(1) Extra Space Solution:**
    
    ```python
    def product_except_self(nums):
        """Product of all elements except self, no division"""
        n = len(nums)
        result = [1] * n
        
        # Left pass: prefix products
        prefix = 1
        for i in range(n):
            result[i] = prefix
            prefix *= nums[i]
        
        # Right pass: multiply by suffix products
        suffix = 1
        for i in range(n - 1, -1, -1):
            result[i] *= suffix
            suffix *= nums[i]
        
        return result
    
    # Example
    nums = [1, 2, 3, 4]
    print(product_except_self(nums))  # [24, 12, 8, 6]
    ```
    
    **Key Insight:**
    
    ```
    result[i] = (product of all left) × (product of all right)
    
    For [1, 2, 3, 4]:
    - Prefix: [1, 1, 2, 6]
    - Suffix: [24, 12, 4, 1]
    - Result: [24, 12, 8, 6]
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Prefix/suffix pattern, no division.
        
        **Strong answer signals:**
        
        - Uses output array for O(1) extra space
        - Handles zeros correctly
        - No division allowed
        - Two-pass approach

---

### Group Anagrams - Hash Map - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Hash Table`, `String`, `Sorting` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Using Sorted String as Key:**
    
    ```python
    from collections import defaultdict
    
    def group_anagrams(strs):
        """Group strings that are anagrams"""
        groups = defaultdict(list)
        
        for s in strs:
            key = ''.join(sorted(s))
            groups[key].append(s)
        
        return list(groups.values())
    
    # Example
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    print(group_anagrams(strs))
    # [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
    ```
    
    **Using Character Count as Key:**
    
    ```python
    def group_anagrams_count(strs):
        """O(n × k) instead of O(n × k log k)"""
        groups = defaultdict(list)
        
        for s in strs:
            count = [0] * 26
            for char in s:
                count[ord(char) - ord('a')] += 1
            groups[tuple(count)].append(s)
        
        return list(groups.values())
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Hash map key design.
        
        **Strong answer signals:**
        
        - Uses sorted string as key
        - Knows count-based key is O(k) vs O(k log k)
        - Uses defaultdict for cleaner code
        - Understands tuple for hashable key

---

### Linked List Cycle Detection - Floyd's Algorithm - Amazon, Meta Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Linked List`, `Two Pointers`, `Floyd` | **Asked by:** Amazon, Meta, Google

??? success "View Answer"

    **Floyd's Tortoise and Hare:**
    
    ```python
    def has_cycle(head):
        """Detect cycle in linked list - O(1) space"""
        if not head or not head.next:
            return False
        
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
        
        return False
    ```
    
    **Find Cycle Start:**
    
    ```python
    def detect_cycle(head):
        """Return node where cycle begins"""
        if not head or not head.next:
            return None
        
        slow = fast = head
        
        # Detect cycle
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            return None
        
        # Find start: both move one step
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        return slow
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Two-pointer technique, Floyd's algorithm.
        
        **Strong answer signals:**
        
        - Explains why fast moves 2, slow moves 1
        - Knows math behind finding cycle start
        - O(1) space, O(n) time
        - Can find cycle length

---

### Merge Intervals - Sorting - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Array`, `Sorting`, `Intervals` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Sort and Merge:**
    
    ```python
    def merge(intervals):
        """Merge overlapping intervals"""
        if not intervals:
            return []
        
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:  # Overlapping
                last[1] = max(last[1], current[1])
            else:
                merged.append(current)
        
        return merged
    
    # Example
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    print(merge(intervals))  # [[1, 6], [8, 10], [15, 18]]
    ```
    
    **Related Problems:**
    
    | Problem | Approach |
    |---------|----------|
    | Insert Interval | Binary search or linear merge |
    | Non-overlapping Intervals | Greedy, count removals |
    | Meeting Rooms | Sort by start, check overlap |
    | Meeting Rooms II | Sort endpoints, track active |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Interval processing, sorting.
        
        **Strong answer signals:**
        
        - Sorts by start time
        - Uses `<=` for overlapping check
        - Modifies in-place when possible
        - Knows interval problems family

---

### Search in Rotated Sorted Array - Binary Search - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Binary Search`, `Array` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Modified Binary Search:**
    
    ```python
    def search(nums, target):
        """Search in rotated sorted array"""
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            
            # Left half is sorted
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # Right half is sorted
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1
    
    # Example
    nums = [4, 5, 6, 7, 0, 1, 2]
    print(search(nums, 0))  # 4
    ```
    
    **Find Minimum (Pivot):**
    
    ```python
    def find_min(nums):
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        
        return nums[left]
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Binary search variants.
        
        **Strong answer signals:**
        
        - Identifies which half is sorted
        - Uses correct inequality for bounds
        - Handles duplicates (Search II)
        - O(log n) time

---

### Min Stack - Design - Meta, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Stack`, `Design` | **Asked by:** Meta, Amazon, Google

??? success "View Answer"

    **Two Stack Approach:**
    
    ```python
    class MinStack:
        def __init__(self):
            self.stack = []
            self.min_stack = []
        
        def push(self, val):
            self.stack.append(val)
            min_val = min(val, self.min_stack[-1] if self.min_stack else val)
            self.min_stack.append(min_val)
        
        def pop(self):
            self.stack.pop()
            self.min_stack.pop()
        
        def top(self):
            return self.stack[-1]
        
        def getMin(self):
            return self.min_stack[-1]
    ```
    
    **Single Stack with Encoding:**
    
    ```python
    class MinStackSingle:
        def __init__(self):
            self.stack = []
            self.min_val = float('inf')
        
        def push(self, val):
            if val <= self.min_val:
                self.stack.append(self.min_val)  # Save old min
                self.min_val = val
            self.stack.append(val)
        
        def pop(self):
            if self.stack.pop() == self.min_val:
                self.min_val = self.stack.pop()
        
        def top(self):
            return self.stack[-1]
        
        def getMin(self):
            return self.min_val
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Stack design, optimization.
        
        **Strong answer signals:**
        
        - All operations O(1)
        - Knows two-stack vs single-stack trade-offs
        - Can extend to MaxStack
        - Handles edge cases

---

### Valid Sudoku - Matrix Validation - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Matrix`, `Hash Set`, `Validation` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Using Sets:**
    
    ```python
    def is_valid_sudoku(board):
        """Check if 9x9 board is valid"""
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        
        for r in range(9):
            for c in range(9):
                val = board[r][c]
                if val == '.':
                    continue
                
                box_idx = (r // 3) * 3 + (c // 3)
                
                if val in rows[r] or val in cols[c] or val in boxes[box_idx]:
                    return False
                
                rows[r].add(val)
                cols[c].add(val)
                boxes[box_idx].add(val)
        
        return True
    ```
    
    **Box Index Formula:**
    
    ```
    For position (r, c):
    box_row = r // 3
    box_col = c // 3
    box_idx = box_row * 3 + box_col
    
    Grid of boxes:
    0 1 2
    3 4 5
    6 7 8
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Matrix indexing, validation logic.
        
        **Strong answer signals:**
        
        - Uses sets for O(1) lookup
        - Knows box index formula
        - Single pass O(81) = O(1)
        - Can extend to solve Sudoku

---

### House Robber - Dynamic Programming - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `DP`, `Array` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **O(1) Space DP:**
    
    ```python
    def rob(nums):
        """Max amount without robbing adjacent houses"""
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        prev2, prev1 = 0, 0
        
        for num in nums:
            current = max(prev1, prev2 + num)
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    # Example
    nums = [2, 7, 9, 3, 1]
    print(rob(nums))  # 12 (rob houses 1, 3, 5: 2 + 9 + 1)
    ```
    
    **House Robber II (Circular):**
    
    ```python
    def rob_circular(nums):
        """Houses in circle - can't rob first and last"""
        if len(nums) == 1:
            return nums[0]
        
        return max(rob(nums[:-1]), rob(nums[1:]))
    ```
    
    **Recurrence:** dp[i] = max(dp[i-1], dp[i-2] + nums[i])

    !!! tip "Interviewer's Insight"
        **What they're testing:** 1D DP, space optimization.
        
        **Strong answer signals:**
        
        - Optimizes from O(n) to O(1) space
        - Handles circular case
        - Knows House Robber III (binary tree)
        - Clear recurrence explanation

---

### 3Sum - Three Pointers - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Array`, `Two Pointers`, `Sorting` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Two Pointers After Sorting:**
    
    ```python
    def three_sum(nums):
        """Find all unique triplets that sum to zero"""
        nums.sort()
        result = []
        
        for i in range(len(nums) - 2):
            # Skip duplicates for i
            if i > 0 and nums[i] == nums[i-1]:
                continue
            
            left, right = i + 1, len(nums) - 1
            
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                
                if total < 0:
                    left += 1
                elif total > 0:
                    right -= 1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    
                    # Skip duplicates
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
        
        return result
    ```
    
    **Complexity:** O(n²) time, O(1) extra space (ignoring output)

    !!! tip "Interviewer's Insight"
        **What they're testing:** Two-pointer technique, duplicate handling.
        
        **Strong answer signals:**
        
        - Sorts first for two-pointer approach
        - Handles duplicates correctly
        - Can extend to 4Sum, kSum
        - Knows hash-based alternative

---

### Longest Common Subsequence - 2D DP - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `DP`, `String`, `2D` | **Asked by:** Amazon, Google, Microsoft

??? success "View Answer"

    **2D DP Solution:**
    
    ```python
    def longest_common_subsequence(text1, text2):
        """Length of LCS"""
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    # Example
    print(longest_common_subsequence("abcde", "ace"))  # 3 ("ace")
    ```
    
    **Space Optimized O(min(m,n)):**
    
    ```python
    def lcs_optimized(text1, text2):
        if len(text2) > len(text1):
            text1, text2 = text2, text1
        
        prev = [0] * (len(text2) + 1)
        
        for i in range(1, len(text1) + 1):
            curr = [0] * (len(text2) + 1)
            for j in range(1, len(text2) + 1):
                if text1[i-1] == text2[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev = curr
        
        return prev[-1]
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Classic 2D DP, string algorithms.
        
        **Strong answer signals:**
        
        - Explains recurrence clearly
        - Can reconstruct actual LCS
        - Knows space optimization
        - Related: Edit Distance, LIS

---

### Find Kth Largest Element - Quickselect - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Heap`, `Quickselect`, `Sorting` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Using Heap O(n log k):**
    
    ```python
    import heapq
    
    def find_kth_largest(nums, k):
        """Kth largest using min-heap of size k"""
        return heapq.nlargest(k, nums)[-1]
    ```
    
    **Quickselect O(n) average:**
    
    ```python
    import random
    
    def find_kth_largest_qs(nums, k):
        """Average O(n), worst O(n²)"""
        k = len(nums) - k  # Convert to kth smallest
        
        def quickselect(left, right):
            pivot_idx = random.randint(left, right)
            pivot = nums[pivot_idx]
            
            # Move pivot to end
            nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
            store_idx = left
            
            for i in range(left, right):
                if nums[i] < pivot:
                    nums[store_idx], nums[i] = nums[i], nums[store_idx]
                    store_idx += 1
            
            nums[store_idx], nums[right] = nums[right], nums[store_idx]
            
            if store_idx == k:
                return nums[k]
            elif store_idx < k:
                return quickselect(store_idx + 1, right)
            else:
                return quickselect(left, store_idx - 1)
        
        return quickselect(0, len(nums) - 1)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Selection algorithms, heap usage.
        
        **Strong answer signals:**
        
        - Uses min-heap of size k
        - Knows Quickselect for O(n) average
        - Random pivot for better performance
        - Explains kth largest vs kth smallest

---

### Edit Distance (Levenshtein) - 2D DP - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `DP`, `String`, `2D` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **2D DP Solution:**
    
    ```python
    def min_distance(word1, word2):
        """Min operations to convert word1 to word2"""
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base cases
        for i in range(m + 1):
            dp[i][0] = i  # Delete all
        for j in range(n + 1):
            dp[0][j] = j  # Insert all
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No operation
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Delete
                        dp[i][j-1],    # Insert
                        dp[i-1][j-1]   # Replace
                    )
        
        return dp[m][n]
    
    # Example
    print(min_distance("horse", "ros"))  # 3
    ```
    
    **Operations:**
    
    - Insert: dp[i][j-1] + 1
    - Delete: dp[i-1][j] + 1
    - Replace: dp[i-1][j-1] + 1

    !!! tip "Interviewer's Insight"
        **What they're testing:** Classic DP, string transformation.
        
        **Strong answer signals:**
        
        - Explains three operations clearly
        - Correct base case initialization
        - Can optimize to O(n) space
        - Used in spell check, DNA analysis

---

### Word Search - Backtracking - Amazon, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Backtracking`, `Matrix`, `DFS` | **Asked by:** Amazon, Meta, Google

??? success "View Answer"

    **DFS Backtracking:**
    
    ```python
    def exist(board, word):
        """Check if word exists in board (adjacent cells)"""
        rows, cols = len(board), len(board[0])
        
        def dfs(r, c, idx):
            if idx == len(word):
                return True
            
            if (r < 0 or r >= rows or c < 0 or c >= cols or
                board[r][c] != word[idx]):
                return False
            
            # Mark visited
            temp = board[r][c]
            board[r][c] = '#'
            
            # Explore neighbors
            found = (dfs(r+1, c, idx+1) or dfs(r-1, c, idx+1) or
                     dfs(r, c+1, idx+1) or dfs(r, c-1, idx+1))
            
            # Backtrack
            board[r][c] = temp
            
            return found
        
        for r in range(rows):
            for c in range(cols):
                if dfs(r, c, 0):
                    return True
        
        return False
    ```
    
    **Word Search II (Multiple Words):** Use Trie for efficiency:
    
    ```python
    # Build Trie from words
    # DFS with Trie node instead of word index
    # Prune Trie as words are found
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Matrix DFS, backtracking.
        
        **Strong answer signals:**
        
        - Marks visited in-place
        - Backtracks correctly
        - Knows Trie optimization for Word Search II
        - Discusses pruning strategies

---

### Maximum Profit in Job Scheduling - DP + Binary Search - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `DP`, `Binary Search`, `Sorting` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **DP with Binary Search:**
    
    ```python
    import bisect
    
    def job_scheduling(start_time, end_time, profit):
        """Max profit from non-overlapping jobs"""
        n = len(start_time)
        jobs = sorted(zip(end_time, start_time, profit))
        
        # dp[i] = max profit considering first i jobs
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            end, start, p = jobs[i-1]
            
            # Find latest non-overlapping job
            k = bisect.bisect_right([jobs[j][0] for j in range(i-1)], start)
            
            dp[i] = max(dp[i-1], dp[k] + p)
        
        return dp[n]
    ```
    
    **Cleaner Implementation:**
    
    ```python
    def job_scheduling_clean(start_time, end_time, profit):
        jobs = sorted(zip(end_time, start_time, profit))
        ends = [j[0] for j in jobs]
        dp = [(0, 0)]  # (end_time, max_profit)
        
        for end, start, p in jobs:
            idx = bisect.bisect_right(dp, (start, float('inf'))) - 1
            profit_if_taken = dp[idx][1] + p
            
            if profit_if_taken > dp[-1][1]:
                dp.append((end, profit_if_taken))
        
        return dp[-1][1]
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** DP with optimization, binary search.
        
        **Strong answer signals:**
        
        - Sorts by end time
        - Uses binary search for non-overlapping
        - Knows weighted interval scheduling
        - O(n log n) solution

---

### Sliding Window Maximum - Monotonic Deque - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Sliding Window`, `Deque`, `Array` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Using Monotonic Deque:**

    ```python
    from collections import deque

    def max_sliding_window(nums, k):
        """Find max in each sliding window of size k"""
        result = []
        dq = deque()  # Store indices, maintain decreasing order

        for i in range(len(nums)):
            # Remove indices outside window
            while dq and dq[0] < i - k + 1:
                dq.popleft()

            # Remove smaller elements (won't be max)
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()

            dq.append(i)

            # Add to result after first window
            if i >= k - 1:
                result.append(nums[dq[0]])

        return result

    # Example
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    print(max_sliding_window(nums, k))  # [3, 3, 5, 5, 6, 7]
    ```

    **Time:** O(n), **Space:** O(k)

    !!! tip "Interviewer's Insight"
        **What they're testing:** Monotonic queue, sliding window optimization.

        **Strong answer signals:**

        - Uses deque for O(n) solution
        - Explains why elements can be removed
        - Compares with heap approach O(n log k)
        - Handles edge cases (k=1, k=n)

---

### Union-Find with Path Compression - Meta, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Union-Find`, `Graph`, `Disjoint Set` | **Asked by:** Meta, Google, Amazon

??? success "View Answer"

    **Implementation:**

    ```python
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [1] * n
            self.components = n

        def find(self, x):
            """Find with path compression"""
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            """Union by rank"""
            root_x, root_y = self.find(x), self.find(y)

            if root_x == root_y:
                return False

            # Attach smaller tree to larger
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

            self.components -= 1
            return True

        def connected(self, x, y):
            return self.find(x) == self.find(y)

    # Example: Number of Connected Components
    def count_components(n, edges):
        uf = UnionFind(n)
        for a, b in edges:
            uf.union(a, b)
        return uf.components
    ```

    **Time:** O(α(n)) amortized per operation

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of path compression and union by rank.

        **Strong answer signals:**

        - Implements both optimizations
        - Explains inverse Ackermann complexity
        - Applies to problems (cycle detection, MST, connected components)
        - Discusses weighted union-find variants

---

### Minimum Spanning Tree - Kruskal's Algorithm - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Graph`, `MST`, `Greedy`, `Union-Find` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Kruskal's Algorithm:**

    ```python
    def kruskal_mst(n, edges):
        """Find MST using Kruskal's - O(E log E)"""
        # Sort edges by weight
        edges.sort(key=lambda x: x[2])

        uf = UnionFind(n)
        mst_edges = []
        total_weight = 0

        for u, v, weight in edges:
            if uf.union(u, v):
                mst_edges.append((u, v, weight))
                total_weight += weight

                if len(mst_edges) == n - 1:
                    break

        return total_weight, mst_edges

    # Example
    n = 4
    edges = [(0, 1, 10), (0, 2, 6), (0, 3, 5),
             (1, 3, 15), (2, 3, 4)]
    weight, mst = kruskal_mst(n, edges)
    print(f"MST weight: {weight}")  # 19
    ```

    **Prim's Algorithm (Alternative):**

    ```python
    import heapq

    def prim_mst(n, edges):
        """MST using Prim's - O(E log V)"""
        # Build adjacency list
        graph = [[] for _ in range(n)]
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))

        visited = [False] * n
        heap = [(0, 0)]  # (weight, node)
        total = 0

        while heap:
            w, u = heapq.heappop(heap)
            if visited[u]:
                continue

            visited[u] = True
            total += w

            for v, weight in graph[u]:
                if not visited[v]:
                    heapq.heappush(heap, (weight, v))

        return total
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** MST algorithms, graph theory.

        **Strong answer signals:**

        - Knows both Kruskal's and Prim's
        - Uses Union-Find for cycle detection
        - Discusses time complexity tradeoffs
        - Extends to min-cost network problems

---

### Segment Tree for Range Queries - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Segment Tree`, `Data Structures` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Segment Tree Implementation:**

    ```python
    class SegmentTree:
        def __init__(self, nums):
            n = len(nums)
            self.n = n
            self.tree = [0] * (4 * n)
            self._build(nums, 0, 0, n - 1)

        def _build(self, nums, node, start, end):
            if start == end:
                self.tree[node] = nums[start]
            else:
                mid = (start + end) // 2
                left = 2 * node + 1
                right = 2 * node + 2

                self._build(nums, left, start, mid)
                self._build(nums, right, mid + 1, end)
                self.tree[node] = self.tree[left] + self.tree[right]

        def update(self, idx, val):
            """Update element at index idx to val"""
            self._update(0, 0, self.n - 1, idx, val)

        def _update(self, node, start, end, idx, val):
            if start == end:
                self.tree[node] = val
            else:
                mid = (start + end) // 2
                left, right = 2 * node + 1, 2 * node + 2

                if idx <= mid:
                    self._update(left, start, mid, idx, val)
                else:
                    self._update(right, mid + 1, end, idx, val)

                self.tree[node] = self.tree[left] + self.tree[right]

        def query(self, l, r):
            """Query sum in range [l, r]"""
            return self._query(0, 0, self.n - 1, l, r)

        def _query(self, node, start, end, l, r):
            if r < start or l > end:
                return 0
            if l <= start and end <= r:
                return self.tree[node]

            mid = (start + end) // 2
            left_sum = self._query(2 * node + 1, start, mid, l, r)
            right_sum = self._query(2 * node + 2, mid + 1, end, l, r)
            return left_sum + right_sum

    # Example
    nums = [1, 3, 5, 7, 9, 11]
    st = SegmentTree(nums)
    print(st.query(1, 3))  # 15 (3+5+7)
    st.update(1, 10)
    print(st.query(1, 3))  # 22 (10+5+7)
    ```

    **Time:** Build O(n), Query/Update O(log n)

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced data structures, tree concepts.

        **Strong answer signals:**

        - Understands lazy propagation for range updates
        - Compares with Fenwick tree (BIT)
        - Extends to min/max queries, GCD queries
        - Discusses when to use vs simpler alternatives

---

### Trie with Prefix Matching - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Trie`, `String`, `Prefix` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Trie Implementation:**

    ```python
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False
            self.word = None  # Store full word

    class Trie:
        def __init__(self):
            self.root = TrieNode()

        def insert(self, word):
            """Insert word into trie"""
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.word = word

        def search(self, word):
            """Check if word exists"""
            node = self.root
            for char in word:
                if char not in node.children:
                    return False
                node = node.children[char]
            return node.is_end

        def starts_with(self, prefix):
            """Check if prefix exists"""
            node = self.root
            for char in prefix:
                if char not in node.children:
                    return False
                node = node.children[char]
            return True

        def find_words_with_prefix(self, prefix):
            """Find all words starting with prefix"""
            node = self.root
            for char in prefix:
                if char not in node.children:
                    return []
                node = node.children[char]

            result = []
            self._dfs(node, result)
            return result

        def _dfs(self, node, result):
            if node.is_end:
                result.append(node.word)
            for child in node.children.values():
                self._dfs(child, result)

    # Example
    trie = Trie()
    words = ["apple", "app", "apricot", "banana"]
    for w in words:
        trie.insert(w)

    print(trie.search("app"))  # True
    print(trie.find_words_with_prefix("ap"))  # ['apple', 'app', 'apricot']
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** String indexing, prefix operations.

        **Strong answer signals:**

        - Space-efficient implementation
        - Autocomplete use case
        - Discusses compressed trie (radix tree)
        - Knows time complexity: O(m) where m = word length

---

### Longest Palindromic Substring - Expand Around Center - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `String`, `Two Pointers`, `DP` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Expand Around Center:**

    ```python
    def longest_palindrome(s):
        """Find longest palindromic substring - O(n²)"""
        if not s:
            return ""

        def expand(left, right):
            """Expand around center and return length"""
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1

        start, max_len = 0, 0

        for i in range(len(s)):
            # Odd length palindromes (single center)
            len1 = expand(i, i)
            # Even length palindromes (two centers)
            len2 = expand(i, i + 1)

            curr_len = max(len1, len2)
            if curr_len > max_len:
                max_len = curr_len
                start = i - (curr_len - 1) // 2

        return s[start:start + max_len]

    print(longest_palindrome("babad"))  # "bab" or "aba"
    ```

    **DP Approach:**

    ```python
    def longest_palindrome_dp(s):
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        start, max_len = 0, 1

        # All single chars are palindromes
        for i in range(n):
            dp[i][i] = True

        # Check length 2
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                start, max_len = i, 2

        # Check length 3 to n
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    start, max_len = i, length

        return s[start:start + max_len]
    ```

    **Manacher's Algorithm O(n):** (mention for bonus points)

    !!! tip "Interviewer's Insight"
        **What they're testing:** String manipulation, optimization.

        **Strong answer signals:**

        - Starts with expand around center
        - Mentions DP approach
        - Knows Manacher's algorithm exists
        - Handles both odd and even length palindromes

---

### Merge Intervals - Amazon, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Array`, `Sorting`, `Intervals` | **Asked by:** Amazon, Meta, Google

??? success "View Answer"

    **Solution:**

    ```python
    def merge_intervals(intervals):
        """Merge overlapping intervals"""
        if not intervals:
            return []

        # Sort by start time
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]

        for curr in intervals[1:]:
            last = merged[-1]

            if curr[0] <= last[1]:  # Overlap
                # Merge by updating end
                merged[-1] = [last[0], max(last[1], curr[1])]
            else:
                merged.append(curr)

        return merged

    # Example
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    print(merge_intervals(intervals))  # [[1, 6], [8, 10], [15, 18]]
    ```

    **Variants:**

    | Variant | Solution |
    |---------|----------|
    | Insert interval | Find insertion point, merge overlaps |
    | Remove covered | Remove if contained in another |
    | Count overlaps | Sweep line algorithm |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Interval manipulation, edge cases.

        **Strong answer signals:**

        - Sorts by start time
        - Handles all overlap cases
        - O(n log n) time complexity
        - Extends to insert interval, meeting rooms

---

### Top K Frequent Elements - Bucket Sort - Meta, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Array`, `Hash Table`, `Bucket Sort`, `Heap` | **Asked by:** Meta, Amazon, Google

??? success "View Answer"

    **Bucket Sort O(n):**

    ```python
    def top_k_frequent(nums, k):
        """Find k most frequent elements - O(n)"""
        from collections import Counter

        # Count frequencies
        freq = Counter(nums)

        # Bucket sort: index = frequency
        buckets = [[] for _ in range(len(nums) + 1)]
        for num, count in freq.items():
            buckets[count].append(num)

        # Collect top k from highest frequency
        result = []
        for i in range(len(buckets) - 1, 0, -1):
            result.extend(buckets[i])
            if len(result) >= k:
                return result[:k]

        return result

    print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))  # [1, 2]
    ```

    **Heap Approach O(n log k):**

    ```python
    import heapq

    def top_k_frequent_heap(nums, k):
        freq = Counter(nums)
        # Min heap of size k
        return heapq.nlargest(k, freq.keys(), key=freq.get)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Optimization, multiple approaches.

        **Strong answer signals:**

        - Knows bucket sort for O(n)
        - Compares with heap O(n log k)
        - Mentions quickselect alternative
        - Handles ties correctly

---

### Longest Increasing Subsequence - DP + Binary Search - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `DP`, `Binary Search`, `Array` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **DP O(n²):**

    ```python
    def length_of_LIS(nums):
        """Longest Increasing Subsequence"""
        if not nums:
            return 0

        n = len(nums)
        dp = [1] * n

        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)
    ```

    **Optimized with Binary Search O(n log n):**

    ```python
    import bisect

    def length_of_LIS_optimized(nums):
        """LIS with binary search - O(n log n)"""
        tails = []  # tails[i] = smallest ending value of LIS of length i+1

        for num in nums:
            pos = bisect.bisect_left(tails, num)

            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num

        return len(tails)

    # Example
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    print(length_of_LIS_optimized(nums))  # 4: [2, 3, 7, 101]
    ```

    **Reconstructing LIS:**

    ```python
    def find_LIS(nums):
        n = len(nums)
        tails = []
        parent = [-1] * n
        indices = []

        for i, num in enumerate(nums):
            pos = bisect.bisect_left(tails, num)

            if pos == len(tails):
                tails.append(num)
                indices.append(i)
            else:
                tails[pos] = num
                indices[pos] = i

            if pos > 0:
                parent[i] = indices[pos - 1]

        # Reconstruct
        lis = []
        curr = indices[-1]
        while curr != -1:
            lis.append(nums[curr])
            curr = parent[curr]

        return lis[::-1]
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** DP optimization, binary search.

        **Strong answer signals:**

        - Starts with O(n²) DP
        - Optimizes to O(n log n) with binary search
        - Can reconstruct actual subsequence
        - Knows patience sorting connection

---

### Word Break - DP with Trie - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `DP`, `String`, `Trie` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **DP Solution:**

    ```python
    def word_break(s, word_dict):
        """Check if string can be segmented into words"""
        word_set = set(word_dict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True

        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break

        return dp[n]

    # Example
    s = "leetcode"
    word_dict = ["leet", "code"]
    print(word_break(s, word_dict))  # True
    ```

    **Optimized with Trie:**

    ```python
    def word_break_trie(s, word_dict):
        # Build Trie
        trie = Trie()
        for word in word_dict:
            trie.insert(word)

        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True

        for i in range(n):
            if not dp[i]:
                continue

            # Check all words starting at position i
            node = trie.root
            for j in range(i, n):
                if s[j] not in node.children:
                    break
                node = node.children[s[j]]
                if node.is_end:
                    dp[j + 1] = True

        return dp[n]
    ```

    **Return All Possible Sentences:**

    ```python
    def word_break_ii(s, word_dict):
        """Return all possible sentences"""
        word_set = set(word_dict)
        memo = {}

        def backtrack(start):
            if start in memo:
                return memo[start]

            if start == len(s):
                return [[]]

            result = []
            for end in range(start + 1, len(s) + 1):
                word = s[start:end]
                if word in word_set:
                    for rest in backtrack(end):
                        result.append([word] + rest)

            memo[start] = result
            return result

        return [' '.join(words) for words in backtrack(0)]
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** DP, string manipulation, backtracking.

        **Strong answer signals:**

        - DP solution O(n² × m) where m = average word length
        - Trie optimization for many words
        - Extends to Word Break II (return all solutions)
        - Discusses memoization

---

### Coin Change - Unbounded Knapsack DP - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `DP`, `Knapsack`, `Greedy` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **DP Solution:**

    ```python
    def coin_change(coins, amount):
        """Minimum coins to make amount"""
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)

        return dp[amount] if dp[amount] != float('inf') else -1

    # Example
    coins = [1, 2, 5]
    amount = 11
    print(coin_change(coins, amount))  # 3 (5+5+1)
    ```

    **Count Number of Ways:**

    ```python
    def coin_change_ways(coins, amount):
        """Number of ways to make amount"""
        dp = [0] * (amount + 1)
        dp[0] = 1

        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] += dp[x - coin]

        return dp[amount]
    ```

    **Space Optimized:**

    ```python
    def coin_change_optimized(coins, amount):
        # Use set to avoid duplicates
        dp = {0}  # Possible amounts

        for _ in range(amount):
            new_dp = set()
            for amt in dp:
                for coin in coins:
                    if amt + coin <= amount:
                        new_dp.add(amt + coin)
            dp = new_dp

            if amount in dp:
                return True

        return False
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Unbounded knapsack, DP variants.

        **Strong answer signals:**

        - Correct loop order (coins outer or inner affects result)
        - Knows difference: min coins vs count ways
        - Space optimization possible
        - Extends to coin change with limited coins

---

### Decode Ways - DP with Constraints - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `DP`, `String` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **DP Solution:**

    ```python
    def num_decodings(s):
        """Count ways to decode string (1=A, 2=B, ..., 26=Z)"""
        if not s or s[0] == '0':
            return 0

        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1  # Empty string
        dp[1] = 1  # First character

        for i in range(2, n + 1):
            # Single digit
            if s[i-1] != '0':
                dp[i] += dp[i-1]

            # Two digits
            two_digit = int(s[i-2:i])
            if 10 <= two_digit <= 26:
                dp[i] += dp[i-2]

        return dp[n]

    # Example
    print(num_decodings("226"))  # 3: 2,2,6 or 22,6 or 2,26
    print(num_decodings("06"))   # 0: invalid
    ```

    **Space Optimized O(1):**

    ```python
    def num_decodings_optimized(s):
        if not s or s[0] == '0':
            return 0

        prev2, prev1 = 1, 1

        for i in range(1, len(s)):
            curr = 0

            if s[i] != '0':
                curr = prev1

            two_digit = int(s[i-1:i+1])
            if 10 <= two_digit <= 26:
                curr += prev2

            prev2, prev1 = prev1, curr

        return prev1
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** DP with constraints, edge cases.

        **Strong answer signals:**

        - Handles leading zeros
        - Checks valid ranges (1-26)
        - Space optimization to O(1)
        - Extends to Decode Ways II (with wildcards)

---

### Regular Expression Matching - DP - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `DP`, `String`, `Recursion` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **DP Solution:**

    ```python
    def is_match(s, p):
        """Regex matching with '.' and '*'"""
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        # Handle patterns like a*, a*b*, etc.
        for j in range(2, n + 1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-2]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j-1] == '*':
                    # Zero occurrences or one+ occurrences
                    dp[i][j] = dp[i][j-2] or \
                               (dp[i-1][j] and (s[i-1] == p[j-2] or p[j-2] == '.'))
                elif p[j-1] == '.' or s[i-1] == p[j-1]:
                    dp[i][j] = dp[i-1][j-1]

        return dp[m][n]

    # Examples
    print(is_match("aa", "a"))       # False
    print(is_match("aa", "a*"))      # True
    print(is_match("ab", ".*"))      # True
    print(is_match("aab", "c*a*b"))  # True
    ```

    **Recursive with Memoization:**

    ```python
    def is_match_recursive(s, p):
        memo = {}

        def dp(i, j):
            if (i, j) in memo:
                return memo[(i, j)]

            if j == len(p):
                return i == len(s)

            first_match = i < len(s) and (p[j] == s[i] or p[j] == '.')

            if j + 1 < len(p) and p[j + 1] == '*':
                result = dp(i, j + 2) or (first_match and dp(i + 1, j))
            else:
                result = first_match and dp(i + 1, j + 1)

            memo[(i, j)] = result
            return result

        return dp(0, 0)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Complex DP, state management.

        **Strong answer signals:**

        - Handles '*' matching zero or more
        - Edge cases: empty strings, multiple '*'
        - Both DP and recursive + memo solutions
        - Compares with wildcard matching (simpler)

---

### Edit Distance - Levenshtein Distance - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `DP`, `String` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **DP Solution:**

    ```python
    def min_distance(word1, word2):
        """Minimum edits to transform word1 to word2"""
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base cases
        for i in range(m + 1):
            dp[i][0] = i  # Delete all
        for j in range(n + 1):
            dp[0][j] = j  # Insert all

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Delete
                        dp[i][j-1],      # Insert
                        dp[i-1][j-1]     # Replace
                    )

        return dp[m][n]

    # Example
    print(min_distance("horse", "ros"))  # 3
    # horse -> rorse (replace h->r)
    # rorse -> rose (delete r)
    # rose -> ros (delete e)
    ```

    **Space Optimized:**

    ```python
    def min_distance_optimized(word1, word2):
        m, n = len(word1), len(word2)
        prev = list(range(n + 1))

        for i in range(1, m + 1):
            curr = [i]
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    curr.append(prev[j-1])
                else:
                    curr.append(1 + min(prev[j], curr[j-1], prev[j-1]))
            prev = curr

        return prev[n]
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Classic DP, space optimization.

        **Strong answer signals:**

        - Explains 3 operations (insert, delete, replace)
        - Correct DP transitions
        - Space optimization from O(mn) to O(n)
        - Reconstructs actual edit sequence

---

### Trapping Rain Water - Two Pointers - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Array`, `Two Pointers`, `Stack` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Two Pointers O(n) O(1):**

    ```python
    def trap(height):
        """Calculate trapped rainwater"""
        if not height:
            return 0

        left, right = 0, len(height) - 1
        left_max, right_max = 0, 0
        water = 0

        while left < right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water += right_max - height[right]
                right -= 1

        return water

    # Example
    height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    print(trap(height))  # 6
    ```

    **Stack Approach:**

    ```python
    def trap_stack(height):
        stack = []
        water = 0

        for i in range(len(height)):
            while stack and height[i] > height[stack[-1]]:
                top = stack.pop()

                if not stack:
                    break

                distance = i - stack[-1] - 1
                bounded_height = min(height[i], height[stack[-1]]) - height[top]
                water += distance * bounded_height

            stack.append(i)

        return water
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Two pointers, visualization.

        **Strong answer signals:**

        - Two pointer O(n) O(1) solution
        - Explains why each pointer moves
        - Mentions stack approach as alternative
        - Extends to 2D version (pour water)

---

### Serialize and Deserialize Binary Tree - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Tree`, `DFS`, `BFS`, `Design` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **DFS Preorder:**

    ```python
    class Codec:
        def serialize(self, root):
            """Serialize tree to string"""
            if not root:
                return "null"

            left = self.serialize(root.left)
            right = self.serialize(root.right)

            return f"{root.val},{left},{right}"

        def deserialize(self, data):
            """Deserialize string to tree"""
            def dfs():
                val = next(vals)
                if val == "null":
                    return None

                node = TreeNode(int(val))
                node.left = dfs()
                node.right = dfs()
                return node

            vals = iter(data.split(','))
            return dfs()

    # Example
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.right.left = TreeNode(4)
    root.right.right = TreeNode(5)

    codec = Codec()
    serialized = codec.serialize(root)
    print(serialized)  # "1,2,null,null,3,4,null,null,5,null,null"
    deserialized = codec.deserialize(serialized)
    ```

    **BFS Level Order:**

    ```python
    from collections import deque

    def serialize_bfs(root):
        if not root:
            return ""

        result = []
        queue = deque([root])

        while queue:
            node = queue.popleft()
            if node:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append("null")

        return ",".join(result)

    def deserialize_bfs(data):
        if not data:
            return None

        vals = data.split(',')
        root = TreeNode(int(vals[0]))
        queue = deque([root])
        i = 1

        while queue:
            node = queue.popleft()

            if vals[i] != "null":
                node.left = TreeNode(int(vals[i]))
                queue.append(node.left)
            i += 1

            if vals[i] != "null":
                node.right = TreeNode(int(vals[i]))
                queue.append(node.right)
            i += 1

        return root
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Tree traversal, string manipulation.

        **Strong answer signals:**

        - Both DFS and BFS approaches
        - Handles null nodes correctly
        - Space-efficient encoding
        - Extends to BST (can optimize further)

---

### Meeting Rooms II - Minimum Conference Rooms - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Array`, `Heap`, `Sorting`, `Greedy` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Heap Solution:**

    ```python
    import heapq

    def min_meeting_rooms(intervals):
        """Minimum rooms needed for all meetings"""
        if not intervals:
            return 0

        # Sort by start time
        intervals.sort(key=lambda x: x[0])

        # Min heap of end times
        heap = []
        heapq.heappush(heap, intervals[0][1])

        for start, end in intervals[1:]:
            # If earliest ending meeting is done, reuse room
            if start >= heap[0]:
                heapq.heappop(heap)

            heapq.heappush(heap, end)

        return len(heap)

    # Example
    intervals = [[0, 30], [5, 10], [15, 20]]
    print(min_meeting_rooms(intervals))  # 2
    ```

    **Chronological Ordering:**

    ```python
    def min_meeting_rooms_sweep(intervals):
        """Using sweep line algorithm"""
        starts = sorted([i[0] for i in intervals])
        ends = sorted([i[1] for i in intervals])

        rooms = 0
        max_rooms = 0
        s, e = 0, 0

        while s < len(starts):
            if starts[s] < ends[e]:
                rooms += 1
                max_rooms = max(max_rooms, rooms)
                s += 1
            else:
                rooms -= 1
                e += 1

        return max_rooms
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Interval scheduling, greedy algorithms.

        **Strong answer signals:**

        - Min heap O(n log n) solution
        - Sweep line alternative
        - Extends to: can attend all meetings (single room)
        - Explains why sorting by start time works

---

### Find Median from Data Stream - Two Heaps - Meta, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Heap`, `Design`, `Data Structures` | **Asked by:** Meta, Amazon, Google

??? success "View Answer"

    **Two Heaps Approach:**

    ```python
    import heapq

    class MedianFinder:
        def __init__(self):
            self.small = []  # Max heap (invert values)
            self.large = []  # Min heap

        def addNum(self, num):
            """Add number maintaining median"""
            # Add to max heap (small)
            heapq.heappush(self.small, -num)

            # Balance: ensure max of small <= min of large
            if self.small and self.large and (-self.small[0] > self.large[0]):
                val = -heapq.heappop(self.small)
                heapq.heappush(self.large, val)

            # Balance sizes: small can have at most 1 more than large
            if len(self.small) > len(self.large) + 1:
                val = -heapq.heappop(self.small)
                heapq.heappush(self.large, val)
            elif len(self.large) > len(self.small):
                val = heapq.heappop(self.large)
                heapq.heappush(self.small, -val)

        def findMedian(self):
            """Return current median"""
            if len(self.small) > len(self.large):
                return -self.small[0]
            return (-self.small[0] + self.large[0]) / 2.0

    # Example
    mf = MedianFinder()
    mf.addNum(1)
    mf.addNum(2)
    print(mf.findMedian())  # 1.5
    mf.addNum(3)
    print(mf.findMedian())  # 2.0
    ```

    **Time Complexity:**
    - addNum: O(log n)
    - findMedian: O(1)

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced data structures, design.

        **Strong answer signals:**

        - Two heap approach
        - Maintains invariants correctly
        - O(log n) add, O(1) find
        - Discusses alternatives (BST, segment tree)

---

### Largest Rectangle in Histogram - Monotonic Stack - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Stack`, `Array` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Monotonic Stack:**

    ```python
    def largest_rectangle_area(heights):
        """Find largest rectangle in histogram"""
        stack = []
        max_area = 0
        heights = heights + [0]  # Add sentinel

        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height_idx = stack.pop()
                height = heights[height_idx]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)

            stack.append(i)

        return max_area

    # Example
    heights = [2, 1, 5, 6, 2, 3]
    print(largest_rectangle_area(heights))  # 10 (5*2)
    ```

    **Cleaner Version:**

    ```python
    def largest_rectangle_cleaner(heights):
        stack = [-1]
        max_area = 0

        for i in range(len(heights)):
            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(i)

        while stack[-1] != -1:
            h = heights[stack.pop()]
            w = len(heights) - stack[-1] - 1
            max_area = max(max_area, h * w)

        return max_area
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Monotonic stack, histogram problems.

        **Strong answer signals:**

        - Uses monotonic increasing stack
        - O(n) time, O(n) space
        - Explains width calculation
        - Extends to maximal rectangle in matrix

---

### Maximal Rectangle in Binary Matrix - DP + Histogram - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `DP`, `Stack`, `Matrix` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Using Largest Rectangle in Histogram:**

    ```python
    def maximal_rectangle(matrix):
        """Find largest rectangle of 1s in binary matrix"""
        if not matrix:
            return 0

        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        max_area = 0

        for i in range(rows):
            for j in range(cols):
                # Update histogram heights
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0

            # Find max rectangle in current histogram
            max_area = max(max_area, largest_rectangle_area(heights))

        return max_area

    def largest_rectangle_area(heights):
        """Helper from previous problem"""
        stack = []
        max_area = 0
        heights = heights + [0]

        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height_idx = stack.pop()
                height = heights[height_idx]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)

        return max_area

    # Example
    matrix = [
        ["1", "0", "1", "0", "0"],
        ["1", "0", "1", "1", "1"],
        ["1", "1", "1", "1", "1"],
        ["1", "0", "0", "1", "0"]
    ]
    print(maximal_rectangle(matrix))  # 6
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Combining algorithms, 2D problems.

        **Strong answer signals:**

        - Reduces to histogram problem
        - Builds histogram row by row
        - O(rows × cols) complexity
        - Clean code reuse

---

### Word Ladder - BFS Shortest Path - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `BFS`, `Graph`, `String` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Bidirectional BFS:**

    ```python
    from collections import deque

    def ladder_length(begin_word, end_word, word_list):
        """Find shortest transformation sequence length"""
        word_set = set(word_list)
        if end_word not in word_set:
            return 0

        # Bidirectional BFS
        begin_set = {begin_word}
        end_set = {end_word}
        visited = set()
        length = 1

        while begin_set and end_set:
            # Always expand smaller set
            if len(begin_set) > len(end_set):
                begin_set, end_set = end_set, begin_set

            next_set = set()
            for word in begin_set:
                for i in range(len(word)):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        new_word = word[:i] + c + word[i+1:]

                        if new_word in end_set:
                            return length + 1

                        if new_word in word_set and new_word not in visited:
                            next_set.add(new_word)
                            visited.add(new_word)

            begin_set = next_set
            length += 1

        return 0

    # Example
    begin = "hit"
    end = "cog"
    word_list = ["hot", "dot", "dog", "lot", "log", "cog"]
    print(ladder_length(begin, end, word_list))  # 5: hit->hot->dot->dog->cog
    ```

    **Standard BFS:**

    ```python
    def ladder_length_bfs(begin_word, end_word, word_list):
        word_set = set(word_list)
        if end_word not in word_set:
            return 0

        queue = deque([(begin_word, 1)])

        while queue:
            word, length = queue.popleft()

            if word == end_word:
                return length

            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = word[:i] + c + word[i+1:]

                    if new_word in word_set:
                        word_set.remove(new_word)
                        queue.append((new_word, length + 1))

        return 0
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** BFS, graph modeling, optimization.

        **Strong answer signals:**

        - Models as graph problem
        - Bidirectional BFS for optimization
        - Character-by-character transformation
        - Word Ladder II: return all shortest paths

---

### Palindrome Partitioning - Backtracking + DP - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Backtracking`, `DP`, `String` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Backtracking with Palindrome Check:**

    ```python
    def partition(s):
        """Return all palindrome partitions"""
        def is_palindrome(sub):
            return sub == sub[::-1]

        def backtrack(start, path):
            if start == len(s):
                result.append(path[:])
                return

            for end in range(start + 1, len(s) + 1):
                substring = s[start:end]
                if is_palindrome(substring):
                    path.append(substring)
                    backtrack(end, path)
                    path.pop()

        result = []
        backtrack(0, [])
        return result

    # Example
    print(partition("aab"))
    # [['a', 'a', 'b'], ['aa', 'b']]
    ```

    **Optimized with DP Palindrome Check:**

    ```python
    def partition_optimized(s):
        n = len(s)

        # Precompute palindrome checks
        is_pal = [[False] * n for _ in range(n)]
        for i in range(n):
            is_pal[i][i] = True

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    is_pal[i][j] = (length == 2) or is_pal[i+1][j-1]

        def backtrack(start, path):
            if start == n:
                result.append(path[:])
                return

            for end in range(start, n):
                if is_pal[start][end]:
                    path.append(s[start:end+1])
                    backtrack(end + 1, path)
                    path.pop()

        result = []
        backtrack(0, [])
        return result
    ```

    **Minimum Cuts (Variation):**

    ```python
    def min_cut(s):
        """Minimum cuts to make all palindromes"""
        n = len(s)
        # dp[i] = min cuts for s[:i+1]
        dp = list(range(n))

        for i in range(n):
            # Odd length palindromes
            left = right = i
            while left >= 0 and right < n and s[left] == s[right]:
                dp[right] = min(dp[right], (dp[left-1] if left > 0 else 0) + 1)
                left -= 1
                right += 1

            # Even length palindromes
            left, right = i, i + 1
            while left >= 0 and right < n and s[left] == s[right]:
                dp[right] = min(dp[right], (dp[left-1] if left > 0 else 0) + 1)
                left -= 1
                right += 1

        return dp[n-1] - 1
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Backtracking, DP optimization.

        **Strong answer signals:**

        - Backtracking with pruning
        - Precomputes palindrome checks
        - Extends to min cuts problem
        - Discusses time complexity improvement

---

### N-Queens Problem - Backtracking - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Backtracking`, `Matrix` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Backtracking Solution:**

    ```python
    def solve_n_queens(n):
        """Find all solutions to n-queens"""
        def is_safe(row, col):
            # Check column
            for r in range(row):
                if board[r] == col:
                    return False

            # Check diagonals
            for r in range(row):
                if abs(board[r] - col) == abs(r - row):
                    return False

            return True

        def backtrack(row):
            if row == n:
                result.append(construct_board())
                return

            for col in range(n):
                if is_safe(row, col):
                    board[row] = col
                    backtrack(row + 1)
                    board[row] = -1

        def construct_board():
            return ['.' * board[i] + 'Q' + '.' * (n - board[i] - 1)
                    for i in range(n)]

        board = [-1] * n
        result = []
        backtrack(0)
        return result

    # Example
    solutions = solve_n_queens(4)
    for sol in solutions:
        for row in sol:
            print(row)
        print()
    ```

    **Optimized with Sets:**

    ```python
    def solve_n_queens_optimized(n):
        def backtrack(row):
            if row == n:
                result.append(board[:])
                return

            for col in range(n):
                diag1 = row - col
                diag2 = row + col

                if col in cols or diag1 in diag1_set or diag2 in diag2_set:
                    continue

                board[row] = col
                cols.add(col)
                diag1_set.add(diag1)
                diag2_set.add(diag2)

                backtrack(row + 1)

                cols.remove(col)
                diag1_set.remove(diag1)
                diag2_set.remove(diag2)

        board = [-1] * n
        cols = set()
        diag1_set = set()
        diag2_set = set()
        result = []

        backtrack(0)
        return [['..' * board[i] + 'Q' + '.' * (n - board[i] - 1)
                 for i in range(n)] for board in result]
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Backtracking, constraint satisfaction.

        **Strong answer signals:**

        - Efficient conflict checking with sets
        - Diagonal formula: row ± col
        - Discusses symmetry reduction
        - Total N-Queens count (just count, not construct)

---

### Shortest Path in Weighted Graph - Dijkstra's Algorithm - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Graph`, `Dijkstra`, `Heap`, `Shortest Path` | **Asked by:** Amazon, Google, Microsoft

??? success "View Answer"

    **Dijkstra's Algorithm:**

    ```python
    import heapq

    def dijkstra(graph, start):
        """Find shortest paths from start to all nodes"""
        distances = {node: float('inf') for node in graph}
        distances[start] = 0

        # Min heap: (distance, node)
        heap = [(0, start)]
        visited = set()

        while heap:
            curr_dist, curr_node = heapq.heappop(heap)

            if curr_node in visited:
                continue

            visited.add(curr_node)

            for neighbor, weight in graph[curr_node]:
                distance = curr_dist + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(heap, (distance, neighbor))

        return distances

    # Example
    graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('C', 1), ('D', 5)],
        'C': [('D', 8), ('E', 10)],
        'D': [('E', 2)],
        'E': []
    }
    print(dijkstra(graph, 'A'))
    # {'A': 0, 'B': 4, 'C': 2, 'D': 9, 'E': 11}
    ```

    **With Path Reconstruction:**

    ```python
    def dijkstra_with_path(graph, start, end):
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        parent = {node: None for node in graph}

        heap = [(0, start)]
        visited = set()

        while heap:
            curr_dist, curr_node = heapq.heappop(heap)

            if curr_node == end:
                break

            if curr_node in visited:
                continue

            visited.add(curr_node)

            for neighbor, weight in graph[curr_node]:
                distance = curr_dist + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    parent[neighbor] = curr_node
                    heapq.heappush(heap, (distance, neighbor))

        # Reconstruct path
        path = []
        curr = end
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()

        return distances[end], path
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Graph algorithms, greedy approach.

        **Strong answer signals:**

        - Min heap for O((V + E) log V)
        - Handles visited nodes correctly
        - Path reconstruction
        - Compares with Bellman-Ford (negative weights), A* (heuristic)

---

### Alien Dictionary - Topological Sort - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Graph`, `Topological Sort`, `BFS` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Topological Sort Solution:**

    ```python
    from collections import defaultdict, deque

    def alien_order(words):
        """Find alien alphabet order from sorted words"""
        # Build graph
        graph = defaultdict(set)
        in_degree = {char: 0 for word in words for char in word}

        # Compare adjacent words
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            min_len = min(len(w1), len(w2))

            # Find first different character
            for j in range(min_len):
                if w1[j] != w2[j]:
                    if w2[j] not in graph[w1[j]]:
                        graph[w1[j]].add(w2[j])
                        in_degree[w2[j]] += 1
                    break
            else:
                # w1 is prefix of w2, check validity
                if len(w1) > len(w2):
                    return ""  # Invalid

        # Kahn's algorithm (BFS topological sort)
        queue = deque([char for char in in_degree if in_degree[char] == 0])
        result = []

        while queue:
            char = queue.popleft()
            result.append(char)

            for neighbor in graph[char]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycle
        if len(result) != len(in_degree):
            return ""

        return ''.join(result)

    # Example
    words = ["wrt", "wrf", "er", "ett", "rftt"]
    print(alien_order(words))  # "wertf"
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Graph modeling, topological sort.

        **Strong answer signals:**

        - Builds graph from adjacent words
        - Uses Kahn's algorithm for topo sort
        - Detects cycles (invalid input)
        - Edge case: prefix longer than following word

---

### Word Search II - Trie + DFS - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Trie`, `Backtracking`, `Matrix` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Trie + Backtracking:**

    ```python
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word = None

    def find_words(board, words):
        """Find all words that exist in board"""
        # Build Trie
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word

        rows, cols = len(board), len(board[0])
        result = []

        def dfs(r, c, node):
            char = board[r][c]

            if char not in node.children:
                return

            next_node = node.children[char]

            # Found word
            if next_node.word:
                result.append(next_node.word)
                next_node.word = None  # Avoid duplicates

            # Mark visited
            board[r][c] = '#'

            # Explore neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                    dfs(nr, nc, next_node)

            # Backtrack
            board[r][c] = char

            # Prune trie
            if not next_node.children:
                del node.children[char]

        for r in range(rows):
            for c in range(cols):
                if board[r][c] in root.children:
                    dfs(r, c, root)

        return result

    # Example
    board = [
        ['o', 'a', 'a', 'n'],
        ['e', 't', 'a', 'e'],
        ['i', 'h', 'k', 'r'],
        ['i', 'f', 'l', 'v']
    ]
    words = ["oath", "pea", "eat", "rain"]
    print(find_words(board, words))  # ['oath', 'eat']
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Trie optimization, backtracking.

        **Strong answer signals:**

        - Uses Trie to avoid repeated work
        - Prunes Trie as words are found
        - O(m × n × 4^L) where L = max word length
        - Much better than checking each word separately

---

### Maximum Profit in Job Scheduling - DP + Binary Search - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `DP`, `Binary Search`, `Sorting` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **DP with Binary Search:**

    ```python
    import bisect

    def job_scheduling(start_time, end_time, profit):
        """Max profit from non-overlapping jobs"""
        n = len(start_time)
        jobs = sorted(zip(end_time, start_time, profit))

        # dp[i] = max profit considering first i jobs
        dp = [0] * (n + 1)

        for i in range(1, n + 1):
            end, start, p = jobs[i-1]

            # Find latest non-overlapping job
            k = bisect.bisect_right([jobs[j][0] for j in range(i-1)], start)

            dp[i] = max(dp[i-1], dp[k] + p)

        return dp[n]
    ```

    **Cleaner Implementation:**

    ```python
    def job_scheduling_clean(start_time, end_time, profit):
        jobs = sorted(zip(end_time, start_time, profit))
        ends = [j[0] for j in jobs]
        dp = [(0, 0)]  # (end_time, max_profit)

        for end, start, p in jobs:
            idx = bisect.bisect_right(dp, (start, float('inf'))) - 1
            profit_if_taken = dp[idx][1] + p

            if profit_if_taken > dp[-1][1]:
                dp.append((end, profit_if_taken))

        return dp[-1][1]
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** DP with optimization, binary search.

        **Strong answer signals:**

        - Sorts by end time
        - Uses binary search for non-overlapping
        - Knows weighted interval scheduling
        - O(n log n) solution

---

## Quick Reference: 100+ Interview Questions


|-----|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|--------------|--------------------------------------------------|
| 1   | Two Number Sum                                      | [LeetCode Two Sum](https://leetcode.com/problems/two-sum/)                                                                                                              | Google, Facebook, Amazon                        | Easy         | Array, Hashing                                   |
| 2   | Reverse Linked List                                 | [LeetCode Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)                                                                                        | Amazon, Facebook, Microsoft                     | Easy         | Linked List                                      |
| 3   | Valid Parentheses                                   | [LeetCode Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)                                                                                            | Amazon, Facebook, Google                        | Easy         | Stack, String                                    |
| 4   | Binary Search                                       | [LeetCode Binary Search](https://leetcode.com/problems/binary-search/)                                                                                                    | Google, Facebook, Amazon                        | Easy         | Array, Binary Search                             |
| 5   | Merge Two Sorted Arrays                             | [LeetCode Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)                                                                                          | Google, Microsoft, Amazon                       | Easy         | Array, Two Pointers                              |
| 6   | Meeting Rooms                                       | [LeetCode Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)                                                                                              | Microsoft, Google                               | Medium       | Array, Sorting, Interval Scheduling              |
| 7   | Climbing Stairs                                     | [LeetCode Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)                                                                                                | Amazon, Facebook, Google                        | Easy         | Dynamic Programming                              |
| 8   | Valid Anagram                                       | [LeetCode Valid Anagram](https://leetcode.com/problems/valid-anagram/)                                                                                                    | Google, Amazon                                  | Easy         | String, Hashing                                  |
| 9   | Longest Substring Without Repeating Characters      | [LeetCode Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)                                | Amazon, Facebook, Google                        | Medium       | String, Hashing, Sliding Window                  |
| 10  | Maximum Subarray (Kadane's Algorithm)               | [LeetCode Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)                                                                                              | Google, Amazon, Facebook                        | Medium       | Array, Dynamic Programming                       |
| 11  | Word Ladder                                         | [LeetCode Word Ladder](https://leetcode.com/problems/word-ladder/)                                                                                                       | Google, Amazon, Facebook                        | Very Hard    | Graph, BFS, String Transformation                |
| 12  | 4Sum (Four Number Sum)                              | [LeetCode 4Sum](https://leetcode.com/problems/4sum/)                                                                                                                    | Amazon, Facebook, Google                        | Hard         | Array, Hashing, Two Pointers                      |
| 13  | Median of Two Sorted Arrays                         | [LeetCode Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)                                                                         | Google, Amazon, Microsoft                       | Hard         | Array, Binary Search                             |
| 14  | Longest Increasing Subsequence                      | [LeetCode Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)                                                                  | Google, Facebook, Amazon                        | Hard         | Array, Dynamic Programming                       |
| 15  | Longest Palindromic Substring                        | [LeetCode Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)                                                                   | Amazon, Google                                  | Hard         | String, Dynamic Programming                      |
| 16  | Design LRU Cache                                    | [LeetCode LRU Cache](https://leetcode.com/problems/lru-cache/)                                                                                                          | Amazon, Facebook, Google, Microsoft             | Hard         | Design, Hashing, Linked List                     |
| 17  | Top K Frequent Elements                             | [LeetCode Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)                                                                                | Google, Facebook, Amazon                        | Medium       | Array, Hashing, Heap                             |
| 18  | Find Peak Element                                   | [LeetCode Find Peak Element](https://leetcode.com/problems/find-peak-element/)                                                                                            | Google, Facebook, Amazon                        | Medium       | Array, Binary Search                             |
| 19  | Candy (Min Rewards)                                 | [LeetCode Candy](https://leetcode.com/problems/candy/)                                                                                                                  | Amazon, Facebook, Google                        | Hard         | Array, Greedy                                    |
| 20  | Array of Products                                   | [LeetCode Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)                                                                      | Amazon, Google                                  | Medium       | Array, Prefix/Suffix Products                    |
| 21  | First Duplicate Value                               | [LeetCode Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)                                                                            | Google, Facebook                                | Medium       | Array, Hashing                                   |
| 22  | Validate Subsequence                                | [GFG Validate Subsequence](https://www.geeksforgeeks.org/problems/array-subset-of-another-array2317/1)                                                                   | Amazon, Google, Microsoft                       | Easy         | Array, Two Pointers                              |
| 23  | Nth Fibonacci                                       | [LeetCode Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)                                                                                            | Google, Facebook, Microsoft                     | Easy         | Recursion, Dynamic Programming                   |
| 24  | Spiral Traverse                                     | [LeetCode Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)                                                                                                  | Facebook, Amazon, Google                        | Medium       | Matrix, Simulation                               |
| 25  | Subarray Sort                                       | [GFG Minimum Unsorted Subarray](https://www.geeksforgeeks.org/minimum-length-unsorted-subarray-sorting-possible/)                                                         | Google, Uber                                    | Hard         | Array, Two Pointers                              |
| 26  | Largest Range                                       | [GFG Largest Range](https://www.geeksforgeeks.org/find-the-largest-range/)                                                                                              | Google, Amazon                                  | Hard         | Array, Hashing                                   |
| 27  | Diagonal Traverse                                   | [LeetCode Diagonal Traverse](https://leetcode.com/problems/diagonal-traverse/)                                                                                            | Google, Facebook                                | Medium       | Array, Simulation                               |
| 28  | Longest Peak                                        | [GFG Longest Peak](https://www.geeksforgeeks.org/longest-peak-problem/)                                                                                                  | Google, Uber                                    | Medium       | Array, Dynamic Programming                       |
| 29  | Product Sum                                         | [GFG Product Sum](https://www.geeksforgeeks.org/product-sum-of-a-special-array/)                                                                                        | Amazon, Facebook, Google                        | Easy         | Array, Recursion                                 |
| 30  | Merge Two Sorted Lists                              | [LeetCode Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)                                                                                  | Google, Amazon, Facebook                        | Medium       | Linked List, Recursion                           |
| 31  | Binary Tree Level Order Traversal                   | [LeetCode Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)                                                                        | Amazon, Google, Microsoft                       | Easy         | Tree, BFS                                      |
| 32  | Longest Valid Parentheses                           | [LeetCode Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/)                                                                            | Facebook, Google, Amazon                        | Medium       | String, Stack, Dynamic Programming               |
| 33  | Word Break                                         | [LeetCode Word Break](https://leetcode.com/problems/word-break/)                                                                                                        | Amazon, Google, Facebook                        | Hard         | Dynamic Programming, String                      |
| 34  | Find Median from Data Stream                        | [LeetCode Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)                                                                      | Facebook, Amazon, Google                        | Hard         | Heap, Data Structures                            |
| 35  | Longest Repeating Character Replacement           | [LeetCode Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)                                                | Google, Amazon, Facebook                        | Hard         | String, Sliding Window, Greedy                   |
| 36  | Kth Largest Element in an Array                     | [LeetCode Kth Largest Element](https://leetcode.com/problems/kth-largest-element-in-an-array/)                                                                            | Google, Amazon, Facebook                        | Medium       | Heap, Sorting                                    |
| 37  | River Sizes                                         | [GFG River Sizes](https://www.geeksforgeeks.org/river-sizes/)                                                                                                            | Facebook, Google                                | Very Hard    | Graph, DFS/BFS, Matrix                           |
| 38  | Youngest Common Ancestor                            | [LeetCode Lowest Common Ancestor](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)                                                                | Google, Microsoft                               | Very Hard    | Tree, Ancestor Tracking                          |
| 39  | BST Construction                                    | [LeetCode Validate BST](https://leetcode.com/problems/validate-binary-search-tree/)                                                                                      | Facebook, Amazon, Google                        | Very Hard    | Tree, Binary Search Tree                         |
| 40  | Invert Binary Tree                                  | [LeetCode Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)                                                                                         | Amazon, Facebook, Google                        | Very Hard    | Tree, Recursion                                  |
| 41  | Validate BST                                        | [LeetCode Validate BST](https://leetcode.com/problems/validate-binary-search-tree/)                                                                                      | Google, Amazon                                  | Very Hard    | Tree, Binary Search Tree                         |
| 42  | Node Depths                                         | [GFG Sum of Node Depths](https://www.geeksforgeeks.org/sum-of-depths-of-all-nodes-in-a-binary-tree/)                                                                    | Google, Facebook                                | Very Hard    | Tree, Recursion                                  |
| 43  | Branch Sums                                         | [GFG Branch Sums](https://www.geeksforgeeks.org/branch-sum/)                                                                                                             | Amazon, Facebook, Google                        | Very Hard    | Tree, Recursion                                  |
| 44  | Find Successor                                      | [LeetCode Inorder Successor](https://leetcode.com/problems/inorder-successor-in-bst/)                                                                                     | Facebook, Amazon, Google                        | Very Hard    | Tree, BST, Inorder Traversal                     |
| 45  | Binary Tree Diameter                                | [GFG Diameter of Binary Tree](https://www.geeksforgeeks.org/diameter-of-a-binary-tree/)                                                                                 | Google, Uber                                    | Very Hard    | Tree, Recursion                                  |
| 46  | Lowest Common Ancestor                              | [LeetCode Lowest Common Ancestor](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)                                                                | Amazon, Facebook, Google                        | Very Hard    | Tree, Recursion                                  |
| 47  | Dijkstra's Algorithm                                | [LeetCode Network Delay Time](https://leetcode.com/problems/network-delay-time/)                                                                                        | Google, Amazon                                  | Very Hard    | Graph, Shortest Paths, Greedy                    |
| 48  | Topological Sort                                    | [GFG Topological Sort](https://www.geeksforgeeks.org/topological-sorting/)                                                                                             | Google, Microsoft, Amazon                       | Very Hard    | Graph, DFS/BFS, Sorting                          |
| 49  | Knapsack Problem                                    | [LeetCode Coin Change 2](https://leetcode.com/problems/coin-change-2/)                                                                                                  | Facebook, Amazon, Google                        | Very Hard    | Dynamic Programming, Knapsack                    |
| 50  | Disk Stacking                                       | [GFG Disk Stacking](https://www.geeksforgeeks.org/disk-stacking-problem/)                                                                                               | Google, Facebook                                | Very Hard    | Dynamic Programming, Sorting                     |
| 51  | Numbers In Pi                                       | N/A                                                                                                                                                                    | Google, Facebook                                | Very Hard    | Dynamic Programming, String Processing           |
| 52  | Longest Common Subsequence                          | [LeetCode Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)                                                                         | Amazon, Google, Microsoft                       | Very Hard    | Dynamic Programming, Strings                     |
| 53  | Min Number of Jumps                                 | [LeetCode Min Number of Jumps](https://leetcode.com/problems/min-number-of-jumps/)                                                                                       | Google, Facebook, Amazon                        | Very Hard    | Dynamic Programming, Greedy                      |
| 54  | Water Area (Trapping Rain Water)                    | [LeetCode Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)                                                                                       | Google, Amazon, Facebook                        | Very Hard    | Array, Two Pointers, Greedy                      |
| 55  | Minimum Characters For Palindrome                   | [GFG Minimum Characters For Palindrome](https://www.geeksforgeeks.org/minimum-number-of-characters-needed-to-make-a-string-palindrome/)                                 | Amazon, Google                                  | Very Hard    | String, Dynamic Programming, KMP                 |
| 56  | Regular Expression Matching                         | [LeetCode Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/)                                                                       | Google, Amazon, Facebook                        | Very Hard    | Dynamic Programming, Strings, Recursion          |
| 57  | Wildcard Matching                                   | [LeetCode Wildcard Matching](https://leetcode.com/problems/wildcard-matching/)                                                                                         | Amazon, Google                                  | Very Hard    | Dynamic Programming, Strings                     |
| 58  | Group Anagrams                                      | [LeetCode Group Anagrams](https://leetcode.com/problems/group-anagrams/)                                                                                               | Google, Amazon, Facebook                        | Medium       | Array, Hashing                                   |
| 59  | Longest Consecutive Sequence                        | [LeetCode Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)                                                                     | Facebook, Google, Amazon                        | Hard         | Array, Hashing                                   |
| 60  | Maximum Product Subarray                            | [LeetCode Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)                                                                             | Amazon, Google, Facebook                        | Medium       | Array, Dynamic Programming                       |
| 61  | Sum of Two Integers (Bit Manipulation)              | [LeetCode Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/)                                                                                     | Google, Amazon, Facebook                        | Medium       | Bit Manipulation                                 |
| 62  | Course Schedule                                     | [LeetCode Course Schedule](https://leetcode.com/problems/course-schedule/)                                                                                             | Amazon, Facebook, Google                        | Medium       | Graph, DFS/BFS                                   |
| 63  | Add Two Numbers (Linked List)                       | [LeetCode Add Two Numbers](https://leetcode.com/problems/add-two-numbers/)                                                                                             | Google, Facebook, Amazon                        | Medium       | Linked List, Math                                |
| 64  | Reverse Words in a String                           | [LeetCode Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/)                                                                           | Google, Amazon, Facebook                        | Medium       | String, Two Pointers                             |
| 65  | Intersection of Two Arrays                          | [LeetCode Intersection of Two Arrays](https://leetcode.com/problems/intersection-of-two-arrays/)                                                                         | Amazon, Google, Facebook                        | Easy         | Array, Hashing                                   |
| 66  | Find All Duplicates in an Array                     | [LeetCode Find All Duplicates](https://leetcode.com/problems/find-all-duplicates-in-an-array/)                                                                           | Facebook, Google, Amazon                        | Medium       | Array, Hashing                                   |
| 67  | Majority Element                                    | [LeetCode Majority Element](https://leetcode.com/problems/majority-element/)                                                                                           | Google, Amazon                                  | Easy         | Array, Hashing, Boyer-Moore                        |
| 68  | Rotate Array                                        | [LeetCode Rotate Array](https://leetcode.com/problems/rotate-array/)                                                                                                   | Amazon, Google, Facebook                        | Medium       | Array, Two Pointers                              |
| 69  | Spiral Matrix II                                    | [LeetCode Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/)                                                                                           | Google, Facebook, Amazon                        | Medium       | Matrix, Simulation                               |
| 70  | Search in Rotated Sorted Array                      | [LeetCode Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)                                                                 | Google, Amazon, Facebook                        | Medium       | Array, Binary Search                             |
| 71  | Design a URL Shortener                              | [LeetCode Design TinyURL](https://leetcode.com/problems/design-tinyurl/)                                                                                               | Uber, Airbnb, Flipkart                          | Medium       | Design, Hashing, Strings                         |
| 72  | Implement Autocomplete System                       | [GFG Autocomplete System](https://www.geeksforgeeks.org/implementing-autocomplete-system/)                                                                              | Amazon, Google, Swiggy                          | Hard         | Trie, Design, Strings                            |
| 73  | Design Twitter Feed                                 | [LeetCode Design Twitter](https://leetcode.com/problems/design-twitter/)                                                                                               | Twitter, Flipkart, Ola                          | Medium       | Design, Heap, Linked List                        |
| 74  | Implement LFU Cache                                 | [GFG LFU Cache](https://www.geeksforgeeks.org/lfu-cache-implementation/)                                                                                               | Amazon, Paytm, Flipkart                         | Hard         | Design, Hashing                                  |
| 75  | Design a Rate Limiter                               | N/A                                                                                                                                                                    | Uber, Ola, Swiggy                               | Medium       | Design, Algorithms                               |
| 76  | Serialize and Deserialize Binary Tree             | [LeetCode Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)                                                    | Amazon, Microsoft, Swiggy                       | Hard         | Tree, DFS, Design                                |
| 77  | Design a File System                                | [LeetCode Design File System](https://leetcode.com/problems/design-file-system/)                                                                                         | Google, Flipkart, Amazon                        | Hard         | Design, Trie                                     |
| 78  | Implement Magic Dictionary                          | [LeetCode Implement Magic Dictionary](https://leetcode.com/problems/implement-magic-dictionary/)                                                                         | Facebook, Microsoft, Paytm                      | Medium       | Trie, Design                                     |
| 79  | Longest Substring with At Most K Distinct Characters| [LeetCode Longest Substring with At Most K Distinct Characters](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/)                  | Amazon, Google                                  | Medium       | String, Sliding Window                           |
| 80  | Subarray Sum Equals K                              | [LeetCode Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)                                                                                   | Microsoft, Amazon, Flipkart                     | Medium       | Array, Hashing, Prefix Sum                       |
| 81  | Merge k Sorted Lists                                | [LeetCode Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)                                                                                     | Google, Facebook, Amazon                        | Hard         | Heap, Linked List                                |
| 82  | Longest Increasing Path in a Matrix               | [LeetCode Longest Increasing Path](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/)                                                                 | Google, Microsoft                               | Hard         | DFS, DP, Matrix                                  |
| 83  | Design a Stock Price Fluctuation Tracker           | [LeetCode Stock Price Fluctuation](https://leetcode.com/problems/stock-price-fluctuation/)                                                                               | Amazon, Flipkart, Paytm                         | Medium       | Design, Heap                                     |
| 84  | Implement a Trie                                   | [LeetCode Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/)                                                                                    | Amazon, Google, Microsoft                       | Medium       | Trie, Design                                     |
| 85  | Design a Chat System                               | [Medium: Chat System Design](https://medium.com/swlh/building-a-scalable-chat-system-f7dd1705dce6) *(free article)*                                                   | WhatsApp, Slack, Swiggy                         | Hard         | Design, Messaging                                |
| 86  | Design an Elevator System                          | N/A                                                                                                                                                                    | OYO, Ola, Flipkart                              | Hard         | Design, System Design                            |
| 87  | Implement a Sudoku Solver                          | [LeetCode Sudoku Solver](https://leetcode.com/problems/sudoku-solver/)                                                                                                  | Google, Microsoft, Amazon                       | Hard         | Backtracking, Recursion                          |
| 88  | Find All Anagrams in a String                      | [LeetCode Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)                                                                  | Facebook, Google                                | Medium       | String, Sliding Window, Hashing                  |
| 89  | Design Twitter-like Feed                           | [LeetCode Design Twitter](https://leetcode.com/problems/design-twitter/)                                                                                               | Twitter, Facebook, Uber                         | Medium       | Design, Heap, Linked List                        |
| 90  | Longest Palindromic Subsequence                   | [LeetCode Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)                                                               | Amazon, Google                                  | Medium       | DP, String                                       |
| 91  | Clone Graph                                       | [LeetCode Clone Graph](https://leetcode.com/problems/clone-graph/)                                                                                                     | Amazon, Google                                  | Medium       | Graph, DFS/BFS                                   |
| 92  | Design a Data Structure for the Stock Span Problem | [LeetCode Online Stock Span](https://leetcode.com/problems/online-stock-span/)                                                                                         | Amazon, Microsoft, Paytm                        | Medium       | Stack, Array, Design                             |
| 93  | Design a Stack That Supports getMin()             | [LeetCode Min Stack](https://leetcode.com/problems/min-stack/)                                                                                                         | Facebook, Amazon, Google                        | Easy         | Stack, Design                                    |
| 94  | Convert Sorted Array to Binary Search Tree         | [LeetCode Sorted Array to BST](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)                                                                 | Facebook, Google                                | Easy         | Tree, Recursion                                  |
| 95  | Meeting Rooms II                                  | [LeetCode Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)                                                                                           | Microsoft, Google                               | Medium       | Array, Heap, Sorting                             |
| 96  | Search in Rotated Sorted Array                    | [LeetCode Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)                                                                 | Google, Amazon, Facebook                        | Medium       | Array, Binary Search                             |
| 97  | Design a URL Shortener                            | [LeetCode Design TinyURL](https://leetcode.com/problems/design-tinyurl/)                                                                                               | Uber, Airbnb, Flipkart                          | Medium       | Design, Hashing, Strings                         |
| 98  | Implement Autocomplete System                     | [GFG Autocomplete System](https://www.geeksforgeeks.org/implementing-autocomplete-system/)                                                                              | Amazon, Google, Swiggy                          | Hard         | Trie, Design, Strings                            |
| 99  | Design Twitter Feed                               | [LeetCode Design Twitter](https://leetcode.com/problems/design-twitter/)                                                                                               | Twitter, Flipkart, Ola                          | Medium       | Design, Heap, Linked List                        |
| 100 | Implement LFU Cache                                | [GFG LFU Cache](https://www.geeksforgeeks.org/lfu-cache-implementation/)                                                                                               | Amazon, Paytm, Flipkart                         | Hard         | Design, Hashing                                  |

---

## Questions asked in Google interview
- Two Number Sum
- Valid Parentheses
- Binary Search
- Merge Two Sorted Arrays
- Meeting Rooms
- Climbing Stairs
- Valid Anagram
- Longest Substring Without Repeating Characters
- Maximum Subarray (Kadane's Algorithm)
- Word Ladder
- 4Sum (Four Number Sum)
- Median of Two Sorted Arrays
- Longest Increasing Subsequence
- Longest Palindromic Substring
- Design LRU Cache
- Top K Frequent Elements
- Find Peak Element
- Candy (Min Rewards)
- Array of Products
- First Duplicate Value
- Validate Subsequence
- Nth Fibonacci
- Spiral Traverse
- Largest Range
- Diagonal Traverse
- Longest Peak
- Product Sum
- Merge Two Sorted Lists
- Binary Tree Level Order Traversal
- Longest Valid Parentheses
- Word Break
- Find Median from Data Stream
- Longest Repeating Character Replacement
- Kth Largest Element in an Array
- River Sizes
- Youngest Common Ancestor
- BST Construction
- Invert Binary Tree
- Validate BST
- Node Depths
- Branch Sums
- Find Successor
- Binary Tree Diameter
- Lowest Common Ancestor
- Dijkstra's Algorithm
- Topological Sort
- Knapsack Problem
- Disk Stacking
- Numbers In Pi
- Longest Common Subsequence
- Min Number of Jumps
- Water Area (Trapping Rain Water)
- Minimum Characters For Palindrome
- Regular Expression Matching
- Wildcard Matching
- Group Anagrams
- Longest Consecutive Sequence
- Maximum Product Subarray
- Sum of Two Integers (Bit Manipulation)
- Course Schedule
- Add Two Numbers (Linked List)
- Reverse Words in a String
- Intersection of Two Arrays
- Find All Duplicates in an Array
- Majority Element
- Rotate Array
- Spiral Matrix II
- Search in Rotated Sorted Array
- Implement Autocomplete System
- Design a File System
- Longest Substring with At Most K Distinct Characters
- Merge k Sorted Lists
- Longest Increasing Path in a Matrix
- Implement a Trie
- Implement a Sudoku Solver
- Find All Anagrams in a String
- Longest Palindromic Subsequence
- Clone Graph
- Design a Stack That Supports getMin()
- Convert Sorted Array to Binary Search Tree
- Meeting Rooms II

## Questions asked in Facebook interview
- Two Number Sum
- Reverse Linked List
- Valid Parentheses
- Binary Search
- Merge Two Sorted Arrays
- Climbing Stairs
- Longest Substring Without Repeating Characters
- Maximum Subarray (Kadane's Algorithm)
- Word Ladder
- 4Sum (Four Number Sum)
- Longest Increasing Subsequence
- Design LRU Cache
- Top K Frequent Elements
- Find Peak Element
- Candy (Min Rewards)
- Array of Products
- First Duplicate Value
- Word Break
- Spiral Traverse
- Diagonal Traverse
- Product Sum
- Merge Two Sorted Lists
- Binary Tree Level Order Traversal
- Longest Valid Parentheses
- Find Median from Data Stream
- Longest Repeating Character Replacement
- Kth Largest Element in an Array
- River Sizes
- BST Construction
- Invert Binary Tree
- Node Depths
- Branch Sums
- Find Successor
- Lowest Common Ancestor
- Dijkstra's Algorithm
- Knapsack Problem
- Disk Stacking
- Numbers In Pi
- Longest Common Subsequence
- Min Number of Jumps
- Water Area (Trapping Rain Water)
- Regular Expression Matching
- Wildcard Matching
- Group Anagrams
- Longest Consecutive Sequence
- Maximum Product Subarray
- Sum of Two Integers (Bit Manipulation)
- Add Two Numbers (Linked List)
- Reverse Words in a String
- Intersection of Two Arrays
- Find All Duplicates in an Array
- Rotate Array
- Spiral Matrix II
- Search in Rotated Sorted Array
- Design a Stack That Supports getMin()
- Convert Sorted Array to Binary Search Tree

## Questions asked in Amazon interview
- Two Number Sum
- Valid Parentheses
- Binary Search
- Merge Two Sorted Arrays
- Climbing Stairs
- Valid Anagram
- Longest Substring Without Repeating Characters
- Maximum Subarray (Kadane's Algorithm)
- Word Ladder
- 4Sum (Four Number Sum)
- Median of Two Sorted Arrays
- Longest Increasing Subsequence
- Longest Palindromic Substring
- Design LRU Cache
- Top K Frequent Elements
- Find Peak Element
- Candy (Min Rewards)
- Array of Products
- First Duplicate Value
- Validate Subsequence
- Nth Fibonacci
- Spiral Traverse
- Largest Range
- Diagonal Traverse
- Longest Peak
- Product Sum
- Merge Two Sorted Lists
- Binary Tree Level Order Traversal
- Longest Valid Parentheses
- Word Break
- Find Median from Data Stream
- Longest Repeating Character Replacement
- Kth Largest Element in an Array
- River Sizes
- BST Construction
- Invert Binary Tree
- Validate BST
- Branch Sums
- Find Successor
- Lowest Common Ancestor
- Dijkstra's Algorithm
- Topological Sort
- Knapsack Problem
- Disk Stacking
- Numbers In Pi
- Longest Common Subsequence
- Min Number of Jumps
- Water Area (Trapping Rain Water)
- Minimum Characters For Palindrome
- Regular Expression Matching
- Wildcard Matching
- Group Anagrams
- Longest Consecutive Sequence
- Maximum Product Subarray
- Sum of Two Integers (Bit Manipulation)
- Course Schedule
- Add Two Numbers (Linked List)
- Reverse Words in a String
- Intersection of Two Arrays
- Find All Duplicates in an Array
- Majority Element
- Rotate Array
- Spiral Matrix II
- Search in Rotated Sorted Array
- Design a URL Shortener
- Implement Autocomplete System
- Design a File System
- Longest Substring with At Most K Distinct Characters
- Merge k Sorted Lists
- Implement a Trie
- Implement a Sudoku Solver
- Find All Anagrams in a String
- Longest Palindromic Subsequence
- Clone Graph
- Design a Stack That Supports getMin()
- Meeting Rooms II

## Questions asked in Microsoft interview
- Reverse Linked List
- Merge Two Sorted Arrays
- Meeting Rooms
- Median of Two Sorted Arrays
- Nth Fibonacci
- Binary Tree Level Order Traversal
- Find Median from Data Stream
- Topological Sort
- Youngest Common Ancestor
- Longest Increasing Path in a Matrix
- Implement a Trie
- Implement a Sudoku Solver
- Convert Sorted Array to Binary Search Tree
- Meeting Rooms II
- Course Schedule

## Questions asked in Uber interview
- Subarray Sort
- Longest Peak
- Binary Tree Diameter
- Design Twitter-like Feed

## Questions asked in Swiggy interview
- Implement Autocomplete System
- Design a Rate Limiter
- Serialize and Deserialize Binary Tree

## Questions asked in Flipkart interview
- Design a URL Shortener
- Design Twitter Feed
- Design a File System
- Subarray Sum Equals K
- Design an Elevator System

## Questions asked in Ola interview
- Design Twitter Feed
- Design an Elevator System

## Questions asked in Paytm interview
- Implement LFU Cache
- Design a Data Structure for the Stock Span Problem
- Implement Magic Dictionary

## Questions asked in OYO interview
- Design an Elevator System

## Questions asked in WhatsApp interview
- Design a Chat System

## Questions asked in Slack interview
- Design a Chat System

## Questions asked in Airbnb interview
- Design a URL Shortener
