---
title: SQL Interview Questions for Data Science
description: 100+ SQL interview questions for cracking Data Science and Analytics interviews
---

# SQL Interview Questions for Data Science

This comprehensive guide contains **100+ SQL interview questions** commonly asked at top tech companies like Google, Amazon, Meta, Microsoft, and Netflix. Each premium question includes detailed explanations, query examples, and interviewer insights.

---

## Premium Interview Questions

Master these frequently asked SQL questions with detailed explanations, query examples, and insights into what interviewers really look for.

---

### What is the Difference Between WHERE and HAVING Clauses? - Google, Amazon, Meta Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Filtering`, `Aggregation`, `SQL Basics` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **The Core Difference:**
    
    | Clause | Filters | When Applied | Works With |
    |--------|---------|--------------|------------|
    | WHERE | Individual rows | Before grouping | All columns |
    | HAVING | Groups/aggregates | After GROUP BY | Aggregate functions |
    
    **Execution Order:**
    
    ```sql
    FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY
    ```
    
    **Example Comparison:**
    
    ```sql
    -- Find departments with more than 5 employees earning > $50K
    
    -- WHERE filters rows BEFORE grouping
    SELECT department, COUNT(*) as emp_count
    FROM employees
    WHERE salary > 50000      -- Filters individual rows first
    GROUP BY department
    HAVING COUNT(*) > 5;       -- Filters groups after
    
    -- What actually happens:
    -- 1. FROM: Select all rows from employees
    -- 2. WHERE: Keep only rows where salary > 50000
    -- 3. GROUP BY: Group remaining rows by department
    -- 4. HAVING: Keep only groups with count > 5
    -- 5. SELECT: Return department and count
    ```
    
    **Common Pattern: Filtering with Aggregates**
    
    ```sql
    -- ❌ WRONG: Can't use aggregate in WHERE
    SELECT customer_id, SUM(order_total) as total_spent
    FROM orders
    WHERE SUM(order_total) > 1000  -- ERROR!
    GROUP BY customer_id;
    
    -- ✅ CORRECT: Use HAVING for aggregates
    SELECT customer_id, SUM(order_total) as total_spent
    FROM orders
    GROUP BY customer_id
    HAVING SUM(order_total) > 1000;  -- Works!
    ```
    
    **Performance Tip:**
    
    ```sql
    -- ✅ More efficient: Filter with WHERE when possible
    SELECT category, AVG(price) as avg_price
    FROM products
    WHERE active = 1              -- Filter early (uses index)
    GROUP BY category
    HAVING AVG(price) > 100;      -- Filter aggregated result
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of SQL execution order.
        
        **Strong answer signals:**
        
        - Draws the execution order diagram
        - Knows WHERE filters rows, HAVING filters groups
        - Explains why aggregates can't be in WHERE
        - Mentions performance: "WHERE is more efficient when applicable"

---

### Explain Different Types of JOINs with Examples - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `JOINs`, `SQL Fundamentals`, `Data Modeling` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **JOIN Types Overview:**
    
    | JOIN Type | Returns | NULL Behavior |
    |-----------|---------|---------------|
    | INNER | Matching rows only | Excludes NULLs |
    | LEFT | All left + matching right | Right columns NULL if no match |
    | RIGHT | All right + matching left | Left columns NULL if no match |
    | FULL OUTER | All rows from both | NULLs for non-matches |
    | CROSS | Cartesian product | Every combination |
    
    **Visual Representation:**
    
    ```
    Tables:
    employees (id, name, dept_id)    departments (id, name)
    ┌─────┬───────┬─────────┐        ┌────┬────────────┐
    │ 1   │ Alice │ 10      │        │ 10 │ Engineering│
    │ 2   │ Bob   │ 20      │        │ 20 │ Marketing  │
    │ 3   │ Carol │ NULL    │        │ 30 │ HR         │
    └─────┴───────┴─────────┘        └────┴────────────┘
    ```
    
    ```sql
    -- INNER JOIN: Only matching rows (Alice, Bob)
    SELECT e.name, d.name as department
    FROM employees e
    INNER JOIN departments d ON e.dept_id = d.id;
    -- Result: Alice-Engineering, Bob-Marketing
    
    -- LEFT JOIN: All employees + matching departments
    SELECT e.name, d.name as department
    FROM employees e
    LEFT JOIN departments d ON e.dept_id = d.id;
    -- Result: Alice-Engineering, Bob-Marketing, Carol-NULL
    
    -- RIGHT JOIN: All departments + matching employees
    SELECT e.name, d.name as department
    FROM employees e
    RIGHT JOIN departments d ON e.dept_id = d.id;
    -- Result: Alice-Engineering, Bob-Marketing, NULL-HR
    
    -- FULL OUTER JOIN: All from both (MySQL uses UNION)
    SELECT e.name, d.name as department
    FROM employees e
    FULL OUTER JOIN departments d ON e.dept_id = d.id;
    -- Result: All 4 combinations including NULLs
    
    -- CROSS JOIN: Cartesian product
    SELECT e.name, d.name
    FROM employees e
    CROSS JOIN departments d;
    -- Result: 3 × 3 = 9 rows (every combination)
    ```
    
    **Self JOIN Example:**
    
    ```sql
    -- Find employees and their managers
    SELECT 
        e.name as employee,
        m.name as manager
    FROM employees e
    LEFT JOIN employees m ON e.manager_id = m.id;
    ```
    
    **Anti-JOIN Pattern (Find missing):**
    
    ```sql
    -- Find employees without departments
    SELECT e.name
    FROM employees e
    LEFT JOIN departments d ON e.dept_id = d.id
    WHERE d.id IS NULL;  -- Anti-join pattern
    
    -- Alternative using NOT EXISTS
    SELECT e.name
    FROM employees e
    WHERE NOT EXISTS (
        SELECT 1 FROM departments d 
        WHERE d.id = e.dept_id
    );
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Fundamental SQL understanding and problem-solving.
        
        **Strong answer signals:**
        
        - Can draw Venn diagrams for each JOIN type
        - Knows LEFT vs RIGHT is just table order
        - Shows anti-join pattern for "find missing" problems
        - Mentions execution plans: "JOINs on indexed columns are efficient"

---

### What Are Window Functions? Explain with Examples - Google, Meta, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Window Functions`, `Analytics`, `Advanced SQL` | **Asked by:** Google, Meta, Netflix, Amazon, Microsoft

??? success "View Answer"

    **What Are Window Functions?**
    
    Window functions perform calculations across a set of rows related to the current row, without collapsing them (unlike GROUP BY).
    
    **Syntax:**
    
    ```sql
    function_name() OVER (
        PARTITION BY partition_columns
        ORDER BY order_columns
        ROWS/RANGE frame_specification
    )
    ```
    
    **Common Window Functions:**
    
    | Function | Purpose |
    |----------|---------|
    | ROW_NUMBER() | Sequential numbering |
    | RANK() | Ranking with gaps |
    | DENSE_RANK() | Ranking without gaps |
    | LAG()/LEAD() | Access previous/next rows |
    | SUM()/AVG() OVER | Running totals/averages |
    | FIRST_VALUE()/LAST_VALUE() | First/last in window |
    | NTILE(n) | Divide into n buckets |
    
    ```sql
    -- Sample: sales data
    SELECT 
        date,
        product,
        amount,
        
        -- Running total
        SUM(amount) OVER (ORDER BY date) as running_total,
        
        -- Running total per product
        SUM(amount) OVER (
            PARTITION BY product 
            ORDER BY date
        ) as product_running_total,
        
        -- Rank within each product
        RANK() OVER (
            PARTITION BY product 
            ORDER BY amount DESC
        ) as rank_in_product,
        
        -- Previous day's amount
        LAG(amount, 1, 0) OVER (ORDER BY date) as prev_day,
        
        -- Difference from previous
        amount - LAG(amount, 1, 0) OVER (ORDER BY date) as day_diff,
        
        -- Moving average (3-day)
        AVG(amount) OVER (
            ORDER BY date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as moving_avg_3d
        
    FROM sales;
    ```
    
    **Practical Examples:**
    
    ```sql
    -- 1. Top N per group (Top 3 products per category)
    WITH ranked AS (
        SELECT 
            category,
            product_name,
            revenue,
            ROW_NUMBER() OVER (
                PARTITION BY category 
                ORDER BY revenue DESC
            ) as rn
        FROM products
    )
    SELECT * FROM ranked WHERE rn <= 3;
    
    -- 2. Calculate Month-over-Month growth
    SELECT 
        month,
        revenue,
        LAG(revenue) OVER (ORDER BY month) as prev_month,
        ROUND(
            (revenue - LAG(revenue) OVER (ORDER BY month)) 
            / LAG(revenue) OVER (ORDER BY month) * 100, 
        2) as mom_growth_pct
    FROM monthly_sales;
    
    -- 3. Percentile/Quartile analysis
    SELECT 
        customer_id,
        total_spent,
        NTILE(4) OVER (ORDER BY total_spent) as spending_quartile
    FROM customer_totals;
    
    -- 4. Identify consecutive sequences
    SELECT 
        date,
        event,
        date - ROW_NUMBER() OVER (ORDER BY date)::int as grp
    FROM events
    -- Same 'grp' value = consecutive dates
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced SQL skills crucial for analytics roles.
        
        **Strong answer signals:**
        
        - Knows difference: ROW_NUMBER vs RANK vs DENSE_RANK
        - Can explain PARTITION BY vs GROUP BY
        - Uses LAG/LEAD for time-series analysis
        - Mentions frame specifications: ROWS vs RANGE

---

### Write a Query to Find Duplicate Records - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Quality`, `Aggregation`, `Common Patterns` | **Asked by:** Amazon, Google, Meta, Microsoft

??? success "View Answer"

    **Multiple Approaches:**
    
    ```sql
    -- Sample table with duplicates
    -- users: id, email, name, created_at
    
    -- Method 1: GROUP BY + HAVING (Most common)
    SELECT email, COUNT(*) as count
    FROM users
    GROUP BY email
    HAVING COUNT(*) > 1;
    
    -- Method 2: Get all duplicate rows with details
    SELECT *
    FROM users
    WHERE email IN (
        SELECT email
        FROM users
        GROUP BY email
        HAVING COUNT(*) > 1
    );
    
    -- Method 3: Using ROW_NUMBER (Modern approach)
    WITH duplicates AS (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY email 
                ORDER BY created_at
            ) as rn
        FROM users
    )
    SELECT * FROM duplicates WHERE rn > 1;  -- All duplicates (not first)
    
    -- Method 4: Self-JOIN (older but works everywhere)
    SELECT DISTINCT u1.*
    FROM users u1
    INNER JOIN users u2 
        ON u1.email = u2.email 
        AND u1.id != u2.id;
    ```
    
    **Deleting Duplicates (Keep first/oldest):**
    
    ```sql
    -- Method 1: DELETE with ROW_NUMBER (PostgreSQL, SQL Server)
    WITH duplicates AS (
        SELECT 
            id,
            ROW_NUMBER() OVER (
                PARTITION BY email 
                ORDER BY created_at
            ) as rn
        FROM users
    )
    DELETE FROM users
    WHERE id IN (
        SELECT id FROM duplicates WHERE rn > 1
    );
    
    -- Method 2: DELETE with self-join (MySQL)
    DELETE u1
    FROM users u1
    INNER JOIN users u2
        ON u1.email = u2.email
        AND u1.id > u2.id;  -- Keep lower id
    
    -- Method 3: Keep newest instead of oldest
    WITH duplicates AS (
        SELECT 
            id,
            ROW_NUMBER() OVER (
                PARTITION BY email 
                ORDER BY created_at DESC  -- DESC = keep newest
            ) as rn
        FROM users
    )
    DELETE FROM users
    WHERE id IN (
        SELECT id FROM duplicates WHERE rn > 1
    );
    ```
    
    **Multi-column Duplicates:**
    
    ```sql
    -- Duplicates based on multiple columns
    SELECT first_name, last_name, email, COUNT(*)
    FROM users
    GROUP BY first_name, last_name, email
    HAVING COUNT(*) > 1;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical SQL problem-solving.
        
        **Strong answer signals:**
        
        - Gives multiple approaches (GROUP BY, ROW_NUMBER, self-join)
        - Knows how to DELETE duplicates while keeping one
        - Asks clarifying questions: "Keep oldest or newest?"
        - Mentions performance: "Add index on the grouping column"

---

### Explain UNION vs UNION ALL - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Set Operations`, `Performance`, `SQL Basics` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **The Core Difference:**
    
    | Operation | Duplicates | Performance |
    |-----------|------------|-------------|
    | UNION | Removes duplicates | Slower (sorts/dedupes) |
    | UNION ALL | Keeps all rows | Faster (no dedup) |
    
    ```sql
    -- Table A: (1, 2, 3)
    -- Table B: (2, 3, 4)
    
    -- UNION: Removes duplicates
    SELECT x FROM table_a
    UNION
    SELECT x FROM table_b;
    -- Result: 1, 2, 3, 4 (4 rows)
    
    -- UNION ALL: Keeps everything
    SELECT x FROM table_a
    UNION ALL
    SELECT x FROM table_b;
    -- Result: 1, 2, 3, 2, 3, 4 (6 rows)
    ```
    
    **When to Use Which:**
    
    | Scenario | Use |
    |----------|-----|
    | Need unique results | UNION |
    | Guaranteed no duplicates | UNION ALL (faster) |
    | Counting all occurrences | UNION ALL |
    | Combining partitioned data | UNION ALL |
    
    ```sql
    -- Performance Example: Combining monthly data
    
    -- ❌ Slow: UNION sorts and dedupes unnecessarily
    SELECT * FROM sales_jan
    UNION
    SELECT * FROM sales_feb
    UNION
    SELECT * FROM sales_mar;
    
    -- ✅ Fast: No duplicates between months anyway
    SELECT * FROM sales_jan
    UNION ALL
    SELECT * FROM sales_feb
    UNION ALL
    SELECT * FROM sales_mar;
    ```
    
    **Requirements for UNION:**
    
    1. Same number of columns
    2. Compatible data types
    3. Column names from first query
    
    ```sql
    -- Column alignment matters
    SELECT name, age FROM employees  -- 2 columns
    UNION
    SELECT product_name, price FROM products;  -- 2 columns ✓
    
    -- Types must be compatible
    SELECT id, name FROM users      -- int, varchar
    UNION
    SELECT id, description FROM items; -- int, varchar ✓
    ```
    
    **Other Set Operations:**
    
    ```sql
    -- INTERSECT: Common rows only
    SELECT email FROM customers
    INTERSECT
    SELECT email FROM newsletter_subscribers;
    
    -- EXCEPT/MINUS: In first but not second
    SELECT email FROM customers
    EXCEPT
    SELECT email FROM unsubscribed;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of set operations and performance.
        
        **Strong answer signals:**
        
        - Immediately mentions performance difference
        - Gives example: "Use UNION ALL for partitioned tables"
        - Knows INTERSECT and EXCEPT as well
        - Mentions: "UNION requires sorting, UNION ALL doesn't"

---

### What is a Common Table Expression (CTE)? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `CTEs`, `Readability`, `Recursion` | **Asked by:** Google, Meta, Netflix, Amazon, Microsoft

??? success "View Answer"

    **What is a CTE?**
    
    A Common Table Expression (CTE) is a temporary named result set that exists only within the scope of a single query. It improves readability and enables recursive queries.
    
    **Basic Syntax:**
    
    ```sql
    WITH cte_name AS (
        -- CTE query
        SELECT ...
    )
    SELECT * FROM cte_name;
    ```
    
    **Benefits:**
    
    | Benefit | Description |
    |---------|-------------|
    | Readability | Break complex queries into logical steps |
    | Reusability | Reference same result multiple times |
    | Recursion | Enable hierarchical queries |
    | Maintainability | Easier to debug and modify |
    
    ```sql
    -- Example: Complex query made readable
    
    -- Without CTE (hard to read)
    SELECT * FROM (
        SELECT customer_id, SUM(amount) as total
        FROM orders
        WHERE order_date >= '2024-01-01'
        GROUP BY customer_id
    ) customer_totals
    WHERE total > (
        SELECT AVG(total) FROM (
            SELECT customer_id, SUM(amount) as total
            FROM orders
            WHERE order_date >= '2024-01-01'
            GROUP BY customer_id
        ) all_totals
    );
    
    -- With CTE (much clearer!)
    WITH customer_totals AS (
        SELECT 
            customer_id, 
            SUM(amount) as total
        FROM orders
        WHERE order_date >= '2024-01-01'
        GROUP BY customer_id
    ),
    avg_total AS (
        SELECT AVG(total) as avg_val
        FROM customer_totals
    )
    SELECT ct.*
    FROM customer_totals ct, avg_total at
    WHERE ct.total > at.avg_val;
    ```
    
    **Recursive CTE (Hierarchical Data):**
    
    ```sql
    -- Employee hierarchy (org chart)
    WITH RECURSIVE org_chart AS (
        -- Base case: CEO (no manager)
        SELECT id, name, manager_id, 1 as level
        FROM employees
        WHERE manager_id IS NULL
        
        UNION ALL
        
        -- Recursive case: employees with managers
        SELECT e.id, e.name, e.manager_id, oc.level + 1
        FROM employees e
        INNER JOIN org_chart oc ON e.manager_id = oc.id
    )
    SELECT * FROM org_chart ORDER BY level, name;
    
    -- Generate date series
    WITH RECURSIVE dates AS (
        SELECT DATE '2024-01-01' as date
        UNION ALL
        SELECT date + INTERVAL '1 day'
        FROM dates
        WHERE date < '2024-12-31'
    )
    SELECT * FROM dates;
    ```
    
    **Multiple CTEs:**
    
    ```sql
    WITH 
    active_users AS (
        SELECT user_id FROM users WHERE active = true
    ),
    user_orders AS (
        SELECT user_id, COUNT(*) as order_count
        FROM orders
        GROUP BY user_id
    ),
    user_metrics AS (
        SELECT 
            au.user_id,
            COALESCE(uo.order_count, 0) as orders
        FROM active_users au
        LEFT JOIN user_orders uo ON au.user_id = uo.user_id
    )
    SELECT * FROM user_metrics WHERE orders > 10;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Modern SQL skills and code organization.
        
        **Strong answer signals:**
        
        - Uses CTEs for readability, not just because they exist
        - Knows recursive CTEs for hierarchies (org charts, trees)
        - Mentions: "CTEs may be materialized or inline depending on DB"
        - Compares to subqueries/temp tables when appropriate

---

### Write a Query to Calculate Running Total/Cumulative Sum - Netflix, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Window Functions`, `Running Total`, `Analytics` | **Asked by:** Netflix, Amazon, Google, Meta

??? success "View Answer"

    **Using Window Functions (Modern Approach):**
    
    ```sql
    -- Running total of daily sales
    SELECT 
        date,
        daily_revenue,
        SUM(daily_revenue) OVER (ORDER BY date) as running_total
    FROM daily_sales;
    
    -- Running total with partitions (per category)
    SELECT 
        date,
        category,
        revenue,
        SUM(revenue) OVER (
            PARTITION BY category 
            ORDER BY date
        ) as category_running_total
    FROM sales;
    
    -- Running total with explicit frame
    SELECT 
        date,
        amount,
        SUM(amount) OVER (
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as running_total
    FROM transactions;
    ```
    
    **Other Running Calculations:**
    
    ```sql
    SELECT 
        date,
        amount,
        
        -- Running average
        AVG(amount) OVER (ORDER BY date) as running_avg,
        
        -- Running count
        COUNT(*) OVER (ORDER BY date) as running_count,
        
        -- Running max
        MAX(amount) OVER (ORDER BY date) as running_max,
        
        -- 7-day rolling sum
        SUM(amount) OVER (
            ORDER BY date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as rolling_7day_sum,
        
        -- Percentage of running total
        amount / SUM(amount) OVER (ORDER BY date) * 100 as pct_of_total
        
    FROM daily_metrics;
    ```
    
    **Self-JOIN Approach (Legacy/MySQL < 8):**
    
    ```sql
    -- Running total without window functions
    SELECT 
        t1.date,
        t1.amount,
        SUM(t2.amount) as running_total
    FROM transactions t1
    INNER JOIN transactions t2 
        ON t2.date <= t1.date
    GROUP BY t1.date, t1.amount
    ORDER BY t1.date;
    ```
    
    **Correlated Subquery (Another Legacy Option):**
    
    ```sql
    SELECT 
        date,
        amount,
        (
            SELECT SUM(amount)
            FROM transactions t2
            WHERE t2.date <= t1.date
        ) as running_total
    FROM transactions t1
    ORDER BY date;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Analytics SQL skills.
        
        **Strong answer signals:**
        
        - Uses window functions as first choice
        - Knows PARTITION BY for grouped running totals
        - Can explain frame specifications (ROWS vs RANGE)
        - Knows legacy approaches for older MySQL versions

---

### Explain the Difference Between DELETE, TRUNCATE, and DROP - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `DDL`, `DML`, `Data Management` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **Quick Comparison:**
    
    | Aspect | DELETE | TRUNCATE | DROP |
    |--------|--------|----------|------|
    | Type | DML | DDL | DDL |
    | Removes | Rows (filtered) | All rows | Entire table |
    | WHERE clause | Yes | No | No |
    | Rollback | Yes | No (usually) | No |
    | Triggers | Fires | No | No |
    | Speed | Slowest | Fast | Fastest |
    | Space | Keeps allocated | Releases | Releases all |
    | Identity/Auto-inc | Continues | Resets | N/A |
    
    ```sql
    -- DELETE: Remove specific rows (logged, triggers fire)
    DELETE FROM orders WHERE status = 'cancelled';
    DELETE FROM orders;  -- All rows, but can rollback
    
    -- TRUNCATE: Remove all rows quickly (DDL, no triggers)
    TRUNCATE TABLE temp_data;
    -- Cannot use WHERE
    -- Auto-increment resets to 1
    
    -- DROP: Remove entire table structure
    DROP TABLE old_logs;
    -- Table definition, indexes, constraints all gone
    ```
    
    **When to Use Each:**
    
    | Scenario | Use |
    |----------|-----|
    | Remove specific rows | DELETE |
    | Clear table for reload | TRUNCATE |
    | Remove table entirely | DROP |
    | Need to undo | DELETE (in transaction) |
    | Fastest wipe | TRUNCATE |
    | Keep table structure | DELETE or TRUNCATE |
    
    **Transaction Behavior:**
    
    ```sql
    -- DELETE can be rolled back
    BEGIN TRANSACTION;
    DELETE FROM important_data WHERE id = 123;
    -- Oops! Wrong ID
    ROLLBACK;  -- Data restored
    
    -- TRUNCATE typically cannot (varies by DB)
    BEGIN TRANSACTION;
    TRUNCATE TABLE staging;
    COMMIT;  -- Can't undo in most databases
    ```
    
    **Performance Difference:**
    
    ```sql
    -- DELETE (10M rows) - SLOW
    -- Writes to transaction log for each row
    -- Fires triggers for each row
    -- Acquires row-level locks
    DELETE FROM huge_table;  -- Minutes to hours
    
    -- TRUNCATE (10M rows) - FAST
    -- Deallocates data pages directly
    -- No per-row logging
    -- Table-level lock only
    TRUNCATE TABLE huge_table;  -- Seconds
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of SQL operations and their implications.
        
        **Strong answer signals:**
        
        - Knows DELETE is DML, TRUNCATE/DROP are DDL
        - Mentions transaction log and performance
        - Knows identity/auto-increment behavior
        - Asks: "Do you need to keep the table structure?"

---

### What Are Indexes? When Should You Use Them? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Indexing`, `Performance`, `Database Design` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **What is an Index?**
    
    An index is a data structure (usually B-tree or Hash) that speeds up data retrieval at the cost of additional storage and slower writes.
    
    **Types of Indexes:**
    
    | Type | Use Case |
    |------|----------|
    | B-tree | Range queries, sorting, equality |
    | Hash | Exact match only (=) |
    | Clustered | Physical row order (one per table) |
    | Non-clustered | Pointer to rows (many per table) |
    | Composite | Multiple columns |
    | Covering | Includes all query columns |
    | Partial | Subset of rows |
    
    ```sql
    -- Create basic index
    CREATE INDEX idx_email ON users(email);
    
    -- Composite index (order matters!)
    CREATE INDEX idx_status_date ON orders(status, order_date);
    
    -- Unique index
    CREATE UNIQUE INDEX idx_unique_email ON users(email);
    
    -- Partial index (PostgreSQL)
    CREATE INDEX idx_active_users ON users(email) WHERE active = true;
    
    -- Covering index (includes extra columns)
    CREATE INDEX idx_covering ON orders(customer_id) INCLUDE (order_date, total);
    ```
    
    **When to Add Indexes:**
    
    | Add Index For | Avoid Index For |
    |---------------|-----------------|
    | Frequent WHERE clauses | Frequently updated columns |
    | JOIN columns | Low-cardinality columns |
    | ORDER BY columns | Small tables |
    | Foreign keys | Write-heavy tables |
    
    **Query Analysis:**
    
    ```sql
    -- Check if index is used
    EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';
    
    -- Without index: Seq Scan (full table scan)
    -- With index: Index Scan (fast lookup)
    
    -- Check existing indexes
    SELECT * FROM pg_indexes WHERE tablename = 'users';
    ```
    
    **Composite Index Column Order:**
    
    ```sql
    -- Index: (status, order_date)
    
    -- ✅ Uses index (left-most columns)
    SELECT * FROM orders WHERE status = 'shipped';
    SELECT * FROM orders WHERE status = 'shipped' AND order_date > '2024-01-01';
    
    -- ❌ Doesn't use index (skips leftmost column)
    SELECT * FROM orders WHERE order_date > '2024-01-01';
    ```
    
    **Index Anti-patterns:**
    
    ```sql
    -- ❌ Function on indexed column breaks index
    SELECT * FROM users WHERE LOWER(email) = 'test@example.com';
    -- Fix: CREATE INDEX idx_email_lower ON users(LOWER(email));
    
    -- ❌ Leading wildcard breaks index
    SELECT * FROM products WHERE name LIKE '%phone';
    -- ✅ Trailing wildcard uses index
    SELECT * FROM products WHERE name LIKE 'phone%';
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Database performance understanding.
        
        **Strong answer signals:**
        
        - Knows trade-off: faster reads, slower writes
        - Understands composite index column ordering
        - Can read EXPLAIN output
        - Mentions: "Indexes aren't free, they cost storage and write performance"

---

### Write a Query to Find the Second Highest Salary - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Subqueries`, `Ranking`, `Common Patterns` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **Multiple Approaches:**
    
    ```sql
    -- Method 1: LIMIT OFFSET (Simple)
    SELECT DISTINCT salary
    FROM employees
    ORDER BY salary DESC
    LIMIT 1 OFFSET 1;
    
    -- Method 2: Subquery with MAX
    SELECT MAX(salary) as second_highest
    FROM employees
    WHERE salary < (SELECT MAX(salary) FROM employees);
    
    -- Method 3: DENSE_RANK (Handles ties correctly)
    WITH ranked AS (
        SELECT 
            salary,
            DENSE_RANK() OVER (ORDER BY salary DESC) as rnk
        FROM employees
    )
    SELECT DISTINCT salary
    FROM ranked
    WHERE rnk = 2;
    
    -- Method 4: NOT IN (Exclude max)
    SELECT MAX(salary)
    FROM employees
    WHERE salary NOT IN (
        SELECT MAX(salary) FROM employees
    );
    ```
    
    **Generalized: Nth Highest Salary:**
    
    ```sql
    -- Find Nth highest (parameter = N)
    WITH ranked AS (
        SELECT 
            salary,
            DENSE_RANK() OVER (ORDER BY salary DESC) as rnk
        FROM employees
    )
    SELECT DISTINCT salary
    FROM ranked
    WHERE rnk = @N;  -- Replace @N with desired rank
    
    -- Find 3rd highest
    SELECT salary
    FROM employees e1
    WHERE 3 = (
        SELECT COUNT(DISTINCT salary)
        FROM employees e2
        WHERE e2.salary >= e1.salary
    );
    ```
    
    **Handle NULL/Empty Cases:**
    
    ```sql
    -- Returns NULL if no second highest exists
    SELECT (
        SELECT DISTINCT salary
        FROM employees
        ORDER BY salary DESC
        LIMIT 1 OFFSET 1
    ) as second_highest;
    
    -- With COALESCE for default
    SELECT COALESCE(
        (SELECT DISTINCT salary
         FROM employees
         ORDER BY salary DESC
         LIMIT 1 OFFSET 1),
        0
    ) as second_highest;
    ```
    
    **Per-Department Second Highest:**
    
    ```sql
    WITH ranked AS (
        SELECT 
            department,
            salary,
            DENSE_RANK() OVER (
                PARTITION BY department 
                ORDER BY salary DESC
            ) as rnk
        FROM employees
    )
    SELECT department, salary as second_highest
    FROM ranked
    WHERE rnk = 2;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** SQL problem-solving and edge case handling.
        
        **Strong answer signals:**
        
        - Gives multiple solutions (OFFSET, subquery, window function)
        - Uses DENSE_RANK for handling ties
        - Handles NULL case when N rows don't exist
        - Can generalize to Nth highest

---

### Explain ACID Properties in Databases - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Transactions`, `Database Theory`, `Reliability` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **ACID Properties:**
    
    | Property | Meaning | Ensures |
    |----------|---------|---------|
    | **A**tomicity | All or nothing | Transaction completes fully or not at all |
    | **C**onsistency | Valid state to valid state | Database constraints maintained |
    | **I**solation | Concurrent transactions independent | No interference between transactions |
    | **D**urability | Permanent once committed | Data survives crashes |
    
    **Atomicity Example:**
    
    ```sql
    -- Bank transfer must be atomic
    BEGIN TRANSACTION;
    
    UPDATE accounts SET balance = balance - 100 WHERE id = 1;  -- Debit
    UPDATE accounts SET balance = balance + 100 WHERE id = 2;  -- Credit
    
    -- If either fails, ROLLBACK both
    COMMIT;  -- Both succeed or neither
    
    -- If error occurs:
    ROLLBACK;  -- Undo partial changes
    ```
    
    **Consistency Example:**
    
    ```sql
    -- Constraint: balance >= 0
    ALTER TABLE accounts ADD CONSTRAINT positive_balance 
        CHECK (balance >= 0);
    
    -- This will fail (maintains consistency)
    UPDATE accounts SET balance = balance - 1000 
    WHERE id = 1 AND balance = 500;
    -- Error: violates check constraint
    ```
    
    **Isolation Levels:**
    
    | Level | Dirty Read | Non-repeatable | Phantom |
    |-------|------------|----------------|---------|
    | Read Uncommitted | Yes | Yes | Yes |
    | Read Committed | No | Yes | Yes |
    | Repeatable Read | No | No | Yes |
    | Serializable | No | No | No |
    
    ```sql
    -- Set isolation level
    SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
    
    -- Serializable (strictest)
    SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
    BEGIN;
    SELECT * FROM accounts WHERE id = 1;
    -- Another transaction cannot modify this row until COMMIT
    COMMIT;
    ```
    
    **Durability Example:**
    
    ```sql
    BEGIN TRANSACTION;
    INSERT INTO orders (customer_id, total) VALUES (1, 500);
    COMMIT;
    -- Power failure happens immediately after COMMIT
    -- Data is STILL persisted (written to disk/WAL)
    ```
    
    **Trade-offs:**
    
    | Property | Cost |
    |----------|------|
    | Atomicity | Logging overhead |
    | Consistency | Constraint checking overhead |
    | Isolation | Lock contention, deadlocks |
    | Durability | fsync/write performance |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Database fundamentals and trade-offs.
        
        **Strong answer signals:**
        
        - Gives real examples for each property
        - Knows isolation levels and their trade-offs
        - Mentions: "NoSQL often relaxes ACID for scalability"
        - Can discuss CAP theorem connection

---

### What is Database Normalization? Explain Normal Forms - Amazon, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Database Design`, `Normalization`, `Data Modeling` | **Asked by:** Amazon, Microsoft, Google, Meta

??? success "View Answer"

    **What is Normalization?**
    
    Normalization organizes database tables to reduce redundancy and improve data integrity by dividing data into related tables.
    
    **Normal Forms Summary:**
    
    | Form | Rule | Eliminates |
    |------|------|------------|
    | 1NF | Atomic values, no repeating groups | Multi-valued attributes |
    | 2NF | 1NF + No partial dependencies | Composite key issues |
    | 3NF | 2NF + No transitive dependencies | Non-key dependencies |
    | BCNF | Every determinant is a candidate key | More anomalies |
    
    **1NF Example:**
    
    ```sql
    -- ❌ Violates 1NF (repeating group)
    students
    | id | name  | phones                |
    |----|-------|-----------------------|
    | 1  | Alice | 555-1111, 555-2222    |
    
    -- ✅ 1NF (atomic values)
    students             student_phones
    | id | name  |       | student_id | phone    |
    |----|-------|       |------------|----------|
    | 1  | Alice |       | 1          | 555-1111 |
                         | 1          | 555-2222 |
    ```
    
    **2NF Example:**
    
    ```sql
    -- ❌ Violates 2NF (partial dependency on composite key)
    -- Key: (student_id, course_id), but student_name depends only on student_id
    enrollments
    | student_id | course_id | student_name | grade |
    |------------|-----------|--------------|-------|
    
    -- ✅ 2NF (separate tables)
    students              enrollments
    | id | name  |        | student_id | course_id | grade |
    |----|-------|        |------------|-----------|-------|
    | 1  | Alice |        | 1          | 101       | A     |
    ```
    
    **3NF Example:**
    
    ```sql
    -- ❌ Violates 3NF (transitive dependency)
    -- zip → city (non-key depends on non-key)
    employees
    | id | name  | zip   | city    |
    |----|-------|-------|---------|
    | 1  | Alice | 10001 | New York|
    
    -- ✅ 3NF (remove transitive dependency)
    employees             zipcodes
    | id | name  | zip   |  | zip   | city     |
    |----|-------|-------|  |-------|----------|
    | 1  | Alice | 10001 |  | 10001 | New York |
    ```
    
    **When to Denormalize:**
    
    | Normalize | Denormalize |
    |-----------|-------------|
    | OLTP (transactions) | OLAP (analytics) |
    | High write volume | High read volume |
    | Data integrity critical | Query performance critical |
    | Storage is expensive | Storage is cheap |
    
    ```sql
    -- Denormalized for reporting (star schema)
    sales_fact
    | sale_id | date | product_name | category | customer_name | amount |
    -- Redundant but fast for aggregations
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Database design principles.
        
        **Strong answer signals:**
        
        - Explains each normal form with examples
        - Knows when to denormalize (analytics, reporting)
        - Mentions: "3NF is usually sufficient for OLTP"
        - Discusses trade-offs: integrity vs performance

---

### Write a Query to Pivot/Unpivot Data - Google, Meta, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Pivoting`, `Data Transformation`, `Advanced SQL` | **Asked by:** Google, Meta, Netflix, Amazon

??? success "View Answer"

    **Pivot: Rows to Columns**
    
    ```sql
    -- Original data
    -- sales: product, quarter, amount
    -- | Product | Q1    | Q2    | Q3    | Q4    |
    
    -- Method 1: CASE + GROUP BY (Works everywhere)
    SELECT 
        product,
        SUM(CASE WHEN quarter = 'Q1' THEN amount ELSE 0 END) as Q1,
        SUM(CASE WHEN quarter = 'Q2' THEN amount ELSE 0 END) as Q2,
        SUM(CASE WHEN quarter = 'Q3' THEN amount ELSE 0 END) as Q3,
        SUM(CASE WHEN quarter = 'Q4' THEN amount ELSE 0 END) as Q4
    FROM sales
    GROUP BY product;
    
    -- Method 2: PIVOT (SQL Server)
    SELECT *
    FROM (SELECT product, quarter, amount FROM sales) src
    PIVOT (
        SUM(amount)
        FOR quarter IN ([Q1], [Q2], [Q3], [Q4])
    ) pvt;
    
    -- Method 3: crosstab (PostgreSQL)
    SELECT *
    FROM crosstab(
        'SELECT product, quarter, amount FROM sales ORDER BY 1,2',
        'SELECT DISTINCT quarter FROM sales ORDER BY 1'
    ) AS ct(product text, Q1 numeric, Q2 numeric, Q3 numeric, Q4 numeric);
    ```
    
    **Unpivot: Columns to Rows**
    
    ```sql
    -- Original: product | Q1 | Q2 | Q3 | Q4
    -- Want: product | quarter | amount
    
    -- Method 1: UNION ALL (Works everywhere)
    SELECT product, 'Q1' as quarter, Q1 as amount FROM quarterly_sales
    UNION ALL
    SELECT product, 'Q2' as quarter, Q2 as amount FROM quarterly_sales
    UNION ALL
    SELECT product, 'Q3' as quarter, Q3 as amount FROM quarterly_sales
    UNION ALL
    SELECT product, 'Q4' as quarter, Q4 as amount FROM quarterly_sales;
    
    -- Method 2: UNPIVOT (SQL Server)
    SELECT product, quarter, amount
    FROM quarterly_sales
    UNPIVOT (
        amount FOR quarter IN (Q1, Q2, Q3, Q4)
    ) unpvt;
    
    -- Method 3: LATERAL + VALUES (PostgreSQL)
    SELECT qs.product, t.quarter, t.amount
    FROM quarterly_sales qs
    CROSS JOIN LATERAL (
        VALUES ('Q1', qs.Q1), ('Q2', qs.Q2), ('Q3', qs.Q3), ('Q4', qs.Q4)
    ) AS t(quarter, amount);
    ```
    
    **Dynamic Pivot (Variable columns):**
    
    ```sql
    -- PostgreSQL with dynamic SQL
    DO $$
    DECLARE
        sql_query text;
        quarters text;
    BEGIN
        SELECT string_agg(DISTINCT 
            'SUM(CASE WHEN quarter = ''' || quarter || ''' THEN amount END) as ' || quarter,
            ', ')
        INTO quarters
        FROM sales;
        
        sql_query := 'SELECT product, ' || quarters || ' FROM sales GROUP BY product';
        EXECUTE sql_query;
    END $$;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced data transformation skills.
        
        **Strong answer signals:**
        
        - Uses CASE/GROUP BY as universal solution
        - Knows DB-specific syntax (PIVOT, crosstab)
        - Can handle dynamic column lists
        - Mentions: "UNION ALL for unpivot works everywhere"

---

### Explain Query Optimization and Execution Plans - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Performance`, `Query Optimization`, `EXPLAIN` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **Reading Execution Plans:**
    
    ```sql
    -- PostgreSQL
    EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';
    
    -- MySQL
    EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';
    
    -- SQL Server
    SET SHOWPLAN_ALL ON;
    SELECT * FROM users WHERE email = 'test@example.com';
    ```
    
    **Key Operations (Good vs Bad):**
    
    | Operation | Good/Bad | Meaning |
    |-----------|----------|---------|
    | Index Scan | Good | Using index |
    | Index Only Scan | Best | All data from index |
    | Seq Scan | Bad* | Full table scan |
    | Nested Loop | Depends | OK for small tables |
    | Hash Join | Good | Efficient for larger tables |
    | Sort | Expensive | Consider index |
    
    **Common Optimization Techniques:**
    
    ```sql
    -- 1. Add appropriate indexes
    CREATE INDEX idx_email ON users(email);
    
    -- 2. Avoid SELECT *
    -- ❌ Bad
    SELECT * FROM users WHERE id = 1;
    -- ✅ Good
    SELECT id, name, email FROM users WHERE id = 1;
    
    -- 3. Avoid functions on indexed columns
    -- ❌ Breaks index
    SELECT * FROM users WHERE YEAR(created_at) = 2024;
    -- ✅ Uses index
    SELECT * FROM users 
    WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';
    
    -- 4. Use EXISTS instead of IN for large sets
    -- ❌ Slower
    SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE active = 1);
    -- ✅ Faster
    SELECT * FROM orders o WHERE EXISTS (SELECT 1 FROM customers c WHERE c.id = o.customer_id AND c.active = 1);
    
    -- 5. Limit results early
    -- ❌ Sort everything, then limit
    SELECT * FROM huge_table ORDER BY created_at DESC LIMIT 10;
    -- Better with index on created_at
    
    -- 6. Avoid OR on different columns
    -- ❌ Can't use single index
    SELECT * FROM users WHERE email = 'a@b.com' OR phone = '123';
    -- ✅ Use UNION
    SELECT * FROM users WHERE email = 'a@b.com'
    UNION
    SELECT * FROM users WHERE phone = '123';
    ```
    
    **Execution Plan Metrics:**
    
    ```sql
    EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
    SELECT * FROM orders WHERE customer_id = 123;
    
    -- Key metrics:
    -- - actual time: execution time
    -- - rows: rows processed
    -- - loops: number of iterations
    -- - buffers: memory/disk reads
    -- - cost: estimated query cost
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Performance debugging skills.
        
        **Strong answer signals:**
        
        - Can read EXPLAIN output
        - Knows when Seq Scan is actually OK (small tables)
        - Gives specific optimizations (indexes, avoiding functions)
        - Mentions: "Always test with production-like data volumes"

---


### What Are Subqueries? Correlated vs Non-Correlated - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Subqueries`, `Query Optimization`, `SQL Basics` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **What Are Subqueries?**
    
    A subquery is a query nested inside another query. They can appear in SELECT, FROM, WHERE, or HAVING clauses.
    
    **Non-Correlated Subquery:**
    
    Executed once, independently of the outer query.
    
    ```sql
    -- Find employees with salary above average
    SELECT name, salary
    FROM employees
    WHERE salary > (
        SELECT AVG(salary) FROM employees  -- Runs once
    );
    
    -- Find products in categories with high sales
    SELECT product_name
    FROM products
    WHERE category_id IN (
        SELECT category_id 
        FROM categories 
        WHERE total_sales > 1000000
    );
    ```
    
    **Correlated Subquery:**
    
    References outer query, executed once per outer row (usually slower).
    
    ```sql
    -- Find employees earning above their department average
    SELECT e.name, e.salary, e.department_id
    FROM employees e
    WHERE e.salary > (
        SELECT AVG(salary) 
        FROM employees 
        WHERE department_id = e.department_id  -- References outer row
    );
    
    -- Equivalent using window function (often faster)
    SELECT name, salary, department_id
    FROM (
        SELECT *, AVG(salary) OVER (PARTITION BY department_id) AS dept_avg
        FROM employees
    ) sub
    WHERE salary > dept_avg;
    ```
    
    **Subqueries in Different Clauses:**
    
    | Clause | Example Use |
    |--------|-------------|
    | WHERE | Filter based on subquery result |
    | FROM | Derived table (inline view) |
    | SELECT | Scalar subquery for computed column |
    | HAVING | Filter groups based on subquery |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Query composition skills.
        
        **Strong answer signals:**
        
        - Knows correlated subqueries run per-row (performance impact)
        - Can rewrite correlated subqueries as JOINs or window functions
        - Uses EXISTS for existence checks (not IN for large sets)
        - Mentions: "Non-correlated is like a constant, correlated is like a loop"

---

### Explain CASE Statements in SQL - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Conditional Logic`, `CASE`, `SQL Basics` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **What is CASE?**
    
    CASE provides conditional logic in SQL, similar to if-else in programming.
    
    **Simple CASE:**
    
    ```sql
    -- Compare to specific values
    SELECT 
        order_id,
        status,
        CASE status
            WHEN 'pending' THEN 'Awaiting Processing'
            WHEN 'shipped' THEN 'On the Way'
            WHEN 'delivered' THEN 'Complete'
            ELSE 'Unknown'
        END AS status_description
    FROM orders;
    ```
    
    **Searched CASE (more flexible):**
    
    ```sql
    SELECT 
        name,
        salary,
        CASE 
            WHEN salary >= 150000 THEN 'Executive'
            WHEN salary >= 100000 THEN 'Senior'
            WHEN salary >= 70000 THEN 'Mid-Level'
            ELSE 'Junior'
        END AS level
    FROM employees;
    ```
    
    **CASE with Aggregation:**
    
    ```sql
    -- Pivot-like behavior
    SELECT 
        department,
        SUM(CASE WHEN gender = 'M' THEN 1 ELSE 0 END) AS male_count,
        SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) AS female_count,
        AVG(CASE WHEN status = 'active' THEN salary END) AS avg_active_salary
    FROM employees
    GROUP BY department;
    
    -- Conditional aggregation for metrics
    SELECT 
        DATE_TRUNC('month', order_date) AS month,
        COUNT(*) AS total_orders,
        SUM(CASE WHEN status = 'returned' THEN 1 ELSE 0 END) AS returns,
        ROUND(100.0 * SUM(CASE WHEN status = 'returned' THEN 1 ELSE 0 END) / COUNT(*), 2) AS return_rate
    FROM orders
    GROUP BY 1;
    ```
    
    **CASE in ORDER BY:**
    
    ```sql
    -- Custom sort order
    SELECT * FROM tickets
    ORDER BY 
        CASE priority
            WHEN 'critical' THEN 1
            WHEN 'high' THEN 2
            WHEN 'medium' THEN 3
            WHEN 'low' THEN 4
        END;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Conditional logic in SQL.
        
        **Strong answer signals:**
        
        - Uses CASE for pivot-like aggregations
        - Knows CASE returns NULL if no ELSE and no match
        - Uses in ORDER BY for custom sorting
        - Combines with aggregation functions effectively

---

### How Do You Work with Dates in SQL? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Date Functions`, `Temporal Data`, `Data Manipulation` | **Asked by:** Amazon, Google, Meta, Netflix

??? success "View Answer"

    **Common Date Functions (PostgreSQL/Standard):**
    
    ```sql
    -- Current date/time
    SELECT 
        CURRENT_DATE,                    -- 2024-12-10
        CURRENT_TIMESTAMP,               -- 2024-12-10 14:30:00
        NOW();                           -- Same as CURRENT_TIMESTAMP
    
    -- Extract components
    SELECT 
        EXTRACT(YEAR FROM order_date) AS year,
        EXTRACT(MONTH FROM order_date) AS month,
        EXTRACT(DOW FROM order_date) AS day_of_week,  -- 0=Sunday
        DATE_PART('quarter', order_date) AS quarter
    FROM orders;
    
    -- Truncate to period
    SELECT 
        DATE_TRUNC('month', order_date) AS month_start,
        DATE_TRUNC('week', order_date) AS week_start,
        DATE_TRUNC('hour', created_at) AS hour
    FROM orders;
    
    -- Date arithmetic
    SELECT 
        order_date + INTERVAL '7 days' AS one_week_later,
        order_date - INTERVAL '1 month' AS one_month_ago,
        AGE(NOW(), order_date) AS time_since_order,
        order_date - '2024-01-01'::DATE AS days_into_year
    FROM orders;
    ```
    
    **MySQL Differences:**
    
    ```sql
    -- MySQL specific
    SELECT 
        DATE_FORMAT(order_date, '%Y-%m') AS month,
        DATEDIFF(NOW(), order_date) AS days_ago,
        DATE_ADD(order_date, INTERVAL 7 DAY) AS week_later,
        YEAR(order_date), MONTH(order_date), DAY(order_date);
    ```
    
    **Common Date Queries:**
    
    ```sql
    -- Last 30 days
    SELECT * FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '30 days';
    
    -- Same period last year (Year-over-Year)
    SELECT 
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS revenue
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '1 year'
    GROUP BY 1;
    
    -- First day of month, last day of month
    SELECT 
        DATE_TRUNC('month', CURRENT_DATE) AS first_day,
        (DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day')::DATE AS last_day;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Temporal data handling.
        
        **Strong answer signals:**
        
        - Knows DATE_TRUNC for period grouping
        - Handles timezone awareness
        - Uses INTERVAL for date arithmetic
        - Knows differences between PostgreSQL and MySQL

---

### How Do You Handle NULL Values in SQL? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `NULL Handling`, `COALESCE`, `Data Quality` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Understanding NULL:**
    
    NULL represents missing or unknown data. NULL is not equal to anything, including itself.
    
    ```sql
    -- NULL comparisons
    SELECT 1 = NULL;      -- NULL (not TRUE or FALSE)
    SELECT NULL = NULL;   -- NULL (not TRUE)
    SELECT 1 IS NULL;     -- FALSE
    SELECT NULL IS NULL;  -- TRUE
    
    -- NULL in arithmetic
    SELECT 5 + NULL;      -- NULL (NULL propagates)
    SELECT 5 * NULL;      -- NULL
    ```
    
    **Checking for NULL:**
    
    ```sql
    -- IS NULL / IS NOT NULL
    SELECT * FROM users WHERE email IS NULL;
    SELECT * FROM users WHERE email IS NOT NULL;
    
    -- Don't use = NULL (won't work)
    SELECT * FROM users WHERE email = NULL;  -- Returns nothing!
    ```
    
    **Handling NULL Values:**
    
    ```sql
    -- COALESCE: Return first non-NULL
    SELECT 
        name,
        COALESCE(phone, email, 'No contact') AS contact
    FROM users;
    
    -- NULLIF: Return NULL if values equal
    SELECT 
        revenue / NULLIF(costs, 0) AS profit_margin  -- Avoid division by zero
    FROM financials;
    
    -- CASE for NULL
    SELECT 
        name,
        CASE 
            WHEN bonus IS NULL THEN 0
            ELSE bonus
        END AS bonus_amount
    FROM employees;
    
    -- NVL (Oracle) / IFNULL (MySQL)
    SELECT NVL(commission, 0) FROM sales;  -- Oracle
    SELECT IFNULL(commission, 0) FROM sales;  -- MySQL
    ```
    
    **NULL in Aggregations:**
    
    ```sql
    -- COUNT ignores NULL
    SELECT 
        COUNT(*) AS total_rows,       -- Counts all rows
        COUNT(email) AS with_email,   -- Ignores NULL emails
        COUNT(DISTINCT email) AS unique_emails
    FROM users;
    
    -- AVG, SUM ignore NULL
    SELECT AVG(salary) FROM employees;  -- NULLs excluded from calculation
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data quality awareness.
        
        **Strong answer signals:**
        
        - Knows NULL != NULL (uses IS NULL)
        - Uses COALESCE for defaults
        - Uses NULLIF to prevent division by zero
        - Understands NULL behavior in aggregations

---

### Explain GROUP BY and Aggregation Best Practices - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Aggregation`, `GROUP BY`, `SQL Basics` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **GROUP BY Basics:**
    
    ```sql
    -- Basic grouping
    SELECT 
        department,
        COUNT(*) AS employee_count,
        AVG(salary) AS avg_salary,
        MAX(salary) AS max_salary,
        MIN(salary) AS min_salary,
        SUM(salary) AS total_payroll
    FROM employees
    GROUP BY department;
    
    -- Multiple columns
    SELECT 
        department,
        job_title,
        COUNT(*) AS count
    FROM employees
    GROUP BY department, job_title;
    
    -- With expressions
    SELECT 
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS revenue
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date);
    -- Or use column alias position
    GROUP BY 1;
    ```
    
    **HAVING vs WHERE:**
    
    ```sql
    -- WHERE filters rows BEFORE grouping
    -- HAVING filters groups AFTER aggregation
    
    SELECT 
        department,
        AVG(salary) AS avg_salary
    FROM employees
    WHERE status = 'active'         -- Filter rows first
    GROUP BY department
    HAVING AVG(salary) > 75000;     -- Filter aggregated results
    ```
    
    **Advanced Aggregations:**
    
    ```sql
    -- Multiple aggregations with CASE
    SELECT 
        department,
        COUNT(CASE WHEN status = 'active' THEN 1 END) AS active_count,
        COUNT(CASE WHEN status = 'inactive' THEN 1 END) AS inactive_count,
        ROUND(100.0 * COUNT(CASE WHEN status = 'active' THEN 1 END) / COUNT(*), 2) AS active_pct
    FROM employees
    GROUP BY department;
    
    -- ROLLUP for subtotals
    SELECT 
        COALESCE(region, 'Total') AS region,
        COALESCE(country, 'Subtotal') AS country,
        SUM(sales) AS total_sales
    FROM orders
    GROUP BY ROLLUP(region, country);
    
    -- GROUPING SETS for multiple groupings
    SELECT region, product, SUM(sales)
    FROM orders
    GROUP BY GROUPING SETS (
        (region, product),
        (region),
        ()  -- Grand total
    );
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data aggregation skills.
        
        **Strong answer signals:**
        
        - Knows HAVING filters after GROUP BY
        - Uses positional GROUP BY (GROUP BY 1) appropriately
        - Knows ROLLUP/CUBE for subtotals
        - Combines CASE with aggregation

---

### What Are Stored Procedures and Functions? - Oracle, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Stored Procedures`, `Functions`, `Database Programming` | **Asked by:** Oracle, Microsoft, IBM, Amazon

??? success "View Answer"

    **Stored Procedure vs Function:**
    
    | Aspect | Stored Procedure | Function |
    |--------|------------------|----------|
    | Returns | Can return 0, 1, or multiple values | Must return exactly 1 value |
    | Use in SQL | Cannot use in SELECT | Can use in SELECT |
    | Side effects | Can modify data | Usually read-only |
    | Call syntax | CALL/EXECUTE | Part of expression |
    
    **Creating a Stored Procedure (PostgreSQL):**
    
    ```sql
    CREATE OR REPLACE PROCEDURE update_employee_salary(
        emp_id INTEGER,
        new_salary NUMERIC
    )
    LANGUAGE plpgsql
    AS $$
    BEGIN
        UPDATE employees
        SET salary = new_salary,
            updated_at = NOW()
        WHERE id = emp_id;
        
        IF NOT FOUND THEN
            RAISE EXCEPTION 'Employee % not found', emp_id;
        END IF;
    END;
    $$;
    
    -- Call the procedure
    CALL update_employee_salary(123, 75000);
    ```
    
    **Creating a Function:**
    
    ```sql
    CREATE OR REPLACE FUNCTION calculate_bonus(
        salary NUMERIC,
        performance_rating INTEGER
    )
    RETURNS NUMERIC
    LANGUAGE plpgsql
    AS $$
    BEGIN
        RETURN CASE 
            WHEN performance_rating >= 5 THEN salary * 0.20
            WHEN performance_rating >= 4 THEN salary * 0.15
            WHEN performance_rating >= 3 THEN salary * 0.10
            ELSE 0
        END;
    END;
    $$;
    
    -- Use in query
    SELECT 
        name,
        salary,
        calculate_bonus(salary, performance_rating) AS bonus
    FROM employees;
    ```
    
    **When to Use:**
    
    | Use Procedure | Use Function |
    |---------------|--------------|
    | Data modification | Calculations |
    | Complex transactions | Value lookup |
    | Administrative tasks | Use in SELECT |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Database programming knowledge.
        
        **Strong answer signals:**
        
        - Knows procedures can have side effects, functions shouldn't
        - Mentions security: SECURITY DEFINER vs INVOKER
        - Knows performance implications (network round trips)
        - Mentions: "Functions in WHERE can prevent index usage"

---

### What Are Views? When Should You Use Them? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Views`, `Database Design`, `Abstraction` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **What is a View?**
    
    A view is a virtual table defined by a SQL query. It doesn't store data, just the query definition.
    
    **Creating Views:**
    
    ```sql
    -- Simple view
    CREATE VIEW active_employees AS
    SELECT id, name, email, department
    FROM employees
    WHERE status = 'active';
    
    -- Use like a table
    SELECT * FROM active_employees WHERE department = 'Engineering';
    
    -- Complex view with joins
    CREATE VIEW order_summary AS
    SELECT 
        o.id,
        o.order_date,
        c.name AS customer_name,
        SUM(oi.quantity * oi.unit_price) AS total_amount
    FROM orders o
    JOIN customers c ON o.customer_id = c.id
    JOIN order_items oi ON o.id = oi.order_id
    GROUP BY o.id, o.order_date, c.name;
    ```
    
    **Materialized View (PostgreSQL):**
    
    ```sql
    -- Stores actual data, needs refresh
    CREATE MATERIALIZED VIEW monthly_sales AS
    SELECT 
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS total_sales
    FROM orders
    GROUP BY 1;
    
    -- Refresh data
    REFRESH MATERIALIZED VIEW monthly_sales;
    REFRESH MATERIALIZED VIEW CONCURRENTLY monthly_sales;  -- No blocking
    ```
    
    **Benefits:**
    
    | Benefit | Explanation |
    |---------|-------------|
    | Abstraction | Hide complex joins |
    | Security | Expose subset of columns |
    | Simplicity | Reusable query logic |
    | Backward compatibility | Change underlying structure |
    
    **Drawbacks:**
    
    | Drawback | Explanation |
    |----------|-------------|
    | Performance | Views aren't always optimized |
    | Updates | Complex views may not be updatable |
    | Maintenance | Cascading dependencies |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Database design understanding.
        
        **Strong answer signals:**
        
        - Knows views are virtual (re-executed each time)
        - Mentions materialized views for performance
        - Uses views for security (column-level access)
        - Knows updatable view requirements

---

### What Are Triggers in SQL? - Oracle, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Triggers`, `Automation`, `Database Programming` | **Asked by:** Oracle, Microsoft, IBM

??? success "View Answer"

    **What Are Triggers?**
    
    Triggers are special procedures that automatically execute in response to specific events on a table (INSERT, UPDATE, DELETE).
    
    **Creating Triggers (PostgreSQL):**
    
    ```sql
    -- Audit trigger
    CREATE OR REPLACE FUNCTION audit_employee_changes()
    RETURNS TRIGGER AS $$
    BEGIN
        IF TG_OP = 'INSERT' THEN
            INSERT INTO employee_audit (action, employee_id, new_data, changed_at)
            VALUES ('INSERT', NEW.id, row_to_json(NEW), NOW());
        ELSIF TG_OP = 'UPDATE' THEN
            INSERT INTO employee_audit (action, employee_id, old_data, new_data, changed_at)
            VALUES ('UPDATE', NEW.id, row_to_json(OLD), row_to_json(NEW), NOW());
        ELSIF TG_OP = 'DELETE' THEN
            INSERT INTO employee_audit (action, employee_id, old_data, changed_at)
            VALUES ('DELETE', OLD.id, row_to_json(OLD), NOW());
        END IF;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    
    CREATE TRIGGER employee_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON employees
    FOR EACH ROW
    EXECUTE FUNCTION audit_employee_changes();
    ```
    
    **Trigger Types:**
    
    | Type | When | Use Case |
    |------|------|----------|
    | BEFORE | Before row change | Validation, modification |
    | AFTER | After row change | Audit, notifications |
    | INSTEAD OF | Replace operation | Updatable views |
    | FOR EACH ROW | Per row | Most common |
    | FOR EACH STATEMENT | Per statement | Batch operations |
    
    **Common Use Cases:**
    
    ```sql
    -- Auto-update timestamp
    CREATE TRIGGER update_timestamp
    BEFORE UPDATE ON orders
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();
    
    CREATE FUNCTION update_modified_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    ```
    
    **Cautions:**
    
    - Triggers add hidden complexity
    - Can cause performance issues
    - Debugging is harder
    - May conflict with ORMs

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced database knowledge.
        
        **Strong answer signals:**
        
        - Knows BEFORE vs AFTER timing
        - Mentions audit logging use case
        - Warns about hidden complexity
        - Knows NEW/OLD row references

---

### Explain Transactions and Isolation Levels - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Transactions`, `Isolation Levels`, `Concurrency` | **Asked by:** Google, Amazon, Meta, Oracle

??? success "View Answer"

    **What Are Transactions?**
    
    A transaction is a sequence of operations treated as a single unit. All succeed together or all fail together.
    
    ```sql
    BEGIN;  -- Start transaction
    
    UPDATE accounts SET balance = balance - 100 WHERE id = 1;
    UPDATE accounts SET balance = balance + 100 WHERE id = 2;
    
    COMMIT;  -- Make permanent
    -- or ROLLBACK; to undo
    ```
    
    **Isolation Levels:**
    
    | Level | Dirty Read | Non-Repeatable Read | Phantom Read |
    |-------|------------|---------------------|--------------|
    | READ UNCOMMITTED | ✅ Possible | ✅ Possible | ✅ Possible |
    | READ COMMITTED | ❌ Prevented | ✅ Possible | ✅ Possible |
    | REPEATABLE READ | ❌ Prevented | ❌ Prevented | ✅ Possible |
    | SERIALIZABLE | ❌ Prevented | ❌ Prevented | ❌ Prevented |
    
    **Setting Isolation Level:**
    
    ```sql
    -- Per session
    SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
    
    BEGIN;
    -- Your queries here
    COMMIT;
    ```
    
    **Concurrency Problems:**
    
    ```sql
    -- Dirty Read: Reading uncommitted changes
    -- Session 1: UPDATE accounts SET balance = 0 WHERE id = 1;
    -- Session 2: SELECT balance FROM accounts WHERE id = 1; -- Sees 0
    -- Session 1: ROLLBACK;
    -- Problem: Session 2 read invalid data
    
    -- Non-Repeatable Read: Same query, different results
    -- Session 1: SELECT balance FROM accounts WHERE id = 1; -- Returns 100
    -- Session 2: UPDATE accounts SET balance = 50 WHERE id = 1; COMMIT;
    -- Session 1: SELECT balance FROM accounts WHERE id = 1; -- Returns 50
    
    -- Phantom Read: New rows appear
    -- Session 1: SELECT COUNT(*) FROM orders WHERE status = 'pending'; -- 10
    -- Session 2: INSERT INTO orders (status) VALUES ('pending'); COMMIT;
    -- Session 1: SELECT COUNT(*) FROM orders WHERE status = 'pending'; -- 11
    ```
    
    **Deadlock Prevention:**
    
    ```sql
    -- Lock ordering: Always acquire locks in same order
    -- Timeout: SET lock_timeout = '5s';
    -- FOR UPDATE SKIP LOCKED: Skip locked rows
    SELECT * FROM tasks WHERE status = 'pending'
    FOR UPDATE SKIP LOCKED
    LIMIT 1;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Concurrency understanding.
        
        **Strong answer signals:**
        
        - Explains all three anomalies with examples
        - Knows default levels (PostgreSQL: READ COMMITTED)
        - Mentions performance vs consistency trade-off
        - Knows FOR UPDATE, SKIP LOCKED for concurrent processing

---

### How Do You Handle Locking in SQL? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Locking`, `Concurrency`, `Performance` | **Asked by:** Google, Amazon, Meta, Oracle

??? success "View Answer"

    **Types of Locks:**
    
    | Lock Type | Purpose |
    |-----------|---------|
    | Shared (S) | Read, allows other reads |
    | Exclusive (X) | Write, blocks all access |
    | Row-level | Lock specific rows |
    | Table-level | Lock entire table |
    
    **Explicit Locking:**
    
    ```sql
    -- Lock rows for update
    BEGIN;
    SELECT * FROM inventory WHERE product_id = 123
    FOR UPDATE;  -- Exclusive lock on this row
    
    UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 123;
    COMMIT;
    
    -- Lock variants
    FOR UPDATE;              -- Wait for lock
    FOR UPDATE NOWAIT;       -- Error if locked
    FOR UPDATE SKIP LOCKED;  -- Skip locked rows
    FOR SHARE;               -- Shared lock (allow reads)
    ```
    
    **Optimistic vs Pessimistic Locking:**
    
    ```sql
    -- Pessimistic: Lock upfront
    SELECT * FROM products WHERE id = 1 FOR UPDATE;
    -- Make changes
    UPDATE products SET quantity = ? WHERE id = 1;
    
    -- Optimistic: Check version on update
    SELECT id, name, quantity, version FROM products WHERE id = 1;
    -- Application makes changes
    UPDATE products 
    SET quantity = new_quantity, version = version + 1
    WHERE id = 1 AND version = old_version;
    -- If 0 rows affected, someone else modified it
    ```
    
    **Deadlock:**
    
    ```sql
    -- Session 1:                    -- Session 2:
    BEGIN;                           BEGIN;
    UPDATE a SET x=1 WHERE id=1;     UPDATE b SET y=1 WHERE id=1;
    UPDATE b SET y=1 WHERE id=1;     UPDATE a SET x=1 WHERE id=1;
    -- Waits for Session 2           -- Waits for Session 1
    -- DEADLOCK!
    
    -- Prevention: Lock in consistent order
    BEGIN;
    SELECT * FROM a WHERE id = 1 FOR UPDATE;
    SELECT * FROM b WHERE id = 1 FOR UPDATE;
    -- Now safe to update both
    ```
    
    **Advisory Locks (PostgreSQL):**
    
    ```sql
    -- Application-level locks
    SELECT pg_advisory_lock(12345);  -- Acquire lock
    -- Do work
    SELECT pg_advisory_unlock(12345);  -- Release
    
    -- Try without waiting
    SELECT pg_try_advisory_lock(12345);  -- Returns t/f
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Concurrency control design.
        
        **Strong answer signals:**
        
        - Knows optimistic vs pessimistic locking trade-offs
        - Uses SKIP LOCKED for worker queues
        - Prevents deadlocks with ordering
        - Chooses row-level locks when possible

---

### What is Database Partitioning? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Partitioning`, `Scalability`, `Performance` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **What is Partitioning?**
    
    Partitioning divides large tables into smaller, more manageable pieces while appearing as a single table.
    
    **Types of Partitioning:**
    
    | Type | How It Works | Use Case |
    |------|--------------|----------|
    | Range | By range of values | Time-series data |
    | List | By explicit values | Categories, regions |
    | Hash | By hash of column | Even distribution |
    | Composite | Combination | Complex requirements |
    
    **Range Partitioning (PostgreSQL):**
    
    ```sql
    -- Create partitioned table
    CREATE TABLE orders (
        id SERIAL,
        order_date DATE NOT NULL,
        customer_id INTEGER,
        amount NUMERIC
    ) PARTITION BY RANGE (order_date);
    
    -- Create partitions
    CREATE TABLE orders_2024_q1 PARTITION OF orders
        FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
    
    CREATE TABLE orders_2024_q2 PARTITION OF orders
        FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
    
    -- Query scans only relevant partitions
    SELECT * FROM orders 
    WHERE order_date BETWEEN '2024-02-01' AND '2024-02-28';
    -- Only scans orders_2024_q1
    ```
    
    **List Partitioning:**
    
    ```sql
    CREATE TABLE customers (
        id SERIAL,
        name TEXT,
        region TEXT NOT NULL
    ) PARTITION BY LIST (region);
    
    CREATE TABLE customers_us PARTITION OF customers
        FOR VALUES IN ('us-east', 'us-west', 'us-central');
    
    CREATE TABLE customers_eu PARTITION OF customers
        FOR VALUES IN ('eu-west', 'eu-north', 'eu-central');
    ```
    
    **Benefits:**
    
    | Benefit | Explanation |
    |---------|-------------|
    | Query Performance | Partition pruning |
    | Maintenance | Archives, bulk deletes |
    | Parallel Processing | Partition-wise joins |
    | Storage | Different tablespaces |
    
    **Partition Maintenance:**
    
    ```sql
    -- Drop old partition (instant delete)
    DROP TABLE orders_2020_q1;
    
    -- Add new partition
    CREATE TABLE orders_2025_q1 PARTITION OF orders
        FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');
    
    -- Detach for archive
    ALTER TABLE orders DETACH PARTITION orders_2023_q1;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Scalability knowledge.
        
        **Strong answer signals:**
        
        - Knows partition pruning for query performance
        - Uses range for time-series, hash for even distribution
        - Mentions partition key must be in all queries
        - Knows: "Dropping partition is instant vs DELETE"

---

### Explain Database Constraints and Their Uses - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Constraints`, `Data Integrity`, `Database Design` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Types of Constraints:**
    
    | Constraint | Purpose |
    |------------|---------|
    | PRIMARY KEY | Unique identifier, not null |
    | FOREIGN KEY | Referential integrity |
    | UNIQUE | No duplicate values |
    | NOT NULL | Value required |
    | CHECK | Custom validation |
    | DEFAULT | Default value |
    
    **Creating Constraints:**
    
    ```sql
    CREATE TABLE orders (
        id SERIAL PRIMARY KEY,
        customer_id INTEGER NOT NULL REFERENCES customers(id),
        order_date DATE DEFAULT CURRENT_DATE,
        status VARCHAR(20) CHECK (status IN ('pending', 'shipped', 'delivered')),
        total_amount NUMERIC CHECK (total_amount >= 0),
        email VARCHAR(255) UNIQUE
    );
    
    -- Add constraint to existing table
    ALTER TABLE orders ADD CONSTRAINT positive_amount CHECK (total_amount >= 0);
    
    -- Named constraints for better error messages
    ALTER TABLE orders 
    ADD CONSTRAINT fk_orders_customer 
    FOREIGN KEY (customer_id) REFERENCES customers(id)
    ON DELETE RESTRICT
    ON UPDATE CASCADE;
    ```
    
    **Foreign Key Actions:**
    
    | Action | On Delete | On Update |
    |--------|-----------|-----------|
    | CASCADE | Delete child rows | Update child rows |
    | RESTRICT | Prevent if children exist | Prevent if children exist |
    | SET NULL | Set FK to NULL | Set FK to NULL |
    | SET DEFAULT | Set FK to default | Set FK to default |
    | NO ACTION | Same as RESTRICT | Same as RESTRICT |
    
    **Deferred Constraints:**
    
    ```sql
    -- Check at transaction end, not per statement
    ALTER TABLE orders_items
    ADD CONSTRAINT fk_order
    FOREIGN KEY (order_id) REFERENCES orders(id)
    DEFERRABLE INITIALLY DEFERRED;
    
    BEGIN;
    INSERT INTO order_items (order_id, product_id) VALUES (999, 1);  -- Order 999 doesn't exist yet
    INSERT INTO orders (id, customer_id) VALUES (999, 123);  -- Now it does
    COMMIT;  -- FK check happens here
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data integrity knowledge.
        
        **Strong answer signals:**
        
        - Uses CHECK constraints for business rules
        - Knows FK actions (CASCADE, RESTRICT)
        - Mentions performance impact of FKs
        - Uses DEFERRABLE for complex inserts

---

### How Do You Design a Database Schema? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Database Design`, `Schema`, `Normalization` | **Asked by:** Amazon, Google, Meta, Microsoft

??? success "View Answer"

    **Design Process:**
    
    1. **Identify Entities**: Users, Orders, Products
    2. **Define Relationships**: One-to-many, many-to-many
    3. **Choose Keys**: Natural vs surrogate
    4. **Apply Normalization**: Eliminate redundancy
    5. **Consider Denormalization**: For read performance
    
    **Example: E-Commerce Schema:**
    
    ```sql
    -- Core entities
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        name VARCHAR(100) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    CREATE TABLE products (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        price NUMERIC(10, 2) NOT NULL CHECK (price >= 0),
        stock_quantity INTEGER DEFAULT 0
    );
    
    -- One-to-many: User has many orders
    CREATE TABLE orders (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        status VARCHAR(20) DEFAULT 'pending',
        total_amount NUMERIC(10, 2),
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Many-to-many: Order contains many products
    CREATE TABLE order_items (
        order_id INTEGER REFERENCES orders(id),
        product_id INTEGER REFERENCES products(id),
        quantity INTEGER NOT NULL CHECK (quantity > 0),
        unit_price NUMERIC(10, 2) NOT NULL,
        PRIMARY KEY (order_id, product_id)
    );
    
    -- Self-referencing: Categories hierarchy
    CREATE TABLE categories (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        parent_id INTEGER REFERENCES categories(id)
    );
    ```
    
    **Normalization Forms:**
    
    | Form | Rule |
    |------|------|
    | 1NF | Atomic values, no repeating groups |
    | 2NF | 1NF + No partial dependencies |
    | 3NF | 2NF + No transitive dependencies |
    | BCNF | Every determinant is a candidate key |
    
    **Denormalization Examples:**
    
    ```sql
    -- Store calculated total (update via trigger)
    ALTER TABLE orders ADD COLUMN item_count INTEGER;
    
    -- Store user name in orders for faster reads
    ALTER TABLE orders ADD COLUMN user_name VARCHAR(100);
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Design thinking.
        
        **Strong answer signals:**
        
        - Starts with entities and relationships
        - Knows when to denormalize
        - Uses appropriate data types
        - Considers indexing strategy upfront

---

### How Do You Optimize Slow Queries? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Performance`, `Query Optimization`, `Indexing` | **Asked by:** Google, Amazon, Netflix, Meta

??? success "View Answer"

    **Optimization Workflow:**
    
    1. **Identify slow queries** (logs, monitoring)
    2. **Analyze with EXPLAIN**
    3. **Check indexes**
    4. **Optimize query structure**
    5. **Consider schema changes**
    
    **EXPLAIN Analysis:**
    
    ```sql
    EXPLAIN ANALYZE
    SELECT o.*, c.name
    FROM orders o
    JOIN customers c ON o.customer_id = c.id
    WHERE o.status = 'pending' AND o.created_at > '2024-01-01';
    
    -- Look for:
    -- Seq Scan (table scan) → Add index
    -- High rows vs actual rows → Stale statistics
    -- Nested Loop with many rows → Consider different join
    ```
    
    **Index Optimization:**
    
    ```sql
    -- Single column for equality
    CREATE INDEX idx_orders_status ON orders(status);
    
    -- Composite for multiple conditions
    CREATE INDEX idx_orders_status_date ON orders(status, created_at);
    
    -- Covering index (avoids table lookup)
    CREATE INDEX idx_orders_covering ON orders(status, created_at) 
    INCLUDE (total_amount);
    
    -- Partial index (smaller, faster)
    CREATE INDEX idx_orders_pending ON orders(created_at) 
    WHERE status = 'pending';
    ```
    
    **Query Rewrites:**
    
    ```sql
    -- ❌ Function on indexed column (can't use index)
    SELECT * FROM orders WHERE YEAR(created_at) = 2024;
    
    -- ✅ Range condition (uses index)
    SELECT * FROM orders 
    WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';
    
    -- ❌ OR can prevent index use
    SELECT * FROM orders WHERE status = 'pending' OR customer_id = 123;
    
    -- ✅ UNION as alternative
    SELECT * FROM orders WHERE status = 'pending'
    UNION ALL
    SELECT * FROM orders WHERE customer_id = 123 AND status != 'pending';
    
    -- ❌ SELECT * (fetches unnecessary data)
    SELECT * FROM large_table WHERE id = 1;
    
    -- ✅ Select only needed columns
    SELECT id, name FROM large_table WHERE id = 1;
    ```
    
    **Statistics Maintenance:**
    
    ```sql
    -- Update statistics (PostgreSQL)
    ANALYZE orders;
    
    -- Reindex for bloated indexes
    REINDEX INDEX idx_orders_status;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Performance debugging ability.
        
        **Strong answer signals:**
        
        - Uses EXPLAIN ANALYZE, not just EXPLAIN
        - Knows composite index column order matters
        - Avoids functions on indexed columns
        - Mentions covering indexes for read-heavy queries

---


### What is the Difference Between UNION, INTERSECT, and EXCEPT? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Set Operations`, `Query Combination`, `SQL Basics` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Set Operations in SQL:**
    
    | Operation | Result |
    |-----------|--------|
    | UNION | All rows from both queries (removes duplicates) |
    | UNION ALL | All rows from both queries (keeps duplicates) |
    | INTERSECT | Only rows in both queries |
    | EXCEPT / MINUS | Rows in first but not in second |
    
    ```sql
    -- Sample tables
    -- table1: (1, 2, 3, 4)
    -- table2: (3, 4, 5, 6)
    
    -- UNION - combines, removes duplicates
    SELECT id FROM table1
    UNION
    SELECT id FROM table2;
    -- Result: 1, 2, 3, 4, 5, 6
    
    -- UNION ALL - combines, keeps duplicates
    SELECT id FROM table1
    UNION ALL
    SELECT id FROM table2;
    -- Result: 1, 2, 3, 4, 3, 4, 5, 6
    
    -- INTERSECT - common elements
    SELECT id FROM table1
    INTERSECT
    SELECT id FROM table2;
    -- Result: 3, 4
    
    -- EXCEPT (MINUS in Oracle) - difference
    SELECT id FROM table1
    EXCEPT
    SELECT id FROM table2;
    -- Result: 1, 2
    ```
    
    **Requirements:**
    
    - Same number of columns
    - Compatible data types
    - Column names from first query
    
    **Performance Tip:**
    
    ```sql
    -- UNION ALL is faster (no duplicate removal)
    -- Use UNION ALL when duplicates are impossible or acceptable
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Set theory understanding.
        
        **Strong answer signals:**
        
        - Knows UNION removes duplicates, UNION ALL doesn't
        - Uses UNION ALL for performance when appropriate
        - Knows EXCEPT is MINUS in Oracle
        - Can explain when each operation is useful

---

### When to Use IN vs EXISTS? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Subqueries`, `Performance`, `Optimization` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **IN vs EXISTS:**
    
    | Aspect | IN | EXISTS |
    |--------|---------|--------------|
    | How it works | Checks value against list | Checks if subquery returns rows |
    | NULL handling | Returns NULL if list contains NULL | Handles NULLs correctly |
    | Performance | Better for small lists | Better for large outer, small inner |
    
    ```sql
    -- IN - checks if value is in subquery result
    SELECT * FROM orders
    WHERE customer_id IN (
        SELECT id FROM customers WHERE country = 'USA'
    );
    
    -- EXISTS - checks if subquery returns any rows
    SELECT * FROM orders o
    WHERE EXISTS (
        SELECT 1 FROM customers c
        WHERE c.id = o.customer_id AND c.country = 'USA'
    );
    ```
    
    **When to Use IN:**
    
    ```sql
    -- Small, static list
    WHERE status IN ('pending', 'processing', 'shipped')
    
    -- Subquery returns few rows
    WHERE id IN (SELECT id FROM small_table)
    ```
    
    **When to Use EXISTS:**
    
    ```sql
    -- Correlated subquery with large outer table
    SELECT * FROM large_orders o
    WHERE EXISTS (
        SELECT 1 FROM small_customers c
        WHERE c.id = o.customer_id
    );
    
    -- Checking existence only (no values needed)
    IF EXISTS (SELECT 1 FROM users WHERE email = 'test@example.com')
    ```
    
    **NOT IN vs NOT EXISTS (Critical Difference):**
    
    ```sql
    -- ⚠️ NOT IN fails with NULLs
    SELECT * FROM orders
    WHERE customer_id NOT IN (SELECT id FROM customers);
    -- If customers.id has NULL, returns empty!
    
    -- ✅ NOT EXISTS handles NULLs correctly
    SELECT * FROM orders o
    WHERE NOT EXISTS (
        SELECT 1 FROM customers c WHERE c.id = o.customer_id
    );
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Query optimization understanding.
        
        **Strong answer signals:**
        
        - Knows NOT IN danger with NULLs
        - Uses EXISTS for existence checks
        - Explains performance differences
        - Mentions query optimizer often makes them equivalent

---

### How Do You Write a Pivot Query? - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Pivot`, `Data Transformation`, `Reporting` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Pivot in SQL:**
    
    Transform rows to columns - useful for reporting.
    
    **Using CASE Statements (Universal):**
    
    ```sql
    -- Sales data: product, month, amount
    SELECT 
        product,
        SUM(CASE WHEN month = 'Jan' THEN amount ELSE 0 END) AS jan,
        SUM(CASE WHEN month = 'Feb' THEN amount ELSE 0 END) AS feb,
        SUM(CASE WHEN month = 'Mar' THEN amount ELSE 0 END) AS mar
    FROM sales
    GROUP BY product;
    
    -- Result:
    -- product | jan  | feb  | mar
    -- A       | 100  | 150  | 200
    -- B       | 80   | 90   | 110
    ```
    
    **SQL Server PIVOT:**
    
    ```sql
    SELECT product, [Jan], [Feb], [Mar]
    FROM (
        SELECT product, month, amount
        FROM sales
    ) AS source
    PIVOT (
        SUM(amount)
        FOR month IN ([Jan], [Feb], [Mar])
    ) AS pivoted;
    ```
    
    **PostgreSQL crosstab:**
    
    ```sql
    -- Requires tablefunc extension
    CREATE EXTENSION IF NOT EXISTS tablefunc;
    
    SELECT * FROM crosstab(
        'SELECT product, month, amount FROM sales ORDER BY 1, 2',
        'SELECT DISTINCT month FROM sales ORDER BY 1'
    ) AS ct(product TEXT, jan INT, feb INT, mar INT);
    ```
    
    **Unpivot (Columns to Rows):**
    
    ```sql
    -- CASE-based unpivot
    SELECT product, 'Jan' AS month, jan AS amount FROM pivoted_table
    UNION ALL
    SELECT product, 'Feb', feb FROM pivoted_table
    UNION ALL
    SELECT product, 'Mar', mar FROM pivoted_table;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced data transformation.
        
        **Strong answer signals:**
        
        - Uses CASE for cross-database compatibility
        - Knows PIVOT syntax varies by database
        - Can handle dynamic column lists
        - Understands when pivoting is appropriate

---

### How Does the MERGE Statement Work? - Amazon, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `MERGE`, `UPSERT`, `DML` | **Asked by:** Amazon, Microsoft, Oracle

??? success "View Answer"

    **MERGE Statement:**
    
    Performs INSERT, UPDATE, or DELETE in a single statement based on source data.
    
    ```sql
    -- SQL Server / Oracle / PostgreSQL 15+
    MERGE INTO target_table AS target
    USING source_table AS source
    ON target.id = source.id
    
    -- When matched (update)
    WHEN MATCHED THEN
        UPDATE SET
            target.name = source.name,
            target.updated_at = CURRENT_TIMESTAMP
    
    -- When not matched (insert)
    WHEN NOT MATCHED THEN
        INSERT (id, name, created_at)
        VALUES (source.id, source.name, CURRENT_TIMESTAMP)
    
    -- When matched but should delete
    WHEN MATCHED AND source.deleted = TRUE THEN
        DELETE;
    ```
    
    **PostgreSQL INSERT ON CONFLICT (UPSERT):**
    
    ```sql
    -- Simpler upsert syntax
    INSERT INTO users (id, name, email)
    VALUES (1, 'John', 'john@example.com')
    ON CONFLICT (id) 
    DO UPDATE SET
        name = EXCLUDED.name,
        email = EXCLUDED.email,
        updated_at = CURRENT_TIMESTAMP;
    
    -- Do nothing on conflict
    INSERT INTO users (id, name)
    VALUES (1, 'John')
    ON CONFLICT (id) DO NOTHING;
    ```
    
    **MySQL UPSERT:**
    
    ```sql
    -- INSERT ... ON DUPLICATE KEY UPDATE
    INSERT INTO users (id, name, email)
    VALUES (1, 'John', 'john@example.com')
    ON DUPLICATE KEY UPDATE
        name = VALUES(name),
        email = VALUES(email);
    
    -- REPLACE (delete then insert)
    REPLACE INTO users (id, name, email)
    VALUES (1, 'John', 'john@example.com');
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data synchronization knowledge.
        
        **Strong answer signals:**
        
        - Knows MERGE syntax and alternatives
        - Uses ON CONFLICT for PostgreSQL
        - Understands atomicity benefits
        - Mentions performance for bulk operations

---

### How Do You Write Recursive CTEs? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Recursive CTE`, `Hierarchical Data`, `Trees` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Recursive CTE Structure:**
    
    ```sql
    WITH RECURSIVE cte_name AS (
        -- Base case (anchor)
        SELECT ...
        
        UNION ALL
        
        -- Recursive case
        SELECT ...
        FROM cte_name  -- Self-reference
        WHERE ...      -- Termination condition
    )
    SELECT * FROM cte_name;
    ```
    
    **Employee Hierarchy:**
    
    ```sql
    -- employees: id, name, manager_id
    WITH RECURSIVE org_chart AS (
        -- Base: CEO (no manager)
        SELECT id, name, manager_id, 1 AS level, name AS path
        FROM employees
        WHERE manager_id IS NULL
        
        UNION ALL
        
        -- Recursive: employees with managers
        SELECT e.id, e.name, e.manager_id, 
               o.level + 1,
               o.path || ' > ' || e.name
        FROM employees e
        JOIN org_chart o ON e.manager_id = o.id
    )
    SELECT * FROM org_chart ORDER BY level, name;
    ```
    
    **Generate Series (Numbers/Dates):**
    
    ```sql
    -- Generate 1 to 10
    WITH RECURSIVE numbers AS (
        SELECT 1 AS n
        UNION ALL
        SELECT n + 1 FROM numbers WHERE n < 10
    )
    SELECT * FROM numbers;
    
    -- Generate date range
    WITH RECURSIVE dates AS (
        SELECT '2024-01-01'::date AS d
        UNION ALL
        SELECT d + 1 FROM dates WHERE d < '2024-01-31'
    )
    SELECT * FROM dates;
    ```
    
    **Bill of Materials (BOM):**
    
    ```sql
    -- Find all components of a product
    WITH RECURSIVE bom AS (
        SELECT component_id, parent_id, quantity, 1 AS level
        FROM components
        WHERE parent_id = 'PRODUCT_001'
        
        UNION ALL
        
        SELECT c.component_id, c.parent_id, 
               c.quantity * b.quantity,
               b.level + 1
        FROM components c
        JOIN bom b ON c.parent_id = b.component_id
    )
    SELECT * FROM bom;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced SQL and tree processing.
        
        **Strong answer signals:**
        
        - Understands anchor + recursive structure
        - Knows to include termination condition
        - Can track depth/level
        - Mentions max recursion limits

---

### What is a Lateral Join? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Lateral Join`, `Correlated Subquery`, `Advanced` | **Asked by:** Google, Amazon, Snowflake

??? success "View Answer"

    **Lateral Join (PostgreSQL/MySQL 8+):**
    
    Allows subquery to reference columns from preceding tables (like a correlated subquery but as a join).
    
    ```sql
    -- Get top 3 orders per customer
    SELECT c.name, o.*
    FROM customers c
    CROSS JOIN LATERAL (
        SELECT *
        FROM orders
        WHERE orders.customer_id = c.id  -- References c!
        ORDER BY created_at DESC
        LIMIT 3
    ) o;
    ```
    
    **Use Cases:**
    
    **Top-N per Group:**
    
    ```sql
    -- Top 5 products per category
    SELECT cat.name, prod.*
    FROM categories cat
    CROSS JOIN LATERAL (
        SELECT *
        FROM products
        WHERE category_id = cat.id
        ORDER BY sales DESC
        LIMIT 5
    ) prod;
    ```
    
    **Running Calculations:**
    
    ```sql
    -- Calculate moving average
    SELECT 
        t.id,
        t.value,
        stats.moving_avg
    FROM time_series t
    CROSS JOIN LATERAL (
        SELECT AVG(value) AS moving_avg
        FROM time_series
        WHERE id BETWEEN t.id - 4 AND t.id
    ) stats;
    ```
    
    **Expanding JSON Arrays:**
    
    ```sql
    -- PostgreSQL: Expand JSON array
    SELECT 
        users.id,
        tags.tag
    FROM users
    CROSS JOIN LATERAL jsonb_array_elements_text(users.tags) AS tags(tag);
    ```
    
    **SQL Server Equivalent (APPLY):**
    
    ```sql
    SELECT c.name, o.*
    FROM customers c
    CROSS APPLY (
        SELECT TOP 3 *
        FROM orders
        WHERE orders.customer_id = c.id
        ORDER BY created_at DESC
    ) o;
    -- OUTER APPLY for left join behavior
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced join understanding.
        
        **Strong answer signals:**
        
        - Knows LATERAL allows correlated subqueries in FROM
        - Uses for top-N per group problems
        - Knows CROSS APPLY in SQL Server
        - Compares to window functions for alternatives

---

### How Do You Query JSON Data in SQL? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `JSON`, `Semi-structured`, `Modern SQL` | **Asked by:** Google, Amazon, Snowflake, MongoDB

??? success "View Answer"

    **PostgreSQL JSON/JSONB:**
    
    ```sql
    -- Create table with JSON
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        data JSONB
    );
    
    -- Insert JSON
    INSERT INTO users (data) VALUES 
    ('{"name": "John", "age": 30, "tags": ["admin", "user"]}');
    
    -- Extract values
    SELECT 
        data->>'name' AS name,           -- Text extraction
        (data->>'age')::int AS age,      -- With casting
        data->'tags'->0 AS first_tag     -- Array access
    FROM users;
    
    -- Query JSON fields
    SELECT * FROM users
    WHERE data->>'name' = 'John';
    
    SELECT * FROM users
    WHERE data @> '{"tags": ["admin"]}';  -- Contains
    
    -- Index for performance
    CREATE INDEX idx_users_name ON users ((data->>'name'));
    CREATE INDEX idx_users_data ON users USING GIN (data);
    ```
    
    **MySQL JSON:**
    
    ```sql
    -- Extract with JSON_EXTRACT
    SELECT 
        JSON_EXTRACT(data, '$.name') AS name,
        JSON_UNQUOTE(JSON_EXTRACT(data, '$.name')) AS name_text,
        data->>'$.name' AS name_shorthand  -- MySQL 8+
    FROM users;
    
    -- Query
    SELECT * FROM users
    WHERE JSON_CONTAINS(data, '"admin"', '$.tags');
    ```
    
    **SQL Server JSON:**
    
    ```sql
    -- Extract
    SELECT 
        JSON_VALUE(data, '$.name') AS name,
        JSON_QUERY(data, '$.tags') AS tags  -- Returns JSON
    FROM users;
    
    -- Query
    SELECT * FROM users
    WHERE JSON_VALUE(data, '$.name') = 'John';
    
    -- Cross apply for array expansion
    SELECT u.id, t.value AS tag
    FROM users u
    CROSS APPLY OPENJSON(u.data, '$.tags') t;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Modern data handling.
        
        **Strong answer signals:**
        
        - Knows JSON syntax varies by database
        - Uses JSONB over JSON in PostgreSQL
        - Indexes JSON fields for performance
        - Understands when to use relational vs JSON

---

### What is Full-Text Search in SQL? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Full-Text Search`, `Text Search`, `Performance` | **Asked by:** Google, Amazon, Elastic

??? success "View Answer"

    **PostgreSQL Full-Text Search:**
    
    ```sql
    -- Create text search index
    ALTER TABLE articles ADD COLUMN search_vector tsvector;
    
    UPDATE articles SET search_vector = 
        to_tsvector('english', coalesce(title,'') || ' ' || coalesce(body,''));
    
    CREATE INDEX idx_fts ON articles USING GIN(search_vector);
    
    -- Search
    SELECT * FROM articles
    WHERE search_vector @@ to_tsquery('english', 'database & performance');
    
    -- With ranking
    SELECT 
        title,
        ts_rank(search_vector, query) AS rank
    FROM articles, to_tsquery('database | postgresql') query
    WHERE search_vector @@ query
    ORDER BY rank DESC;
    
    -- Phrase search
    SELECT * FROM articles
    WHERE search_vector @@ phraseto_tsquery('full text search');
    ```
    
    **MySQL Full-Text Search:**
    
    ```sql
    -- Create FULLTEXT index
    ALTER TABLE articles
    ADD FULLTEXT INDEX ft_idx (title, body);
    
    -- Natural language search
    SELECT * FROM articles
    WHERE MATCH(title, body) AGAINST('database performance');
    
    -- Boolean mode
    SELECT * FROM articles
    WHERE MATCH(title, body) 
    AGAINST('+database -mysql' IN BOOLEAN MODE);
    
    -- With relevance score
    SELECT 
        title,
        MATCH(title, body) AGAINST('database') AS relevance
    FROM articles
    HAVING relevance > 0
    ORDER BY relevance DESC;
    ```
    
    **SQL Server Full-Text:**
    
    ```sql
    -- Create catalog and index
    CREATE FULLTEXT CATALOG ftCatalog AS DEFAULT;
    CREATE FULLTEXT INDEX ON articles(title, body) 
        KEY INDEX PK_articles;
    
    -- Search
    SELECT * FROM articles
    WHERE CONTAINS(body, 'database AND performance');
    
    SELECT * FROM articles
    WHERE FREETEXT(body, 'database optimization tips');
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Text search capabilities.
        
        **Strong answer signals:**
        
        - Knows full-text vs LIKE performance
        - Uses appropriate index types (GIN, GIST)
        - Understands stemming and stop words
        - Knows when to use dedicated search (Elasticsearch)

---

### How Do You Use Regular Expressions in SQL? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Regex`, `Pattern Matching`, `Text Processing` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **PostgreSQL Regular Expressions:**
    
    ```sql
    -- Match patterns
    SELECT * FROM users
    WHERE email ~ '^[a-z]+@[a-z]+\.[a-z]{2,}$';  -- Case sensitive
    
    SELECT * FROM users
    WHERE email ~* 'GMAIL\.COM$';  -- Case insensitive
    
    -- Not match
    SELECT * FROM users
    WHERE phone !~ '^\+1';  -- Doesn't start with +1
    
    -- Extract with regexp_match
    SELECT 
        email,
        (regexp_match(email, '^([^@]+)@(.+)$'))[1] AS username,
        (regexp_match(email, '^([^@]+)@(.+)$'))[2] AS domain
    FROM users;
    
    -- Replace
    SELECT regexp_replace(phone, '[^0-9]', '', 'g') AS digits_only
    FROM users;
    
    -- Split
    SELECT regexp_split_to_array('a,b;c|d', '[,;|]');
    ```
    
    **MySQL Regular Expressions:**
    
    ```sql
    -- Match (MySQL 8+)
    SELECT * FROM users
    WHERE email REGEXP '^[a-z]+@[a-z]+\\.[a-z]{2,}$';
    
    -- Case insensitive by default
    SELECT * FROM users
    WHERE email REGEXP 'gmail\\.com$';
    
    -- Replace (MySQL 8+)
    SELECT REGEXP_REPLACE(phone, '[^0-9]', '');
    
    -- Extract (MySQL 8+)
    SELECT REGEXP_SUBSTR(email, '@.+$');
    ```
    
    **SQL Server:**
    
    ```sql
    -- Limited native regex, use LIKE or CLR
    SELECT * FROM users
    WHERE email LIKE '%@%.%';  -- Simple pattern only
    
    -- For complex patterns, use CLR functions or 
    -- PATINDEX with wildcards
    SELECT * FROM users
    WHERE PATINDEX('%[0-9]%', phone) > 0;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Text processing ability.
        
        **Strong answer signals:**
        
        - Knows regex syntax varies by database
        - Uses regex for validation and extraction
        - Warns about performance impact
        - Knows LIKE for simple patterns

---

### What Are Sequences in SQL? - Amazon, Oracle Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Sequences`, `Auto-increment`, `Primary Keys` | **Asked by:** Amazon, Oracle, Microsoft

??? success "View Answer"

    **What is a Sequence?**
    
    A database object that generates unique numbers, commonly used for primary keys.
    
    **PostgreSQL Sequences:**
    
    ```sql
    -- Create sequence
    CREATE SEQUENCE order_seq
        START WITH 1000
        INCREMENT BY 1
        NO MAXVALUE
        CACHE 20;
    
    -- Use sequence
    INSERT INTO orders (id, product)
    VALUES (nextval('order_seq'), 'Product A');
    
    -- Get current value
    SELECT currval('order_seq');
    
    -- Serial (auto-sequence) - shorthand
    CREATE TABLE orders (
        id SERIAL PRIMARY KEY,  -- Creates sequence automatically
        product TEXT
    );
    
    -- IDENTITY (SQL standard, PostgreSQL 10+)
    CREATE TABLE orders (
        id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        product TEXT
    );
    ```
    
    **Oracle Sequences:**
    
    ```sql
    CREATE SEQUENCE order_seq
        START WITH 1
        INCREMENT BY 1
        NOCACHE;
    
    INSERT INTO orders (id, product)
    VALUES (order_seq.NEXTVAL, 'Product A');
    
    SELECT order_seq.CURRVAL FROM dual;
    ```
    
    **SQL Server:**
    
    ```sql
    -- IDENTITY (column property)
    CREATE TABLE orders (
        id INT IDENTITY(1,1) PRIMARY KEY,
        product VARCHAR(100)
    );
    
    -- Sequence (SQL Server 2012+)
    CREATE SEQUENCE order_seq
        START WITH 1
        INCREMENT BY 1;
    
    INSERT INTO orders (id, product)
    VALUES (NEXT VALUE FOR order_seq, 'Product A');
    ```
    
    **Gaps in Sequences:**
    
    ```sql
    -- Sequences can have gaps due to:
    -- - Rollbacks
    -- - Cached values
    -- - Concurrent inserts
    
    -- Reset sequence
    ALTER SEQUENCE order_seq RESTART WITH 1;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** ID generation knowledge.
        
        **Strong answer signals:**
        
        - Knows sequences vs IDENTITY differences
        - Understands gaps are normal
        - Uses CACHE for performance
        - Knows SERIAL is PostgreSQL shorthand

---

### How Do You Create and Use Temporary Tables? - Amazon, Google Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Temp Tables`, `Session Data`, `Performance` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Temporary Tables:**
    
    Tables that exist only for the session/transaction.
    
    **PostgreSQL:**
    
    ```sql
    -- Session-scoped (default)
    CREATE TEMPORARY TABLE temp_results (
        id INT,
        value TEXT
    );
    
    -- Transaction-scoped
    CREATE TEMPORARY TABLE temp_data (
        id INT
    ) ON COMMIT DROP;
    
    -- Create from query
    CREATE TEMP TABLE temp_orders AS
    SELECT * FROM orders WHERE created_at > '2024-01-01';
    
    -- Use it
    SELECT * FROM temp_orders;
    
    -- Automatically dropped at session end
    ```
    
    **SQL Server:**
    
    ```sql
    -- Local temp table (session-scoped)
    CREATE TABLE #temp_orders (
        id INT,
        product VARCHAR(100)
    );
    
    -- Global temp table (visible to all sessions)
    CREATE TABLE ##global_temp (
        id INT
    );
    
    -- Table variable (stored in memory)
    DECLARE @temp TABLE (
        id INT,
        value VARCHAR(100)
    );
    
    INSERT INTO @temp VALUES (1, 'test');
    SELECT * FROM @temp;
    ```
    
    **MySQL:**
    
    ```sql
    CREATE TEMPORARY TABLE temp_users AS
    SELECT * FROM users WHERE active = 1;
    
    -- Explicitly drop
    DROP TEMPORARY TABLE IF EXISTS temp_users;
    ```
    
    **When to Use:**
    
    | Use Case | Recommendation |
    |----------|----------------|
    | Complex multi-step queries | Temp table |
    | Store intermediate results | Temp table |
    | Small row sets | Table variable (SQL Server) |
    | Need indexes | Temp table |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Query optimization strategies.
        
        **Strong answer signals:**
        
        - Knows temp tables for multi-step processing
        - Uses CTEs for simpler cases
        - Understands session vs transaction scope
        - Knows #local vs ##global in SQL Server

---

### How Do You Perform a Safe UPDATE with JOIN? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `UPDATE`, `JOIN`, `Data Modification` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **UPDATE with JOIN:**
    
    Update one table based on values from another.
    
    **PostgreSQL / MySQL:**
    
    ```sql
    -- PostgreSQL syntax
    UPDATE orders o
    SET 
        customer_name = c.name,
        customer_email = c.email
    FROM customers c
    WHERE o.customer_id = c.id
      AND o.status = 'pending';
    
    -- MySQL syntax
    UPDATE orders o
    JOIN customers c ON o.customer_id = c.id
    SET 
        o.customer_name = c.name,
        o.customer_email = c.email
    WHERE o.status = 'pending';
    ```
    
    **SQL Server:**
    
    ```sql
    UPDATE o
    SET 
        o.customer_name = c.name,
        o.customer_email = c.email
    FROM orders o
    INNER JOIN customers c ON o.customer_id = c.id
    WHERE o.status = 'pending';
    ```
    
    **Safe Update Practices:**
    
    ```sql
    -- 1. Preview with SELECT first
    SELECT o.id, o.customer_name, c.name AS new_name
    FROM orders o
    JOIN customers c ON o.customer_id = c.id
    WHERE o.status = 'pending';
    
    -- 2. Use transaction
    BEGIN;
    UPDATE orders o
    SET customer_name = c.name
    FROM customers c
    WHERE o.customer_id = c.id;
    
    -- Verify
    SELECT * FROM orders WHERE customer_name IS NULL;
    
    -- Commit or rollback
    COMMIT;  -- or ROLLBACK;
    
    -- 3. Limit rows for testing
    UPDATE orders
    SET status = 'processed'
    WHERE id IN (
        SELECT id FROM orders 
        WHERE status = 'pending' 
        LIMIT 10
    );
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Safe data modification.
        
        **Strong answer signals:**
        
        - Always previews with SELECT first
        - Uses transactions for safety
        - Knows UPDATE syntax varies by database
        - Tests on limited rows before full update

---

### What is the Difference Between HAVING and WHERE? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Filtering`, `Aggregation`, `SQL Basics` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **WHERE vs HAVING:**
    
    | Aspect | WHERE | HAVING |
    |--------|-------|--------|
    | Filters | Row-level before grouping | Groups after aggregation |
    | Can use aggregates | No | Yes |
    | Executed | Before GROUP BY | After GROUP BY |
    | Performance | Reduces data early | Filters after calculation |
    
    ```sql
    -- WHERE: filters rows before grouping
    SELECT category, COUNT(*) AS product_count
    FROM products
    WHERE price > 10  -- Filter individual rows
    GROUP BY category;
    
    -- HAVING: filters groups after aggregation
    SELECT category, COUNT(*) AS product_count
    FROM products
    GROUP BY category
    HAVING COUNT(*) > 5;  -- Filter groups
    
    -- Combined: both WHERE and HAVING
    SELECT 
        category,
        AVG(price) AS avg_price,
        COUNT(*) AS count
    FROM products
    WHERE active = true          -- First: filter rows
    GROUP BY category
    HAVING AVG(price) > 100;     -- Then: filter groups
    ```
    
    **Common Mistake:**
    
    ```sql
    -- ❌ Wrong: Can't use aggregate in WHERE
    SELECT category, AVG(price)
    FROM products
    WHERE AVG(price) > 100  -- ERROR!
    GROUP BY category;
    
    -- ✅ Correct: Use HAVING for aggregates
    SELECT category, AVG(price)
    FROM products
    GROUP BY category
    HAVING AVG(price) > 100;
    
    -- ❌ Inefficient: Filtering rows in HAVING
    SELECT category, COUNT(*)
    FROM products
    GROUP BY category
    HAVING price > 10;  -- Should be in WHERE!
    
    -- ✅ Better: Filter early
    SELECT category, COUNT(*)
    FROM products
    WHERE price > 10
    GROUP BY category;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Aggregation understanding.
        
        **Strong answer signals:**
        
        - Knows WHERE is for rows, HAVING for groups
        - Filters as early as possible (WHERE)
        - Uses correct syntax for aggregates
        - Understands execution order

---

### How Do You Handle Duplicate Detection and Removal? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Duplicates`, `Data Quality`, `Deduplication` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Find Duplicates:**
    
    ```sql
    -- Find duplicate rows
    SELECT email, COUNT(*) AS count
    FROM users
    GROUP BY email
    HAVING COUNT(*) > 1;
    
    -- Find duplicate with all columns
    SELECT *, COUNT(*) OVER (PARTITION BY email) AS dup_count
    FROM users
    WHERE email IN (
        SELECT email FROM users
        GROUP BY email HAVING COUNT(*) > 1
    );
    
    -- Find duplicate pairs
    SELECT 
        u1.id AS id1, u2.id AS id2, 
        u1.email
    FROM users u1
    JOIN users u2 ON u1.email = u2.email AND u1.id < u2.id;
    ```
    
    **Remove Duplicates (Keep One):**
    
    ```sql
    -- Delete keeping lowest ID
    DELETE FROM users
    WHERE id NOT IN (
        SELECT MIN(id)
        FROM users
        GROUP BY email
    );
    
    -- PostgreSQL: Using ctid (row identifier)
    DELETE FROM users a
    USING users b
    WHERE a.ctid > b.ctid
      AND a.email = b.email;
    
    -- SQL Server: Using ROW_NUMBER
    WITH cte AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS rn
        FROM users
    )
    DELETE FROM cte WHERE rn > 1;
    
    -- PostgreSQL equivalent
    DELETE FROM users
    WHERE id IN (
        SELECT id FROM (
            SELECT id,
                ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS rn
            FROM users
        ) t
        WHERE rn > 1
    );
    ```
    
    **Prevent Duplicates:**
    
    ```sql
    -- Unique constraint
    ALTER TABLE users ADD CONSTRAINT unique_email UNIQUE (email);
    
    -- Insert ignore duplicates
    INSERT INTO users (email, name)
    VALUES ('test@example.com', 'Test')
    ON CONFLICT (email) DO NOTHING;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data quality skills.
        
        **Strong answer signals:**
        
        - Uses window functions for efficient dedup
        - Previews before deleting
        - Knows to add constraints to prevent
        - Handles which duplicate to keep

---

### What is Database Normalization? Explain 1NF, 2NF, 3NF - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Normalization`, `Database Design`, `Schema` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Normalization:**
    
    Process of organizing data to reduce redundancy and improve integrity.
    
    **1NF (First Normal Form):**
    
    - Atomic values (no arrays/lists in cells)
    - Unique rows (primary key)
    
    ```sql
    -- ❌ Not 1NF
    -- id | name  | phones
    -- 1  | John  | 123-456, 789-012
    
    -- ✅ 1NF
    -- users: id, name
    -- phones: id, user_id, phone
    ```
    
    **2NF (Second Normal Form):**
    
    - 1NF + No partial dependencies
    - All non-key columns depend on entire primary key
    
    ```sql
    -- ❌ Not 2NF (partial dependency)
    -- order_items: order_id, product_id, product_name, quantity
    -- product_name depends only on product_id, not full key
    
    -- ✅ 2NF
    -- order_items: order_id, product_id, quantity
    -- products: product_id, product_name
    ```
    
    **3NF (Third Normal Form):**
    
    - 2NF + No transitive dependencies
    - Non-key columns don't depend on other non-key columns
    
    ```sql
    -- ❌ Not 3NF (transitive dependency)
    -- employees: id, department_id, department_name
    -- department_name depends on department_id (non-key)
    
    -- ✅ 3NF
    -- employees: id, department_id
    -- departments: id, name
    ```
    
    **When to Denormalize:**
    
    | Normalize | Denormalize |
    |-----------|-------------|
    | OLTP (transactions) | OLAP (analytics) |
    | Data integrity critical | Read performance critical |
    | Storage is limited | Complex joins hurt performance |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Database design fundamentals.
        
        **Strong answer signals:**
        
        - Explains normal forms progressively
        - Gives concrete examples
        - Knows when to denormalize
        - Understands trade-offs

---

### How Do You Calculate Running Totals and Moving Averages? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Window Functions`, `Analytics`, `Time Series` | **Asked by:** Google, Amazon, Netflix

??? success "View Answer"

    **Running Total:**
    
    ```sql
    SELECT 
        date,
        amount,
        SUM(amount) OVER (ORDER BY date) AS running_total
    FROM transactions;
    
    -- Running total per category
    SELECT 
        category,
        date,
        amount,
        SUM(amount) OVER (
            PARTITION BY category 
            ORDER BY date
        ) AS category_running_total
    FROM transactions;
    
    -- Running total with reset each month
    SELECT 
        date,
        amount,
        SUM(amount) OVER (
            PARTITION BY DATE_TRUNC('month', date)
            ORDER BY date
        ) AS monthly_running_total
    FROM transactions;
    ```
    
    **Moving Average:**
    
    ```sql
    -- 7-day moving average
    SELECT 
        date,
        value,
        AVG(value) OVER (
            ORDER BY date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS moving_avg_7d
    FROM daily_metrics;
    
    -- Moving average by rows
    SELECT 
        date,
        value,
        AVG(value) OVER (
            ORDER BY date
            ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
        ) AS centered_avg
    FROM daily_metrics;
    
    -- Exponential moving average (approximation)
    WITH recursive_ema AS (
        SELECT 
            date, 
            value,
            value AS ema
        FROM daily_metrics
        WHERE date = (SELECT MIN(date) FROM daily_metrics)
        
        UNION ALL
        
        SELECT 
            d.date,
            d.value,
            0.2 * d.value + 0.8 * r.ema  -- alpha = 0.2
        FROM daily_metrics d
        JOIN recursive_ema r 
            ON d.date = r.date + INTERVAL '1 day'
    )
    SELECT * FROM recursive_ema;
    ```
    
    **ROWS vs RANGE:**
    
    | Frame | Behavior |
    |-------|----------|
    | ROWS | Exact number of rows |
    | RANGE | All rows with same value |
    | GROUPS | Number of peer groups |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Time series analysis in SQL.
        
        **Strong answer signals:**
        
        - Uses window frames correctly
        - Knows ROWS vs RANGE difference
        - Can calculate various moving stats
        - Understands PARTITION BY for grouping

---

### How Do You Find Gaps in Sequential Data? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Gaps`, `Sequences`, `Data Quality` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Find Gaps in ID Sequence:**
    
    ```sql
    -- Using window function
    SELECT 
        id + 1 AS gap_start,
        next_id - 1 AS gap_end
    FROM (
        SELECT 
            id,
            LEAD(id) OVER (ORDER BY id) AS next_id
        FROM orders
    ) t
    WHERE next_id - id > 1;
    
    -- Alternative: generate series
    SELECT s.id AS missing_id
    FROM generate_series(
        (SELECT MIN(id) FROM orders),
        (SELECT MAX(id) FROM orders)
    ) s(id)
    LEFT JOIN orders o ON s.id = o.id
    WHERE o.id IS NULL;
    ```
    
    **Find Gaps in Dates:**
    
    ```sql
    -- Days without transactions
    WITH date_range AS (
        SELECT generate_series(
            (SELECT MIN(transaction_date) FROM transactions),
            (SELECT MAX(transaction_date) FROM transactions),
            INTERVAL '1 day'
        )::date AS date
    )
    SELECT d.date AS missing_date
    FROM date_range d
    LEFT JOIN transactions t ON d.date = t.transaction_date
    WHERE t.transaction_date IS NULL;
    
    -- Using window function
    SELECT 
        transaction_date AS gap_start,
        next_date AS gap_end,
        next_date - transaction_date - 1 AS days_missing
    FROM (
        SELECT 
            transaction_date,
            LEAD(transaction_date) OVER (ORDER BY transaction_date) AS next_date
        FROM (SELECT DISTINCT transaction_date FROM transactions) t
    ) gaps
    WHERE next_date - transaction_date > 1;
    ```
    
    **Find Gaps in Time Ranges (Islands):**
    
    ```sql
    -- Identify islands of continuous data
    WITH numbered AS (
        SELECT 
            *,
            date - (ROW_NUMBER() OVER (ORDER BY date))::int AS grp
        FROM daily_data
    )
    SELECT 
        MIN(date) AS island_start,
        MAX(date) AS island_end,
        COUNT(*) AS days_in_island
    FROM numbered
    GROUP BY grp
    ORDER BY island_start;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data quality analysis.
        
        **Strong answer signals:**
        
        - Uses LEAD/LAG for gap detection
        - Uses generate_series for completeness checks
        - Knows islands and gaps problem
        - Understands performance implications

---

### How Do You Write Efficient Pagination Queries? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Pagination`, `Performance`, `Large Tables` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **OFFSET Pagination (Simple but Slow):**
    
    ```sql
    -- Page 1
    SELECT * FROM products ORDER BY id LIMIT 20 OFFSET 0;
    
    -- Page 50 (slow - scans 1000 rows!)
    SELECT * FROM products ORDER BY id LIMIT 20 OFFSET 980;
    
    -- Problem: OFFSET scans and discards rows
    -- O(n) complexity for large offsets
    ```
    
    **Keyset Pagination (Fast):**
    
    ```sql
    -- First page
    SELECT * FROM products
    ORDER BY id
    LIMIT 20;
    
    -- Next page (after id 20)
    SELECT * FROM products
    WHERE id > 20  -- Last ID from previous page
    ORDER BY id
    LIMIT 20;
    
    -- With multiple columns
    SELECT * FROM products
    WHERE (created_at, id) > ('2024-01-15', 100)
    ORDER BY created_at, id
    LIMIT 20;
    
    -- Previous page
    SELECT * FROM (
        SELECT * FROM products
        WHERE id < 20
        ORDER BY id DESC
        LIMIT 20
    ) t
    ORDER BY id ASC;
    ```
    
    **Indexed Pagination:**
    
    ```sql
    -- Requires covering index
    CREATE INDEX idx_products_pagination 
    ON products (created_at DESC, id DESC);
    
    -- Very fast with index
    SELECT id, name, price, created_at
    FROM products
    WHERE (created_at, id) < ('2024-01-15', 100)
    ORDER BY created_at DESC, id DESC
    LIMIT 20;
    ```
    
    **Comparison:**
    
    | Method | Page 1 | Page 1000 | Jump to page |
    |--------|--------|-----------|--------------|
    | OFFSET | Fast | Very slow | Easy |
    | Keyset | Fast | Fast | Hard |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Performance awareness.
        
        **Strong answer signals:**
        
        - Knows OFFSET performance issue
        - Uses keyset for large datasets
        - Understands index requirements
        - Trade-offs: keyset can't jump to page N

---

### What is Data Warehouse Star Schema? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Warehouse`, `Star Schema`, `OLAP` | **Asked by:** Amazon, Google, Snowflake

??? success "View Answer"

    **Star Schema:**
    
    Dimensional modeling with central fact table surrounded by dimension tables.
    
    ```sql
    -- Fact table (metrics/events)
    CREATE TABLE fact_sales (
        sale_id SERIAL PRIMARY KEY,
        date_key INT REFERENCES dim_date(date_key),
        product_key INT REFERENCES dim_product(product_key),
        store_key INT REFERENCES dim_store(store_key),
        customer_key INT REFERENCES dim_customer(customer_key),
        
        -- Measures
        quantity INT,
        unit_price DECIMAL(10,2),
        total_amount DECIMAL(10,2),
        discount DECIMAL(10,2)
    );
    
    -- Dimension tables (descriptive attributes)
    CREATE TABLE dim_date (
        date_key INT PRIMARY KEY,
        full_date DATE,
        year INT,
        quarter INT,
        month INT,
        month_name VARCHAR(20),
        week INT,
        day_of_week INT,
        is_weekend BOOLEAN,
        is_holiday BOOLEAN
    );
    
    CREATE TABLE dim_product (
        product_key INT PRIMARY KEY,
        product_id VARCHAR(50),
        product_name VARCHAR(200),
        category VARCHAR(100),
        subcategory VARCHAR(100),
        brand VARCHAR(100)
    );
    ```
    
    **Query Example:**
    
    ```sql
    -- Sales by category and quarter
    SELECT 
        p.category,
        d.year,
        d.quarter,
        SUM(f.total_amount) AS total_sales,
        COUNT(DISTINCT f.sale_id) AS transaction_count
    FROM fact_sales f
    JOIN dim_date d ON f.date_key = d.date_key
    JOIN dim_product p ON f.product_key = p.product_key
    WHERE d.year = 2024
    GROUP BY p.category, d.year, d.quarter
    ORDER BY p.category, d.quarter;
    ```
    
    **Star vs Snowflake Schema:**
    
    | Star | Snowflake |
    |------|-----------|
    | Denormalized dimensions | Normalized dimensions |
    | Fewer joins | More joins |
    | Faster queries | Less redundancy |
    | More storage | Complex structure |

    !!! tip "Interviewer's Insight"
        **What they're testing:** DW design understanding.
        
        **Strong answer signals:**
        
        - Knows fact vs dimension distinction
        - Uses surrogate keys
        - Understands star vs snowflake trade-offs
        - Can design for specific analytics needs

---

### How Do You Optimize COUNT(*) on Large Tables? - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Performance`, `Counting`, `Optimization` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Problem with COUNT(*):**
    
    ```sql
    -- On 100M rows, this is SLOW
    SELECT COUNT(*) FROM large_table;
    -- Must scan entire table (or index)
    ```
    
    **Optimization Strategies:**
    
    **1. Table Statistics (Approximate):**
    
    ```sql
    -- PostgreSQL: estimated count from planner
    SELECT reltuples::bigint AS estimate
    FROM pg_class
    WHERE relname = 'large_table';
    
    -- MySQL
    SELECT TABLE_ROWS
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME = 'large_table';
    ```
    
    **2. Covering Index:**
    
    ```sql
    -- PostgreSQL index-only scan
    CREATE INDEX idx_count ON large_table (id);
    
    -- Vacuum to update visibility map
    VACUUM large_table;
    
    SELECT COUNT(*) FROM large_table;
    -- Uses index-only scan if visibility map is current
    ```
    
    **3. Materialized Count:**
    
    ```sql
    -- Maintain count in separate table
    CREATE TABLE table_counts (
        table_name VARCHAR(100) PRIMARY KEY,
        row_count BIGINT,
        updated_at TIMESTAMP
    );
    
    -- Update with trigger or periodic job
    CREATE OR REPLACE FUNCTION update_count()
    RETURNS TRIGGER AS $$
    BEGIN
        IF TG_OP = 'INSERT' THEN
            UPDATE table_counts SET row_count = row_count + 1
            WHERE table_name = 'large_table';
        ELSIF TG_OP = 'DELETE' THEN
            UPDATE table_counts SET row_count = row_count - 1
            WHERE table_name = 'large_table';
        END IF;
        RETURN NULL;
    END; $$ LANGUAGE plpgsql;
    ```
    
    **4. HyperLogLog (Approximate Distinct):**
    
    ```sql
    -- PostgreSQL with extension
    CREATE EXTENSION IF NOT EXISTS hll;
    
    -- Approximate distinct count
    SELECT hll_cardinality(hll_add_agg(hll_hash_bigint(id)))
    FROM large_table;
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Large-scale optimization.
        
        **Strong answer signals:**
        
        - Uses statistics for estimates
        - Knows index-only scan requirements
        - Maintains materialized counts for exact needs
        - Uses approximate algorithms when acceptable

---

## Quick Reference: 100+ Interview Questions

| Sno | Question Title                                     | Practice Links                                                                 | Companies Asking        | Difficulty | Topics                                  |
|-----|----------------------------------------------------|--------------------------------------------------------------------------------|-------------------------|------------|-----------------------------------------|
|-----|----------------------------------------------------|--------------------------------------------------------------------------------|-------------------------|------------|-----------------------------------------|
| 1   | Difference between `DELETE`, `TRUNCATE`, and `DROP`  | [GeeksforGeeks](https://www.geeksforgeeks.org/difference-between-delete-truncate-and-drop-in-sql/) | Most Tech Companies     | Easy       | DDL, DML                                |
| 2   | Types of SQL Joins (INNER, LEFT, RIGHT, FULL)      | [W3Schools](https://www.w3schools.com/sql/sql_join.asp)                        | Google, Amazon, Meta    | Easy       | Joins                                   |
| 3   | What is Normalization? Explain different forms.    | [StudyTonight](https://www.studytonight.com/dbms/database-normalization.php) | Microsoft, Oracle, IBM  | Medium     | Database Design, Normalization          |
| 4   | Explain Primary Key vs Foreign Key vs Unique Key   | [GeeksforGeeks](https://www.geeksforgeeks.org/difference-between-primary-key-and-foreign-key/) | Most Tech Companies     | Easy       | Constraints, Database Design            |
| 5   | What are Indexes and why are they important?       | [Essential SQL](https://essentialsSQL.com/what-is-a-database-index/)           | Google, Amazon, Netflix | Medium     | Performance Optimization, Indexes       |
| 6   | Write a query to find the Nth highest salary.      | [LeetCode](https://leetcode.com/problems/nth-highest-salary/)                  | Amazon, Microsoft, Uber | Medium     | Subqueries, Window Functions, Ranking   |
| 7   | Explain ACID properties in Databases.              | [GeeksforGeeks](https://www.geeksforgeeks.org/acid-properties-in-dbms/)        | Oracle, SAP, Banks      | Medium     | Transactions, Database Fundamentals     |
| 8   | What is a Subquery? Types of Subqueries.           | [SQLTutorial.org](https://www.sqltutorial.org/sql-subquery/)                   | Meta, Google, LinkedIn  | Medium     | Subqueries, Query Structure             |
| 9   | Difference between `UNION` and `UNION ALL`.        | [W3Schools](https://www.w3schools.com/sql/sql_union.asp)                       | Most Tech Companies     | Easy       | Set Operations                          |
| 10  | What are Window Functions? Give examples.          | [Mode Analytics](https://mode.com/sql-tutorial/sql-window-functions/)          | Netflix, Airbnb, Spotify| Hard       | Window Functions, Advanced SQL          |
| 11  | Explain Common Table Expressions (CTEs).           | [SQLShack](https://www.sqlshack.com/sql-server-common-table-expressions-cte/)  | Microsoft, Google       | Medium     | CTEs, Query Readability                 |
| 12  | How to handle NULL values in SQL?                  | [SQL Authority](https://blog.sqlauthority.com/2007/03/29/sql-server-handling-null-values/) | Most Tech Companies     | Easy       | NULL Handling, Functions (COALESCE, ISNULL) |
| 13  | What is SQL Injection and how to prevent it?       | [OWASP](https://owasp.org/www-community/attacks/SQL_Injection)                 | All Security-Conscious  | Medium     | Security, Best Practices                |
| 14  | Difference between `GROUP BY` and `PARTITION BY`.  | [Stack Overflow](https://stackoverflow.com/questions/13387311/whats-the-difference-between-group-by-and-partition-by) | Advanced Roles          | Hard       | Aggregation, Window Functions           |
| 15  | Write a query to find duplicate records in a table.| [GeeksforGeeks](https://www.geeksforgeeks.org/sql-query-to-find-duplicate-rows-in-a-table/) | Data Quality Roles      | Medium     | Aggregation, GROUP BY, HAVING           |
| 16  | Difference between `WHERE` and `HAVING` clause.    | [SQLTutorial.org](https://www.sqltutorial.org/sql-having/)                     | Most Tech Companies     | Easy       | Filtering, Aggregation                  |
| 17  | What are Triggers? Give an example.                | [GeeksforGeeks](https://www.geeksforgeeks.org/sql-trigger/)                    | Database Roles          | Medium     | Triggers, Automation                    |
| 18  | Explain different types of relationships (1:1, 1:N, N:M). | [Lucidchart](https://www.lucidchart.com/pages/database-diagram/database-relationships) | Most Tech Companies     | Easy       | Database Design, Relationships          |
| 19  | What is a View in SQL?                           | [W3Schools](https://www.w3schools.com/sql/sql_view.asp)                        | Google, Microsoft       | Easy       | Views, Abstraction                      |
| 20  | How to optimize a slow SQL query?                | [Several Resources]                                                            | Performance Engineers   | Hard       | Performance Tuning, Optimization        |
| 21  | Difference between `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`. | [SQLShack](https://www.sqlshack.com/overview-of-sql-rank-functions/)           | Data Analysts, Scientists | Medium     | Window Functions, Ranking               |
| 22  | What is Database Denormalization? When to use it? | [GeeksforGeeks](https://www.geeksforgeeks.org/denormalization-in-databases-dbms/) | Performance-critical Apps | Medium     | Database Design, Performance            |
| 23  | Explain Stored Procedures. Advantages?             | [SQLTutorial.org](https://www.sqltutorial.org/sql-stored-procedures/)          | Oracle, Microsoft       | Medium     | Stored Procedures, Reusability          |
| 24  | How does `BETWEEN` operator work?                | [W3Schools](https://www.w3schools.com/sql/sql_between.asp)                     | Most Tech Companies     | Easy       | Operators, Filtering                    |
| 25  | What is the `CASE` statement used for?           | [W3Schools](https://www.w3schools.com/sql/sql_case.asp)                        | Most Tech Companies     | Easy       | Conditional Logic                       |
| 26  | Explain Self Join with an example.                 | [GeeksforGeeks](https://www.geeksforgeeks.org/sql-self-join/)                  | Amazon, Meta            | Medium     | Joins                                   |
| 27  | What is the purpose of `DISTINCT` keyword?       | [W3Schools](https://www.w3schools.com/sql/sql_distinct.asp)                    | Most Tech Companies     | Easy       | Deduplication, Querying                 |
| 28  | How to find the second highest value?            | [Various Methods]                                                              | Common Interview Q      | Medium     | Subqueries, Window Functions            |
| 29  | What is Referential Integrity?                   | [Techopedia](https://www.techopedia.com/definition/24220/referential-integrity) | Database Roles          | Medium     | Constraints, Data Integrity             |
| 30  | Explain `EXISTS` and `NOT EXISTS` operators.     | [SQLTutorial.org](https://www.sqltutorial.org/sql-exists/)                     | Google, LinkedIn        | Medium     | Subqueries, Operators                   |
| 31  | What is a Schema in a database?                  | [Wikipedia](https://en.wikipedia.org/wiki/Database_schema)                   | Most Tech Companies     | Easy       | Database Concepts                       |
| 32  | Difference between `CHAR` and `VARCHAR` data types. | [GeeksforGeeks](https://www.geeksforgeeks.org/difference-between-char-and-varchar-in-sql/) | Most Tech Companies     | Easy       | Data Types, Storage                     |
| 33  | How to concatenate strings in SQL?               | [Database.Guide](https://database.guide/how-to-concatenate-strings-in-sql/)    | Most Tech Companies     | Easy       | String Manipulation                     |
| 34  | What is Data Warehousing?                        | [IBM](https://www.ibm.com/cloud/learn/data-warehouse)                          | BI Roles, Data Engineers| Medium     | Data Warehousing, BI                  |
| 35  | Explain ETL (Extract, Transform, Load) process.  | [AWS](https://aws.amazon.com/what-is/etl/)                                     | Data Engineers          | Medium     | ETL, Data Integration                   |
| 36  | What are Aggregate Functions? List some.         | [W3Schools](https://www.w3schools.com/sql/sql_agg_functions.asp)               | Most Tech Companies     | Easy       | Aggregation                             |
| 37  | How to handle transactions (COMMIT, ROLLBACK)?   | [SQLTutorial.org](https://www.sqltutorial.org/sql-transaction/)                | Database Developers     | Medium     | Transactions, ACID                      |
| 38  | What is Database Sharding?                       | [DigitalOcean](https://www.digitalocean.com/community/tutorials/understanding-database-sharding) | Scalability Roles       | Hard       | Scalability, Database Architecture      |
| 39  | Explain Database Replication.                    | [Wikipedia](https://en.wikipedia.org/wiki/Replication_(computing)#Database_replication) | High Availability Roles | Hard       | High Availability, Replication          |
| 40  | What is the `LIKE` operator used for?            | [W3Schools](https://www.w3schools.com/sql/sql_like.asp)                        | Most Tech Companies     | Easy       | Pattern Matching, Filtering             |
| 41  | Difference between `COUNT(*)` and `COUNT(column)`. | [Stack Overflow](https://stackoverflow.com/questions/10088520/count-vs-countcolumn-name-which-is-more-correct) | Most Tech Companies     | Easy       | Aggregation, NULL Handling              |
| 42  | What is a Candidate Key?                         | [GeeksforGeeks](https://www.geeksforgeeks.org/candidate-key-in-dbms/)          | Database Design Roles   | Medium     | Keys, Database Design                   |
| 43  | Explain Super Key.                               | [GeeksforGeeks](https://www.geeksforgeeks.org/super-key-in-dbms/)              | Database Design Roles   | Medium     | Keys, Database Design                   |
| 44  | What is Composite Key?                           | [GeeksforGeeks](https://www.geeksforgeeks.org/composite-key-in-dbms/)          | Database Design Roles   | Medium     | Keys, Database Design                   |
| 45  | How to get the current date and time in SQL?     | [Varies by RDBMS]                                                              | Most Tech Companies     | Easy       | Date/Time Functions                     |
| 46  | What is the purpose of `ALTER TABLE` statement?  | [W3Schools](https://www.w3schools.com/sql/sql_alter.asp)                       | Database Admins/Devs    | Easy       | DDL, Schema Modification              |
| 47  | Explain `CHECK` constraint.                      | [W3Schools](https://www.w3schools.com/sql/sql_check.asp)                       | Database Design Roles   | Easy       | Constraints, Data Integrity             |
| 48  | What is `DEFAULT` constraint?                    | [W3Schools](https://www.w3schools.com/sql/sql_default.asp)                     | Database Design Roles   | Easy       | Constraints, Default Values             |
| 49  | How to create a temporary table?                 | [Varies by RDBMS]                                                              | Developers              | Medium     | Temporary Storage, Complex Queries      |
| 50  | What is SQL Injection? (Revisited for emphasis)  | [OWASP](https://owasp.org/www-community/attacks/SQL_Injection)                 | All Roles               | Medium     | Security                                |
| 51  | Explain Cross Join. When is it useful?           | [W3Schools](https://www.w3schools.com/sql/sql_join_cross.asp)                  | Specific Scenarios      | Medium     | Joins, Cartesian Product              |
| 52  | What is the difference between Function and Stored Procedure? | [GeeksforGeeks](https://www.geeksforgeeks.org/difference-between-stored-procedure-and-function-in-sql/) | Database Developers     | Medium     | Functions, Stored Procedures            |
| 53  | How to find the length of a string?              | [Varies by RDBMS]                                                              | Most Tech Companies     | Easy       | String Functions                        |
| 54  | What is the `HAVING` clause used for?            | [W3Schools](https://www.w3schools.com/sql/sql_having.asp)                      | Most Tech Companies     | Easy       | Filtering Aggregates                    |
| 55  | Explain database locking mechanisms.             | [Wikipedia](https://en.wikipedia.org/wiki/Lock_(database))                   | Database Admins/Archs   | Hard       | Concurrency Control                     |
| 56  | What are Isolation Levels in transactions?       | [GeeksforGeeks](https://www.geeksforgeeks.org/transaction-isolation-levels-in-dbms/) | Database Developers     | Hard       | Transactions, Concurrency               |
| 57  | How to perform conditional aggregation?          | [SQL Authority](https://blog.sqlauthority.com/2010/07/21/sql-server-conditional-aggregations-using-case-statement/) | Data Analysts           | Medium     | Aggregation, Conditional Logic          |
| 58  | What is a Pivot Table in SQL?                    | [SQLShack](https://www.sqlshack.com/dynamic-pivot-tables-in-sql-server/)       | Data Analysts, BI Roles | Hard       | Data Transformation, Reporting          |
| 59  | Explain the `MERGE` statement.                   | [Microsoft Docs](https://docs.microsoft.com/en-us/sql/t-sql/statements/merge-transact-sql) | SQL Server Devs         | Medium     | DML, Upsert Operations                  |
| 60  | How to handle errors in SQL (e.g., TRY...CATCH)? | [Microsoft Docs](https://docs.microsoft.com/en-us/sql/t-sql/language-elements/try-catch-transact-sql) | SQL Server Devs         | Medium     | Error Handling                          |
| 61  | What is Dynamic SQL? Pros and Cons?              | [SQLShack](https://www.sqlshack.com/dynamic-sql-in-sql-server/)                | Advanced SQL Devs       | Hard       | Dynamic Queries, Flexibility, Security  |
| 62  | Explain Full-Text Search.                        | [Wikipedia](https://en.wikipedia.org/wiki/Full-text_search)                  | Search Functionality    | Medium     | Indexing, Searching Text                |
| 63  | How to work with JSON data in SQL?               | [Varies by RDBMS]                                                              | Modern App Devs         | Medium     | JSON Support, Data Handling             |
| 64  | What is Materialized View?                       | [Wikipedia](https://en.wikipedia.org/wiki/Materialized_view)                 | Performance Optimization| Hard       | Views, Performance                      |
| 65  | Difference between OLTP and OLAP.                | [GeeksforGeeks](https://www.geeksforgeeks.org/difference-between-oltp-and-olap/) | DB Architects, BI Roles | Medium     | Database Systems, Use Cases             |
| 66  | How to calculate running totals?                 | [Mode Analytics](https://mode.com/sql-tutorial/sql-window-functions/#running-totals) | Data Analysts           | Medium     | Window Functions, Aggregation           |
| 67  | What is a Sequence in SQL?                       | [Oracle Docs](https://docs.oracle.com/cd/B19306_01/server.102/b14200/statements_6015.htm) | Oracle/Postgres Devs    | Medium     | Sequence Generation                     |
| 68  | Explain Recursive CTEs.                          | [SQLTutorial.org](https://www.sqltutorial.org/sql-recursive-cte/)              | Advanced SQL Devs       | Hard       | CTEs, Hierarchical Data                 |
| 69  | How to find the median value in SQL?             | [Stack Overflow](https://stackoverflow.com/questions/1290301/sql-query-to-get-median) | Data Analysts           | Hard       | Statistics, Window Functions            |
| 70  | What is Query Execution Plan?                    | [Wikipedia](https://en.wikipedia.org/wiki/Query_plan)                        | Performance Tuning      | Medium     | Query Optimization, Performance         |
| 71  | How to use `COALESCE` or `ISNULL`?             | [W3Schools](https://www.w3schools.com/sql/sql_isnull.asp)                      | Most Tech Companies     | Easy       | NULL Handling                           |
| 72  | What is B-Tree Index?                            | [Wikipedia](https://en.wikipedia.org/wiki/B-tree)                            | Database Internals      | Medium     | Indexes, Data Structures                |
| 73  | Explain Hash Index.                              | [PostgreSQL Docs](https://www.postgresql.org/docs/current/indexes-hash.html)   | Database Internals      | Medium     | Indexes, Data Structures                |
| 74  | Difference between Clustered and Non-Clustered Index. | [GeeksforGeeks](https://www.geeksforgeeks.org/clustered-vs-non-clustered-index/) | Database Performance    | Medium     | Indexes, Performance                    |
| 75  | How to grant and revoke permissions?             | [W3Schools](https://www.w3schools.com/sql/sql_grant.asp)                       | Database Admins         | Easy       | Security, Access Control                |
| 76  | What is SQL Profiler / Tracing?                  | [Microsoft Docs](https://docs.microsoft.com/en-us/sql/tools/sql-server-profiler/sql-server-profiler) | Performance Tuning      | Medium     | Monitoring, Debugging                   |
| 77  | Explain database constraints (NOT NULL, UNIQUE, etc.). | [W3Schools](https://www.w3schools.com/sql/sql_constraints.asp)                 | Most Tech Companies     | Easy       | Constraints, Data Integrity             |
| 78  | How to update multiple rows with different values? | [Stack Overflow](https://stackoverflow.com/questions/18797608/update-multiple-rows-in-one-query-using-postgresql) | Developers              | Medium     | DML, Updates                            |
| 79  | What is database normalization (revisited)?      | [StudyTonight](https://www.studytonight.com/dbms/database-normalization.php) | All Roles               | Medium     | Database Design                         |
| 80  | Explain 1NF, 2NF, 3NF, BCNF.                   | [GeeksforGeeks](https://www.geeksforgeeks.org/normal-forms-in-dbms/)           | Database Design Roles   | Medium     | Normalization Forms                     |
| 81  | How to delete duplicate rows?                    | [GeeksforGeeks](https://www.geeksforgeeks.org/sql-delete-duplicate-rows/)      | Data Cleaning Roles     | Medium     | DML, Data Quality                       |
| 82  | What is the `INTERSECT` operator?                | [W3Schools](https://www.w3schools.com/sql/sql_intersect.asp)                   | Set Operations Roles    | Medium     | Set Operations                          |
| 83  | What is the `EXCEPT` / `MINUS` operator?         | [W3Schools](https://www.w3schools.com/sql/sql_minus.asp)                       | Set Operations Roles    | Medium     | Set Operations                          |
| 84  | How to handle large objects (BLOB, CLOB)?        | [Oracle Docs](https://docs.oracle.com/database/121/ADLOB/lob_concepts.htm#ADLOB45168) | Specific Applications   | Medium     | Data Types, Large Data                  |
| 85  | What is database connection pooling?             | [Wikipedia](https://en.wikipedia.org/wiki/Connection_pool)                   | Application Developers  | Medium     | Performance, Resource Management        |
| 86  | Explain CAP Theorem.                             | [Wikipedia](https://en.wikipedia.org/wiki/CAP_theorem)                       | Distributed Systems     | Hard       | Distributed Databases, Tradeoffs        |
| 87  | How to perform date/time arithmetic?             | [Varies by RDBMS]                                                              | Most Tech Companies     | Easy       | Date/Time Functions                     |
| 88  | What is a correlated subquery?                   | [GeeksforGeeks](https://www.geeksforgeeks.org/sql-correlated-subqueries/)      | Advanced SQL Users      | Medium     | Subqueries, Performance Considerations  |
| 89  | How to use `GROUPING SETS`, `CUBE`, `ROLLUP`?    | [SQLShack](https://www.sqlshack.com/sql-server-grouping-sets-feature/)         | BI / Analytics Roles    | Hard       | Advanced Aggregation                    |
| 90  | What is Parameter Sniffing (SQL Server)?         | [Brent Ozar](https://www.brentozar.com/blitz/parameter-sniffing/)              | SQL Server DBAs/Devs    | Hard       | Performance Tuning (SQL Server)         |
| 91  | How to create and use User-Defined Functions (UDFs)? | [Varies by RDBMS]                                                              | Database Developers     | Medium     | Functions, Reusability                  |
| 92  | What is database auditing?                       | [Wikipedia](https://en.wikipedia.org/wiki/Database_auditing)                 | Security/Compliance Roles| Medium     | Security, Monitoring                    |
| 93  | Explain optimistic vs. pessimistic locking.      | [Stack Overflow](https://stackoverflow.com/questions/129329/optimistic-vs-pessimistic-locking) | Concurrent Applications | Hard       | Concurrency Control                     |
| 94  | How to handle deadlocks?                         | [Microsoft Docs](https://docs.microsoft.com/en-us/sql/relational-databases/sql-server-transaction-locking-and-row-versioning-guide#deadlocking) | Database Admins/Devs    | Hard       | Concurrency, Error Handling             |
| 95  | What is NoSQL? How does it differ from SQL?      | [MongoDB](https://www.mongodb.com/nosql-explained)                           | Modern Data Roles       | Medium     | Database Paradigms                      |
| 96  | Explain eventual consistency.                    | [Wikipedia](https://en.wikipedia.org/wiki/Eventual_consistency)              | Distributed Systems     | Hard       | Distributed Databases, Consistency Models |
| 97  | How to design a schema for a specific scenario (e.g., social media)? | [Design Principles]                                                            | System Design Interviews| Hard       | Database Design, Modeling               |
| 98  | What are spatial data types and functions?       | [PostGIS](https://postgis.net/docs/manual-3.1/reference.html)                | GIS Applications        | Hard       | Spatial Data, GIS                       |
| 99  | How to perform fuzzy string matching in SQL?     | [Stack Overflow](https://stackoverflow.com/questions/6699158/fuzzy-string-search-in-sql) | Data Matching Roles     | Hard       | String Matching, Extensions             |
| 100 | What is Change Data Capture (CDC)?               | [Wikipedia](https://en.wikipedia.org/wiki/Change_data_capture)               | Data Integration/Sync   | Hard       | Data Replication, Event Streaming       |
| 101 | Explain Graph Databases and their use cases.     | [Neo4j](https://neo4j.com/developer/graph-database/)                           | Specialized Roles       | Hard       | Graph Databases, Data Modeling          |

---

## Code Examples

### 1. Nth Highest Salary using DENSE_RANK()

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
    Finding the Nth highest salary is a classic problem. `DENSE_RANK()` is preferred over `ROW_NUMBER()` or `RANK()` because it handles ties without skipping ranks.

    ```sql
    WITH RankedSalaries AS (
        SELECT 
            salary,
            DENSE_RANK() OVER (ORDER BY salary DESC) as rank_num
        FROM employees
    )
    SELECT DISTINCT salary
    FROM RankedSalaries
    WHERE rank_num = :N;
    ```

### 2. Recursive CTE: Employee Hierarchy

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
    Finding all subordinates of a manager (or traversing a graph/tree structure).

    ```sql
    WITH RECURSIVE Hierarchy AS (
        -- Anchor member: Start with the top-level manager
        SELECT employee_id, name, manager_id, 1 as level
        FROM employees
        WHERE manager_id IS NULL

        UNION ALL

        -- Recursive member: Join with the previous level
        SELECT e.employee_id, e.name, e.manager_id, h.level + 1
        FROM employees e
        INNER JOIN Hierarchy h ON e.manager_id = h.employee_id
    )
    SELECT * FROM Hierarchy;
    ```

### 3. Running Total and Moving Average

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
    Using Window Functions for time-series analysis.

    ```sql
    SELECT 
        date,
        sales,
        SUM(sales) OVER (ORDER BY date) as running_total,
        AVG(sales) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as 7_day_moving_avg
    FROM daily_sales;
    ```

### 4. Pivot Data (`CASE WHEN` Aggregation)

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
    Transforming rows into columns (e.g., monthly sales side-by-side).

    ```sql
    SELECT 
        product_id,
        SUM(CASE WHEN month = 'Jan' THEN sales ELSE 0 END) as Jan_Sales,
        SUM(CASE WHEN month = 'Feb' THEN sales ELSE 0 END) as Feb_Sales,
        SUM(CASE WHEN month = 'Mar' THEN sales ELSE 0 END) as Mar_Sales
    FROM monthly_sales
    GROUP BY product_id;
    ```

### 5. Gap Analysis (Identifying Missing Sequences)

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
    Finding gaps in sequential data (e.g., missing ID numbers).

    ```sql
    WITH LaggedData AS (
        SELECT 
            id, 
            LEAD(id) OVER (ORDER BY id) as next_id
        FROM sequences
    )
    SELECT 
        id + 1 as gap_start, 
        next_id - 1 as gap_end
    FROM LaggedData
    WHERE next_id - id > 1;
    ```

---

## Additional Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/) - The gold standard for SQL reference.
- [LeetCode Database Problems](https://leetcode.com/problemset/database/) - Best for practice.
- [Mode Analytics SQL Tutorial](https://mode.com/sql-tutorial/) - Excellent for data analysis focus.
- [Use The Index, Luke!](https://use-the-index-luke.com/) - Deep dive into SQL performance and indexing.
- [Modern SQL](https://modern-sql.com/) - Features of newer SQL standards.

---

## Questions asked in Google interviews

1. Explain window functions and their applications in analytical queries.
2. Write a query to find users who have logged in on consecutive days.
3. How would you optimize a slow-performing query that involves multiple joins?
4. Explain the difference between `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()`.
5. Write a query to calculate a running total or moving average.
6. How would you handle hierarchical data in SQL?
7. Explain Common Table Expressions (CTEs) and their benefits.
8. What are the performance implications of using subqueries vs. joins?
9. How would you design a database schema for a specific application?
10. Explain how indexes work and when they should be used.

## Questions asked in Amazon interviews

1. Write a query to find the nth highest salary in a table.
2. How would you identify and remove duplicate records?
3. Explain the difference between `UNION` and `UNION ALL`.
4. Write a query to pivot data from rows to columns.
5. How would you handle time-series data in SQL?
6. Explain the concept of database sharding.
7. Write a query to find users who purchased products in consecutive months.
8. How would you implement a recommendation system using SQL?
9. Explain how you would optimize a query for large datasets.
10. Write a query to calculate year-over-year growth.

## Questions asked in Microsoft interviews

1. Explain database normalization and denormalization.
2. How would you implement error handling in SQL?
3. Write a query to find departments with above-average salaries.
4. Explain the different types of joins and their use cases.
5. How would you handle slowly changing dimensions?
6. Write a query to implement a pagination system.
7. Explain transaction isolation levels.
8. How would you design a database for high availability?
9. Write a query to find the most frequent values in a column.
10. Explain the differences between clustered and non-clustered indexes.

## Questions asked in Meta interviews

1. Write a query to analyze user engagement metrics.
2. How would you implement a friend recommendation algorithm?
3. Explain how you would handle large-scale data processing.
4. Write a query to identify trending content.
5. How would you design a database schema for a social media platform?
6. Explain the concept of data partitioning.
7. Write a query to calculate the conversion rate between different user actions.
8. How would you implement A/B testing analysis using SQL?
9. Explain how you would handle real-time analytics.
10. Write a query to identify anomalies in user behavior.

## Questions asked in Netflix interviews

1. Write a query to analyze streaming patterns and user retention.
2. How would you implement a content recommendation system?
3. Explain how you would handle data for personalized user experiences.
4. Write a query to identify viewing trends across different demographics.
5. How would you design a database for content metadata?
6. Explain how you would optimize queries for real-time recommendations.
7. Write a query to calculate user engagement metrics.
8. How would you implement A/B testing for UI changes?
9. Explain how you would handle data for regional content preferences.
10. Write a query to identify factors affecting user churn.

## Questions asked in Apple interviews

1. Explain database security best practices.
2. How would you design a database for an e-commerce platform?
3. Write a query to analyze product performance.
4. Explain how you would handle data migration.
5. How would you implement data validation in SQL?
6. Write a query to track user interactions with products.
7. Explain how you would optimize database performance.
8. How would you implement data archiving strategies?
9. Write a query to analyze customer feedback data.
10. Explain how you would handle internationalization in databases.

## Questions asked in LinkedIn interviews

1. Write a query to implement a connection recommendation system.
2. How would you design a database schema for professional profiles?
3. Explain how you would handle data for skill endorsements.
4. Write a query to analyze user networking patterns.
5. How would you implement job recommendation algorithms?
6. Explain how you would handle data for company pages.
7. Write a query to identify trending job skills.
8. How would you implement search functionality for profiles?
9. Explain how you would handle data privacy requirements.
