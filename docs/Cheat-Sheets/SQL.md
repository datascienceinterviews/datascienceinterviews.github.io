
---
title: SQL Cheat Sheet
description: A comprehensive reference guide for SQL, covering data types, DDL, DML, DQL, TCL, joins, subqueries, window functions, and more.
---

# SQL Cheat Sheet

[TOC]

This cheat sheet provides an exhaustive overview of SQL (Structured Query Language), covering data types, Data Definition Language (DDL), Data Manipulation Language (DML), Data Query Language (DQL), Transaction Control Language (TCL), joins, subqueries, window functions, common table expressions (CTEs), and best practices. It aims to be a complete reference for writing and understanding SQL queries.  This cheat sheet is designed to be generally applicable across different SQL database systems (e.g., MySQL, PostgreSQL, SQL Server, Oracle, SQLite), but notes specific differences where significant.

## Data Types

### Numeric

*   `INT`, `INTEGER`: Integer values.
*   `SMALLINT`: Smaller integer values.
*   `BIGINT`: Larger integer values.
*   `TINYINT`: Very small integer values (MySQL, SQL Server).
*   `REAL`: Single-precision floating-point numbers.
*   `FLOAT(p)`: Floating-point number with precision `p`.
*   `DOUBLE PRECISION`: Double-precision floating-point numbers.
*   `DECIMAL(p, s)`, `NUMERIC(p, s)`: Fixed-point numbers with precision `p` and scale `s`.

### String

*   `CHAR(n)`: Fixed-length character string of length `n`.
*   `VARCHAR(n)`: Variable-length character string with a maximum length of `n`.
*   `TEXT`: Variable-length character string with no specified maximum length (or a very large maximum).
*   `NCHAR(n)`, `NVARCHAR(n)`: Unicode character strings (for storing characters from different languages).

### Date and Time

*   `DATE`: Date (YYYY-MM-DD).
*   `TIME`: Time (HH:MI:SS).
*   `DATETIME`, `TIMESTAMP`: Date and time.
*   `INTERVAL`: A period of time.

### Boolean

*   `BOOLEAN`: True or False.  (Some databases, like MySQL, use `TINYINT(1)` to represent booleans).

### Other

*   `BLOB`: Binary large object (for storing binary data).
*   `CLOB`: Character large object (for storing large text data).
*   `JSON`, `JSONB`: JSON data (supported by some databases like PostgreSQL).
*   `UUID`: Universally Unique Identifier (supported by some databases like PostgreSQL).
*   `ENUM`: Enumerated type (MySQL, PostgreSQL).
*   `ARRAY`: Array type (PostgreSQL).

## Data Definition Language (DDL)

### CREATE TABLE

```sql
CREATE TABLE table_name (
    column1 datatype constraints,
    column2 datatype constraints,
    ...
    PRIMARY KEY (column1),
    FOREIGN KEY (column_fk) REFERENCES other_table(other_column)
);

-- Example
CREATE TABLE employees (
    id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100) UNIQUE,
    hire_date DATE,
    salary DECIMAL(10, 2),
    department_id INT,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);
```

### ALTER TABLE

```sql
-- Add a column
ALTER TABLE table_name ADD COLUMN column_name datatype;

-- Drop a column
ALTER TABLE table_name DROP COLUMN column_name;

-- Modify a column
ALTER TABLE table_name MODIFY COLUMN column_name new_datatype;  -- MySQL, SQL Server
ALTER TABLE table_name ALTER COLUMN column_name TYPE new_datatype; -- PostgreSQL

-- Add a constraint
ALTER TABLE table_name ADD CONSTRAINT constraint_name constraint_definition;

-- Drop a constraint
ALTER TABLE table_name DROP CONSTRAINT constraint_name; -- Most databases
ALTER TABLE table_name DROP INDEX constraint_name; -- MySQL (for UNIQUE constraints)
```

### DROP TABLE

```sql
DROP TABLE table_name;

-- Drop table only if it exists (avoids error if it doesn't)
DROP TABLE IF EXISTS table_name;
```

### TRUNCATE TABLE

```sql
TRUNCATE TABLE table_name;  -- Removes all rows, faster than DELETE
```

### CREATE INDEX

```sql
CREATE INDEX index_name ON table_name (column1, column2, ...);

-- Unique index
CREATE UNIQUE INDEX index_name ON table_name (column1);
```

### DROP INDEX

```sql
DROP INDEX index_name ON table_name; -- Most databases
ALTER TABLE table_name DROP INDEX index_name; -- MySQL
```

### CREATE VIEW

```sql
CREATE VIEW view_name AS
SELECT column1, column2, ...
FROM table_name
WHERE condition;

-- Example
CREATE VIEW employee_names AS
SELECT first_name, last_name
FROM employees;
```

### DROP VIEW

```sql
DROP VIEW view_name;
```

### DATABASE Operations

```sql
-- Create a new database
CREATE DATABASE database_name;

-- Delete an existing database
DROP DATABASE database_name;

-- Delete an existing database only if it exists (avoids error if it doesn't)
DROP DATABASE IF EXISTS database_name;

-- Select a database to use (syntax varies, common in MySQL)
USE database_name;
```



## Data Manipulation Language (DML)

### INSERT

```sql
INSERT INTO table_name (column1, column2, ...) VALUES (value1, value2, ...);

-- Insert multiple rows
INSERT INTO table_name (column1, column2) VALUES
(value1a, value2a),
(value1b, value2b),
(value1c, value2c);

-- Insert from another table
INSERT INTO table_name (column1, column2)
SELECT column1, column2
FROM other_table
WHERE condition;
```

### UPDATE

```sql
UPDATE table_name
SET column1 = value1, column2 = value2, ...
WHERE condition;

-- Example
UPDATE employees
SET salary = salary * 1.10
WHERE department_id = 1;
```

### DELETE

```sql
DELETE FROM table_name WHERE condition;

-- Example
DELETE FROM employees WHERE id = 123;

-- Delete all rows (slower than TRUNCATE TABLE)
DELETE FROM table_name;
```

## Data Query Language (DQL)

### SELECT

```sql
SELECT column1, column2, ...
FROM table_name
WHERE condition
ORDER BY column1 ASC, column2 DESC
LIMIT n OFFSET m;

-- Select all columns
SELECT * FROM table_name;

-- Select with aliases
SELECT column1 AS alias1, column2 AS alias2 FROM table_name;

-- Select distinct values
SELECT DISTINCT column1 FROM table_name;
```

### WHERE Clause

```sql
SELECT * FROM table_name WHERE column1 = value1 AND column2 > value2;
SELECT * FROM table_name WHERE column1 IN (value1, value2, value3);
SELECT * FROM table_name WHERE column1 BETWEEN value1 AND value2;
SELECT * FROM table_name WHERE column1 LIKE 'pattern%'; -- % is a wildcard
SELECT * FROM table_name WHERE column1 IS NULL;
SELECT * FROM table_name WHERE column1 IS NOT NULL;
```

### ORDER BY Clause

```sql
SELECT * FROM table_name ORDER BY column1 ASC, column2 DESC;
```

### LIMIT and OFFSET Clauses

```sql
SELECT * FROM table_name LIMIT 10;  -- Get the first 10 rows
SELECT * FROM table_name LIMIT 10 OFFSET 5;  -- Get 10 rows starting from row 6
```

### Aggregate Functions

*   `COUNT()`: Counts rows.
*   `SUM()`: Sums values.
*   `AVG()`: Calculates the average.
*   `MIN()`: Finds the minimum value.
*   `MAX()`: Finds the maximum value.

```sql
SELECT COUNT(*) FROM table_name;
SELECT SUM(salary) FROM employees;
SELECT AVG(age) FROM employees;
SELECT MIN(hire_date) FROM employees;
SELECT MAX(salary) FROM employees;
```

### GROUP BY Clause

```sql
SELECT department_id, AVG(salary) AS average_salary
FROM employees
GROUP BY department_id;
```

### HAVING Clause

```sql
SELECT department_id, AVG(salary) AS average_salary
FROM employees
GROUP BY department_id
HAVING AVG(salary) > 50000;
```

## Joins

### INNER JOIN

```sql
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.id;
```

### LEFT JOIN (LEFT OUTER JOIN)

```sql
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.id;
```

### RIGHT JOIN (RIGHT OUTER JOIN)

```sql
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.id;
```

### FULL JOIN (FULL OUTER JOIN)

```sql
-- Full outer join is not supported by all databases (e.g., MySQL).
-- Use a combination of LEFT JOIN and RIGHT JOIN with UNION for equivalent functionality.
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
FULL OUTER JOIN departments d ON e.department_id = d.id;

-- Equivalent in MySQL:
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.id
UNION
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.id;
```

### Self Join

```sql
SELECT e1.first_name, e2.first_name AS manager_name
FROM employees e1
JOIN employees e2 ON e1.manager_id = e2.id;
```

### Cross Join

```sql
SELECT *
FROM table1
CROSS JOIN table2;
```

## Subqueries

```sql
-- Subquery in WHERE clause
SELECT *
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Subquery in SELECT clause
SELECT first_name, last_name,
       (SELECT COUNT(*) FROM orders WHERE orders.employee_id = employees.id) AS order_count
FROM employees;

-- Subquery in FROM clause
SELECT *
FROM (SELECT first_name, last_name, salary FROM employees) AS employee_salaries
WHERE salary > 60000;

-- Correlated subquery
SELECT e.first_name, e.last_name
FROM employees e
WHERE e.salary > (SELECT AVG(salary) FROM employees WHERE department_id = e.department_id);

-- EXISTS and NOT EXISTS
SELECT *
FROM employees e
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.employee_id = e.id);
```

## Common Table Expressions (CTEs)

```sql
WITH employee_summary AS (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
)
SELECT d.department_name, es.avg_salary
FROM departments d
JOIN employee_summary es ON d.id = es.department_id;
```

## Window Functions

```sql
SELECT
    first_name,
    last_name,
    salary,
    AVG(salary) OVER (PARTITION BY department_id) AS avg_salary_by_department,
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS salary_rank
FROM employees;
```

Common Window Functions:

*   `ROW_NUMBER()`: Assigns a unique sequential integer to each row within its partition.
*   `RANK()`: Assigns a rank to each row within its partition, with gaps in rank values.
*   `DENSE_RANK()`: Assigns a rank to each row within its partition, without gaps.
*   `NTILE(n)`: Divides the rows within a partition into `n` groups.
*   `LAG(column, offset, default)`: Accesses data from a previous row.
*   `LEAD(column, offset, default)`: Accesses data from a subsequent row.
*   `FIRST_VALUE(column)`: Returns the first value in a window frame.
*   `LAST_VALUE(column)`: Returns the last value in a window frame.
*   `NTH_VALUE(column, n)`: Returns the nth value in a window frame.

## Transaction Control Language (TCL)

### START TRANSACTION (or BEGIN)

```sql
START TRANSACTION;
-- or
BEGIN;
```

### COMMIT

```sql
COMMIT;  -- Save changes
```

### ROLLBACK

```sql
ROLLBACK;  -- Discard changes
```

### SAVEPOINT

```sql
SAVEPOINT savepoint_name;
```

### ROLLBACK TO SAVEPOINT

```sql
ROLLBACK TO SAVEPOINT savepoint_name;
```

### SET TRANSACTION

```sql
SET TRANSACTION ISOLATION LEVEL READ COMMITTED; -- Example
```

## String Functions

*   `CONCAT(str1, str2, ...)`: Concatenates strings.
*   `LENGTH(str)` or `LEN(str)`: Returns the length of a string.
*   `SUBSTRING(str, start, length)` or `SUBSTR(str, start, length)`: Extracts a substring.
*   `UPPER(str)` or `UCASE(str)`: Converts a string to uppercase.
*   `LOWER(str)` or `LCASE(str)`: Converts a string to lowercase.
*   `TRIM(str)`: Removes leading and trailing whitespace.
*   `LTRIM(str)`: Removes leading whitespace.
*   `RTRIM(str)`: Removes trailing whitespace.
*   `REPLACE(str, old, new)`: Replaces occurrences of a substring.
*   `INSTR(str, substr)` or `POSITION(substr IN str)`: Returns the position of a substring.
*   `LEFT(str, length)`: Returns the leftmost characters of a string.
*   `RIGHT(str, length)`: Returns the rightmost characters of a string.
*   `LPAD(str, length, padstr)`: Left-pads a string.
*   `RPAD(str, length, padstr)`: Right-pads a string.

## Date and Time Functions

*   `NOW()`, `CURRENT_TIMESTAMP`: Returns the current date and time.
*   `CURDATE()`, `CURRENT_DATE`: Returns the current date.
*   `CURTIME()`, `CURRENT_TIME`: Returns the current time.
*   `DATE(expression)`: Extracts the date part of a date or datetime expression.
*   `TIME(expression)`: Extracts the time part of a time or datetime expression.
*   `YEAR(date)`, `MONTH(date)`, `DAY(date)`: Extracts the year, month, or day.
*   `HOUR(time)`, `MINUTE(time)`, `SECOND(time)`: Extracts the hour, minute, or second.
*   `DATE_ADD(date, INTERVAL expr unit)`, `DATE_SUB(date, INTERVAL expr unit)`: Adds or subtracts a time interval.
*   `DATEDIFF(date1, date2)`: Returns the difference between two dates (in days).
*   `TIMESTAMPDIFF(unit, datetime1, datetime2)`: Returns the difference between two datetimes in a specified unit.
*   `DATE_FORMAT(date, format)`: Formats a date.

## Conditional Expressions

### CASE

```sql
SELECT
    first_name,
    last_name,
    CASE
        WHEN salary > 80000 THEN 'High'
        WHEN salary > 50000 THEN 'Medium'
        ELSE 'Low'
    END AS salary_level
FROM employees;
```

### IF (MySQL, SQL Server)

```sql
SELECT first_name, last_name, IF(salary > 50000, 'High', 'Low') AS salary_level
FROM employees;
```

### COALESCE

```sql
SELECT COALESCE(column1, column2, 'Default Value') AS result FROM table_name;
```

### NULLIF

```sql
SELECT NULLIF(column1, value) AS result FROM table_name;
```

## User-Defined Functions (UDFs)

(Syntax varies significantly between database systems)

Example (MySQL):

```sql
DELIMITER //
CREATE FUNCTION my_function(param1 INT, param2 VARCHAR(255))
RETURNS INT
DETERMINISTIC
BEGIN
    -- Function logic
    RETURN result;
END //
DELIMITER ;
```

## Stored Procedures

(Syntax varies significantly between database systems)

Example (MySQL):

```sql
DELIMITER //
CREATE PROCEDURE my_procedure(IN param1 INT, OUT param2 VARCHAR(255))
BEGIN
    -- Procedure logic
    SELECT column1 INTO param2 FROM table_name WHERE column2 = param1;
END //
DELIMITER ;
```

## Triggers

(Syntax varies significantly between database systems)

Example (MySQL):

```sql
DELIMITER //
CREATE TRIGGER my_trigger
BEFORE INSERT ON employees
FOR EACH ROW
BEGIN
    -- Trigger logic
    SET NEW.created_at = NOW();
END //
DELIMITER ;
```

## Indexes

### Creating Indexes
```sql
CREATE INDEX idx_lastname ON employees (last_name);
CREATE UNIQUE INDEX idx_email ON employees (email);
CREATE INDEX idx_lastname_firstname ON employees (last_name, first_name);
```

### Dropping Indexes
```sql
DROP INDEX idx_lastname ON employees; -- Standard SQL
ALTER TABLE employees DROP INDEX idx_lastname; -- MySQL
```

## Views

### Creating Views
```sql
CREATE VIEW high_salary_employees AS
SELECT employee_id, first_name, last_name, salary
FROM employees
WHERE salary > 80000;
```

### Dropping Views
```sql
DROP VIEW high_salary_employees;
```

## Transactions

```sql
START TRANSACTION; -- or BEGIN;

-- SQL statements

COMMIT; -- Save changes
-- or
ROLLBACK; -- Discard changes
```

## Security

*   **User Management:** `CREATE USER`, `ALTER USER`, `DROP USER`, `GRANT`, `REVOKE`.
*   **Permissions:** Grant specific privileges (e.g., `SELECT`, `INSERT`, `UPDATE`, `DELETE`) to users or roles on database objects.
*   **Roles:** Create roles to group privileges and assign them to users.
*   **Views:** Use views to restrict access to sensitive data.
*   **Stored Procedures:** Use stored procedures to encapsulate logic and control access.
*   **Encryption:** Encrypt sensitive data at rest and in transit.
*   **Auditing:** Enable auditing to track database activity.
*   **SQL Injection Prevention:** Use parameterized queries or prepared statements to prevent SQL injection attacks.

## Best Practices

*   **Use meaningful names:** Choose descriptive names for tables, columns, and other database objects.
*   **Normalize your database:** Design your database schema to reduce data redundancy and improve data integrity.
*   **Use appropriate data types:** Select data types that are appropriate for the data you are storing.
*   **Use indexes:** Create indexes on columns that are frequently used in `WHERE` clauses and `JOIN` conditions.
*   **Optimize your queries:** Write efficient queries that minimize the amount of data that needs to be processed.
*   **Use transactions:** Use transactions to ensure data consistency and integrity.
*   **Back up your database:** Regularly back up your database to prevent data loss.
*   **Secure your database:** Implement appropriate security measures to protect your data.
*   **Use comments:** Add comments to your SQL code to explain what it does.
*   **Use a consistent coding style:** Follow a consistent coding style to make your code easier to read and maintain.
*   **Test your queries:** Thoroughly test your queries to ensure they are working as expected.
*   **Use a database management tool:** Use a tool like MySQL Workbench, pgAdmin, SQL Server Management Studio, or Dbeaver to manage your database.
*   **Use version control:** Use a version control system (e.g., Git) to track changes to your database schema and code.
*   **Use an ORM (Object-Relational Mapper):** Consider using an ORM (e.g., SQLAlchemy, Django ORM) to simplify database interactions.
*   **Avoid `SELECT *`:** Explicitly list the columns you need to retrieve.
*   **Use `EXISTS` instead of `COUNT(*)` when checking for existence:** `EXISTS` is often more efficient.
*   **Use `JOIN` instead of subqueries when possible:** Joins are generally faster.
*   **Use `UNION ALL` instead of `UNION` when you don't need to remove duplicates:** `UNION ALL` is faster.
*   **Use `CASE` expressions for conditional logic:** `CASE` expressions are more flexible than `IF`.
*   **Use CTEs to improve readability:** CTEs can make complex queries easier to understand.
*   **Use window functions for advanced analytics:** Window functions allow you to perform calculations across rows.
*   **Use stored procedures and functions to encapsulate logic:** This can improve code reusability and maintainability.
*   **Use triggers to automate tasks:** Triggers can be used to automatically perform actions when certain events occur.
*   **Use views to simplify complex queries:** Views can make it easier to access data from multiple tables.
*   **Use indexes to improve query performance:** Indexes can significantly speed up queries that filter or sort data.
*   **Use explain plans to analyze query performance:** Explain plans show you how the database is executing your queries.
*   **Use a database profiler to identify performance bottlenecks:** Profilers can help you find slow queries and other performance issues.
*   **Use a database monitoring tool to track database performance:** Monitoring tools can help you identify and resolve performance problems.
*   **Regularly update your database software:** Updates often include performance improvements and security fixes.
*   **Follow database best practices:** Each database system has its own set of best practices.