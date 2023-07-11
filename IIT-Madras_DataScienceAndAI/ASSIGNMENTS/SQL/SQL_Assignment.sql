-- create a table
CREATE TABLE EmployeeTable (
  Employee_id INTEGER PRIMARY KEY,
  First_name VARCHAR(50) NOT NULL,
  Last_name VARCHAR(50) NOT NULL,
  Salary DECIMAL(10),
  Joining_date DATETIME,
  Department VARCHAR(50)
);

-- insert some values
INSERT INTO EmployeeTable (Employee_id, First_name, Last_name, Salary, Joining_date, Department)
VALUES 
(1, 'Anika', 'Arora', 100000, '2020-02-14 9:00:00', 'HR'),
(2, 'Veena', 'Verma', 80000, '2011-06-15 9:00:00', 'Admin'),
(3, 'Vishal', 'Singhal', 300000, '2020-02-16 9:00:00', 'HR'),
(4, 'Sushant', 'Singh', 500000, '2020-02-17 9:00:00', 'Admin'),
(5, 'Bhupal', 'Bhati', 500000, '2011-06-18 9:00:00', 'Admin'),
(6, 'Dheeraj', 'Diwan', 200000, '2011-06-19 9:00:00', 'Account'),
(7, 'Karan', 'Kumar', 75000, '2020-01-14 9:00:00', 'Account'),
(8, 'Chandrika', 'Chauhan', 90000, '2011-04-15 9:00:00', 'Admin');

-- Create 2nd table
CREATE TABLE EmployeeBonusTable (
  Employee_ref_id INTEGER,
  Bonus_Amount DECIMAL(10),
  Bonus_Date DATETIME,
  FOREIGN KEY (Employee_ref_id) REFERENCES EmployeeTable(Employee_id)
);

-- Insert values
INSERT INTO EmployeeBonusTable (Employee_ref_id, Bonus_Amount, Bonus_Date)
VALUES 
(1, 5000, '2020-02-16 0:00:00'),
(2, 3000, '2011-06-16 0:00:00'),
(3, 4000, '2020-02-16 0:00:00'),
(1, 4500, '2020-02-16 0:00:00'),
(2, 3500, '2011-06-16 0:00:00');

-- Create 3rd table
CREATE TABLE EmployeeTitleTable (
  Employee_ref_id VARCHAR(50),
  Employee_title VARCHAR(50),
  Affective_Date DATETIME);
  
-- Insert values
INSERT INTO EmployeeTitleTable (Employee_ref_id, Employee_title, Affective_Date)
VALUES
(1, 'Manager', '2016-02-20 0:00:00'),
(2, 'Executive', '2016-06-11 0:00:00'),
(8, 'Executive', '2016-06-11 0:00:00'),
(5, 'Manager', '2016-06-11 0:00:00'),
(4, 'Asst. Manager', '2016-06-11 0:00:00'),
(7, 'Executive', '2016-06-11 0:00:00'),
(6, 'Lead', '2016-06-11 0:00:00'),
(3, 'Lead', '2016-06-11 0:00:00');

-- 1 Display the “FIRST_NAME” from Employee table using the alias name as Employee_name.

SELECT First_name AS Employee_name 
FROM EmployeeTable;


-- 2 Display “LAST_NAME” from Employee table in upper case.

SELECT UPPER(Last_name) 
FROM EmployeeTable;


-- 3 Display unique values of DEPARTMENT from EMPLOYEE table.

SELECT DISTINCT Department 
FROM EmployeeTable;


-- 4 Display the first three characters of LAST_NAME from EMPLOYEE table.

SELECT SUBSTRING(Last_name, 1, 3) FROM EmployeeTable;
-- or
SELECT LEFT(Last_name, 3)
FROM EmployeeTable;

-- 5 Display the unique values of DEPARTMENT from EMPLOYEE table and prints its length.

SELECT DISTINCT Department, LEN(Department) AS DepartmentLength
FROM EmployeeTable;


-- 6 Display the FIRST_NAME and LAST_NAME from EMPLOYEE table into a single column AS FULL_NAME.
-- a space char should separate them.

SELECT CONCAT(First_name, ' ',Last_name) AS FULL_NAME 
FROM EmployeeTable;


-- 7 DISPLAY all EMPLOYEE details from the employee table order by FIRST_NAME Ascending.

SELECT * 
FROM EmployeeTable 
ORDER BY First_name ASC;


-- 8. Display all EMPLOYEE details order by FIRST_NAME Ascending and DEPARTMENT Descending.

SELECT * 
FROM EmployeeTable 
ORDER BY First_name ASC, Department DESC;


-- 9 Display details for EMPLOYEE with the first name as “VEENA” and “KARAN” from EMPLOYEE table.

SELECT * 
FROM EmployeeTable 
WHERE First_name IN ('VEENA', 'KARAN');


-- 10 Display details of EMPLOYEE with DEPARTMENT name as “Admin”.

SELECT * 
FROM EmployeeTable 
WHERE Department = 'Admin';


-- 11 DISPLAY details of the EMPLOYEES whose FIRST_NAME contains ‘V’.

SELECT * 
FROM EmployeeTable 
WHERE First_name LIKE '%V%';


-- 12 DISPLAY details of the EMPLOYEES whose SALARY lies between 100000 and 500000.

SELECT * 
FROM EmployeeTable 
WHERE Salary BETWEEN 100000 AND 500000;


-- 13 Display details of the employees who have joined in Feb-2020.

SELECT *
FROM EmployeeTable
WHERE Joining_date BETWEEN '2020-02-01' AND '2020-02-28';


-- 14 Display employee names with salaries >= 50000 and <= 100000.

SELECT First_name, Last_name, Salary
FROM EmployeeTable
WHERE Salary >= 50000 AND Salary <= 100000;


-- 16 DISPLAY details of the EMPLOYEES who are also Managers.

SELECT *
FROM EmployeeTable
WHERE Employee_id IN (
  SELECT Employee_ref_id
  FROM EmployeeTitleTable
  WHERE Employee_title = 'Manager'
);

-- 17 DISPLAY duplicate records having matching data in some fields of a table.

SELECT Employee_id, First_name, Last_name, Salary, Joining_date, Department 
FROM EmployeeTable 
GROUP BY Employee_id, First_name, Last_name, Salary, Joining_date, Department
HAVING COUNT(*) > 1;


-- 18 Display only odd rows from a table.

SELECT *
FROM EmployeeTable
WHERE Employee_id % 2 = 1;

-- 19 Clone a new table from EMPLOYEE table.

SELECT *
INTO NewEmployeeTable
FROM EmployeeTable;


-- 20 DISPLAY the TOP 2 highest salary from a table.

SELECT TOP 2 *
FROM EmployeeTable
ORDER BY Salary DESC;


-- 21. DISPLAY the list of employees with the same salary.

SELECT Salary, COUNT(*) AS EmployeeCount
FROM EmployeeTable
GROUP BY Salary
HAVING COUNT(*) > 1;


-- 22 Display the second highest salary from a table.

SELECT MAX(Salary) AS SecondHighestSalary
FROM EmployeeTable
WHERE Salary < (SELECT MAX(Salary) FROM EmployeeTable);


-- 23 Display the first 50% records from a table.

SELECT *
FROM (
  SELECT *, ROW_NUMBER() OVER (ORDER BY Employee_id) AS row_num
  FROM EmployeeTable
) AS subquery
WHERE row_num <= (SELECT COUNT(*) FROM EmployeeTable) / 2;

-- or

SELECT TOP 50 PERCENT *
FROM EmployeeTable;

-- 24. Display the departments that have less than 4 people in it.

SELECT Department
FROM EmployeeTable
GROUP BY Department
HAVING COUNT(*) < 4;


-- 25. Display all departments along with the number of people in there.

SELECT Department, COUNT(*) AS Count
FROM EmployeeTable
GROUP BY Department;


-- 26 Display the name of employees having the highest salary in each department.

SELECT Department, First_name, Last_name, Salary
FROM EmployeeTable e
WHERE Salary = (
  SELECT MAX(Salary)
  FROM EmployeeTable
  WHERE Department = e.Department
);


-- 27 Display the names of employees who earn the highest salary.

SELECT First_name, Last_name
FROM EmployeeTable
WHERE Salary = (
  SELECT MAX(Salary)
  FROM EmployeeTable
);


-- 28 Display the average salaries for each department

SELECT Department, AVG(Salary) AS AverageSalary
FROM EmployeeTable
GROUP BY Department;


-- 29 display the name of the employee who has got maximum bonus

SELECT e.First_name, e.Last_name
FROM EmployeeTable e
INNER JOIN EmployeeBonusTable b ON e.Employee_id = b.Employee_ref_id
WHERE b.Bonus_Amount = (
  SELECT MAX(Bonus_Amount)
  FROM EmployeeBonusTable
);


-- 30 Display the first name and title of all the employees

SELECT First_name, Employee_title
FROM EmployeeTable e
INNER JOIN EmployeeTitleTable t ON e.Employee_id = t.Employee_ref_id;


