Dear NSFG Data User, 

We would like to bring to your attention that when importing a CSV data file into SAS or R, the 
software by default determines the variable attributes (i.e., data type and length) based on the 
first few observations, which may be an insufficient number of observations if these include few 
or no responses. Variables with a length of two or more (e.g., R's age at interview AGE_R with values 15-49) 
and with many missing values can get truncated to length of one (e.g., values 10–12 will be read 
as 1) or be given an incorrect data type. In NSFG a variable may have many missing values in cases where most of the 
respondents are out of universe for the question. 

------

Data users can set the number of observations that the software uses for determining the length of 
variables in the CSV dataset. 

For SAS, at least half the number of observations in the data file 
may be reasonable and using all the observations in the file may be necessary if variables are not
read correctly. 

For R, specifying how to read the variable attribute depends on whether you use 
standard R or the readr R library or an R library that includes readr such as tidyverse. To confirm 
that your software correctly created your data file from the CSV file, we advise that you compare 
the frequencies from your file to the frequencies provided in the codebooks on the website (and 
focus on those variables with few observations). 

------

Options to change the default:

SAS
In SAS, add the option ‘guessingrows’ to the import statement. The 2022-2023 NSFG Male Respondent File has 4371
observations and the example below specifies to use 2,000 observations to determine the length:

proc import datafile="NSFG_2022_2023_MaleRespPUFData.csv" out=female dbms=csv; getnames=yes; guessingrows=MAX; run;

------

Standard R
The read.csv function will read the entire file and wait until the end to determine variable attributes. 
The default that converts text strings into enumerated values can be disabled. The example below reads 
the entire 2022-2023 Male Respondent CSV file and does not convert text strings into factors but leaves them as 
text strings. 

2022_2023_MaleRespData <- read.csv("NSFG_2022_2023_MaleRespPUFData.csv", stringsAsFactors=FALSE)

------

R Library(readr)
The read_csv function by default only reads in 1,000 records to determine variable attributes and will not 
convert strings to factors. If using read_csv, add the argument ‘guess_max.’ The 2022-2023 NSFG Male Respondent file has 
4371 observations and the example below specifies to use 4371 observations to determine the data type.

2022_2023_MaleRespData <- read_csv("NSFG_2022_2023_MaleRespPUFData.csv", guess_max = Inf)
