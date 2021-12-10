# Personal data projects

Use this template repo for your personal data projects. Would you please use the following format for your repository? Please name your repository in your personal space using the following naming structure `[title]-[lastname]-[languagesused]`.  You will then complete a pull request to move your personal data projects into our organization.

- __Project Purpose:__ Take one to two paragraphs to explain your goals.  Why did you use this data? What skills are you going to demonstrate? Did you do this work for a client? 
- __Tools used:__ Provide an overview of the programming languages and packages you leveraged for this work.  You could provide links to the pertinent reading material that you leveraged to complete the job. Provide links to your final scripts stored in your repository.
- __Results:__ This is your conclusion.  Explain why your work matters.  How could others use it?  What are your next steps? Show some key findings.

## Folder structure

```
- readme.md
- scripts
---- readme.md (short description of each script)
---- preprocess.r
---- imputation_exploration
---- model.ipynb
- data
---- MERGED2018_19_PP.csv (CollegeScorecard dataset)
---- data_definitions.csv
---- data_dict.csv
---- full_df.csv
---- state_fp_codes.csv
-- created_subsets
---- (datasets created during the process will be placed here)
- documents
---- readme.md (notes)
---- Nathaniel.Jones.Classification.Project.STAT4310.FALL21.docx
---- Nathaniel.Jones.Classification.Project.STAT4310.FALL21.pdf
---- Nathaniel.Jones.Classification.Project.STAT4310.FALL21.pptx
```

## Data sources

Today, a student may be eligible for a portion of the Pell grant if their income, or their parents’ income, in the case of dependent students, is less than $50,000. The portion the student receives increases as the amount of income decreases. A maximum amount of $6495 is granted to those students whose household income is less than or equal to $20,000. For these reasons, receiving a Pell grant is associated with low-income students. I researched into the CollegeScorecard dataset and created the indicator variable, “PELL_CAT,” by using the proportion of Pell-receiving students (“PCTPELL”) to categorize institutions as either “majority Pell” (>50% of students receiving a Pell grant) or “minority Pell” (≤50% of students receiving a Pell grant). I created this indicator variable to determine where lower-income students attended more frequently and the outcomes at these institutions.

- [The CollegeScorecard dataset](https://collegescorecard.ed.gov/data/)
- [The CollegeScorecard data Dictionary](https://data.ed.gov/dataset/college-scorecard-all-data-files-through-6-2020/resources?resource=658b5b83-ac9f-4e41-913e-9ba9411d7967)

## Github pages

- ["Does the Pell grant come with a price?"](https://github.com/njones738/Does-the-Pell-grant-come-with-a-price-)
- ["Access to Higher Education"](https://github.com/njones738/Access-to-Higher-Education)


## Question

The goal of this project is to create a model that classifies institutions as either “Majority Pell” or “Minority Pell”. Explicitly, I want to answer the question, 

- “Which institutional features are associated with the majority Pell schools?”

By answering this question, we can better understand the most prevalent attributes of schools with a high proportion of students receiving a Pell grant. Gaining the understanding of these attributes will aid in detecting problems, trends, and stratifications at institutions where low-income students are in high attendance.

## Key findings

**Best Model:** {Accuracy/AUC – Training: 97.8%/99.9%, Testing: 86.2%/93.5%, Validation: 88.1%/93.4%}
The model I determined to be the best was a random forest on the standardized dataset that was reduced to the top 10 most independent variables. Figure one displays the features that were most important in classifying. The most important of these features to this model was the percentage of Federal Loan Borrowers at the institution, and the next most important was the agency that accredits the institution. With only 138 incorrect classifications in total (Figure 2), this model correctly classified:

- 537 majority Pell institutions
- 325 minority Pell institutions

The following figures display the important features to this model:

![](https://github.com/njones738/Classification-of-Pell-Institutions/blob/main/images/STAND/REDUC/RanFor/githubpic.jpg)

**Next Best Model:** {Accuracy/AUC – Training: 86.2%/93.9%, Testing: 85.5%/92.2%, Validation: 86%/92.9%}
The next best model used the method k-nearest neighbors on the standardized-transformed dataset. This model correctly classified 511 majority Pell institutions and 320 minority Pell institutions, while misclassifying a total of 169 incorrect classifications across both groups. Figure 3 displays the results of this classifier using the validation dataset. The red and blue dots are the correctly classified Majority and Minority Pell schools while the black and dark brown dots are the incorrect classifications.

![](https://github.com/njones738/Classification-of-Pell-Institutions/blob/main/images/STAND/REDUC/RanFor/githubpic2.jpg)

## Influential Articles

### [Average amount of grant, scholarship aid, and net price](https://nces.ed.gov/programs/digest/d19/tables/dt19_331.30.asp)
### [Unemployment Rate x Median Weekly Earnings by Degree](https://www.bls.gov/emp/chart-unemployment-earnings-education.htm)
 * These education categories reflect only the highest level of educational attainment. They do not take into account completion of training programs in the form of 
   apprenticeships and other on-the-job training, which may also influence earnings and unemployment rates. For more information on training, see the [link](https://www.bls.gov/emp/documentation/education-training-system.htm)

### [College Quality x Class Background](https://www.brookings.edu/blog/social-mobility-memos/2016/02/19/a-college-degree-is-worth-less-if-you-are-raised-poor/) 
 * It turns out that the proportional increase for those who grew up poor is much less than for those who did not. College graduates from families with an income below 
   185 percent of the federal poverty level (the eligibility threshold for the federal assisted lunch program) earn 91 percent more over their careers than high school graduates from the same income group. By comparison, college graduates from families with incomes above 185 percent of the FPL earned 162 percent more over their careers (between the ages of 25 and 62) than those with just a high school diploma.

### [College Quality x Race, Class Background](https://www.brookings.edu/research/the-stubborn-race-and-class-gaps-in-college-quality/)
 * The average black undergraduate is enrolled in a college with a graduation rate rank in the 40th centile of all colleges, compared to the 55th centile for whites, 
   and with a default rate that is 50 higher. Median alumni earnings six years after attendance are almost 10 percent higher at the colleges attended by the average white student.
 * First-generation borrowers—students with federal loans whose parents did not attend college—are more likely to attend colleges with moderately high earnings 
   outcomes but extremely poor graduation and loan default outcomes.

### [STEM Attrition](https://nces.ed.gov/pubs2014/2014001rev.pdf)
 * This Statistical Analysis Report (SAR) presents an examination of students’ attrition from STEM fields over the course of 6 years in college using data from the 
   2004/09 Beginning Postsecondary Students Longitudinal Study (BPS:04/09) and the associated 2009 Postsecondary Education Transcript Study (PETS:09). In this SAR, the term STEM attrition refers to enrollment choices that result in potential STEM graduates (i.e., undergraduates who declare a STEM major) moving away from STEM fields by switching majors to non-STEM fields or leaving postsecondary education before earning a degree or certificate.
 
### [Debt x Public or Private Institutions](https://web.stanford.edu/~kjytay/courses/stats32-aut2018/projects/College_Data.html)
 * Since the R^2 value of the relationship between the median earnings of a college’s graduates and the median family income of the college’s students is the highest 
   among the three variables examined (i.e., admit, med_fam_inc, and price), it is the better predictor of the potential economic outcome of an institution’s students.

### [Skills, Knowledge x Financial Capability](https://www.financialcapability.gov.au/files/research_factors-that-influence-capability-and-effectiveness.pdf)
 * We report that financial counsellors view confidence, self-esteem and self-belief as equally important determinants of financial capability. Also, gender and family 
   socio economic status influence an individual’s ability to engage in financially effective behavior. The results also found that adopting a short-term focus, rather than future orientation, is a key inhibitor of financial effectiveness. Consequently, it is suggested that those developing financial capability programs address these behavioral and contextual factors rather than concentrating purely on literacy.

### [Loan Default and Unemployment Rates (2013)](https://upcea.edu/wp-content/uploads/2018/03/Exploring-the-Determinants-of-Student-Loan-Default-Rates.pdf)
 * While the majority of loan defaults come from traditional college graduates or students who do not finish their degree, professional, continuing, and online 
   education units may be able to play a part in adding value to credits earned through degree completion or alternative credentialing. The latter may also play a role in helping to reduce loan defaults by increasing an employee’s value in the workplace. Other factors that could also increase value are more convenient delivery of programming through online delivery and more modular learning.

### [LinkedIn Learning’s Top Skills Companies Need](https://www.linkedin.com/business/learning/blog/top-skills-and-courses/the-skills-companies-need-most-in-2020and-how-to-learn-them) 
 * Soft skills: Creativity, persuasion, collaboration, adaptability, emotional intelligence
 * Hard skills: Blockchain, cloud computing, analytical reasoning, artificial intelligence, UX design, business analysis, affiliate marketing, sales, scientific 
   computing, video production

### [Classification of Instructional Programs Code Documentation](https://www.ice.gov/sites/default/files/documents/stem-list.pdf) 
 * The U.S. Department of Homeland Security (DHS) STEM Designated Degree Program List is a complete list of fields of study that DHS considers to be science, 
   technology, engineering or mathematics (STEM) fields of study for purposes of the 24-month STEM optional practical training extension described at 8 CFR 214.2(f).
 
### [Definition of “Heightened Cash Monitoring”](https://studentaid.gov/data-center/school/hcm)
 * The U.S. Department of Education (ED) may place institutions on a Heightened Cash Monitoring (HCM) payment method to provide additional oversight of cash 
   management. Heightened Cash Monitoring is a step that FSA can take with institutions to provide additional oversight for a number of financial or federal compliance issues, some of which may be serious and others that may be less troublesome.

### [Definition of Public “non-profit”](https://www.edmit.me/blog/whats-the-difference-between-a-for-profit-and-a-nonprofit-university)
 * By definition, public universities, which are mainly funded by state governments, are not-for-profit.

### [Why we think there are zeros for the variables PCTFLOAN and PCTPELL: students mistakenly are not applying](http://www.collegescholarships.org/loans/community.htm)
 * “Many community college students mistakenly believe that they are not eligible to benefit from college financial aid programs. Consequently, they fail to fill out 
   and submit their Free Application for Federal Student Aid.”
 * “The Federal Direct Loan Program provides low interest loans to students at every stage of their college career.”

### [Why we think schools that allocated more to instructional expenditures give a better education than not](https://files.eric.ed.gov/fulltext/EJ973834.pdf)
 * “School districts that spent less than 60% on instruction had lower passing rates in all five subject areas than districts that spent more than 65%.”
 * “Further, the less than 60% expenditure districts had statistically significant lower Math passing rates scores than the 63-63.99% expenditure districts.”