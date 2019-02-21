# Machine Learning

This repository will be used as a resource to prepare for a data scientist interview as well as a general resource for myself to look back to. 


## What We Expect From You, I.e., Your Qualifications

    Bachelor’s degree in Computer Science, Information Science, Mathematics, Physics or related discipline and 7 - 10 years of experience in that field. Master’s degree with 6 – 9 years of experience.
    
    
I have just completed my bachelors degree in **Applied Mathematics** along with a Computer Science minor. I have been programming for about 7 years now with my introductory programming expreiences coming from classes made available to me my sophomore year of high school. After my introduction to Computer Science through introductory and intermediate Computer Science classes my sophomore year, I then went onto take AP Computer Science my Junior year and lead the Computer Science club in building apps and Java projects my Senior year. 
 
For my first year of college I attended **SUNY Fredonia**, where I planned to major in Computer Science on the advanced computing track. I was able to test out of the first two Computer Science classes while continuing to take them for experience handling common tasks in **C++**, as in high school I mainly programmed in Java. Due to my testing out and with the aid of the chair of the Computer Science department, I was able to take Data Structures my first semester of my college experience, leading to me taking Analysis and Design of Algorithms my second semester and competing on a competitive programming team.
 
Though I had gained valuable experience in Mathematics (Discrete Math for Computer Science I and II, Calculus I and II), I believed that another institutuion may provide more opportunites for available higher level courses in different fields that peaked my interests. Upon transferring to **SUNY Oswego** for my sophomore year of college, I was lucky enough to gain entrance in to the first course in a two course series in **Artificial Intelligene**, learning about [state-space problem solvers](https://jakesauter.github.io/course-sites/csc416.html), [genetic algorithms](https://jakesauter.github.io/course-sites/csc416.html) and other early breakthroughs in AI alongside learning the langauge that most of these staples to the field were written in, being **Common LISP**. I continued on to take the second course in this series, which again was very lucky as it only came around once every two years and it happened to fall after the period when I took the first course in the series. 
 
During my time at SUNY Oswego I had the chance to take many exciting Computer Science courses including Systems Programming, in which I got more intimate with the shell (later having to create a Tiny Shell implementation), and Natural Language processing, in which I used **Python** and the **NLTK (Natural Language Tool Kit)** in order to implement theoretical language processing tasks that we were reviewing in class. Though Computer Science is what truly excites me, I knew after taking this second AI course (In which I implemented a **Feed Forward Neural Network** from scratch) that I needed a firmer Mathematical background. This lead me to change my major to Applied Mathematics, as the good thing to do is not always the easy thing.
 
All in all I am very pleased with my formal education as well as the two **Summer Research Positions** that I was able to aquire along the way due to my academic performance and previous project successes.
    
    Proficiency in Python.
    
Python has genuinely come about as my favorite programming language. It is elegant in the brevity and readability of the code, allowing a master of the language to accomplish much in one line while still allowing programmers not-so-familiar with the language to understand the processing. As we will later see in this repository, Python also has many useful libraries that are continually updated, and has found itself to be in the center of attention in this time of Machine Learning revolution and popularity due to widely supported libraries. This support includes **Google Cloud**, that allows users to run their Machine Learning algorithms on the most up to date hardware, including the newest **TPUs**. 

I very recently have had extensive experience with Python as I am finally a graduate and have the choice of langauge that I would like to work with on my independent projects and choose employers that also enjoy the same langauge that I do. This experience comes from some Bioinformatics training that I have been doing, very helpful for learning the language as after every problem I can consult the forum of answers to see if there is any solution that is remarkably elegant that I can learn from -- thus the reason why some problems on my [Rosalind Python Training Repository](https://github.com/jakesauter/rosalind_python_training) have 2 or more solutions.

   
    
    Experience with machine learning and data modeling techniques such as decision trees, random forests, SVM,
    neural networks and incremental response.
    
    
[**Capstone Project Machine Learning Week 1**](https://github.com/jakesauter/Molecular_Classification_Capstone/blob/master/files/Machine_Learning.pdf)

  * Machine Learning problem types
  * Supervised / Unsupervised learning
  * Model assessment (confusion matrices / cross-validation / metrics)
    * Not covered here: FOC curve, but previous experience with FOC in 2017 REU
  * Feature selection
  * Quadratic and Linear discriminants
  * K-Nearest-Neighbor
  * Decision Trees
  * Neural Nets (BRIEF)
  * SVMs (Covered more thoroghly in next presentation)
  * Results from KNN applications to data


[**Capstone Project Machine Learning Week 2**](https://github.com/jakesauter/Molecular_Classification_Capstone/blob/master/files/Machine_Learning_Continued.pdf)

The motivation for this presentation was to really get a hold of the mathematical foundations, capabilites and limits of SVMs, so in the most thorough manner capable in a week's span, I began with the foundations of Logistic Regression and worked towards SVMS, while following **Andrew NG's** [machine learning Coursera course](https://www.coursera.org/learn/machine-learning).
 
  * Logistic Regression
    * Linear Boundaries
    * Non-Linear Boundaries
    * Fitting Parameters (Using **Gradient Descent** of cost function)
  * SVMs
    * Regualarization
    * Output Interpretation
  * Random Forests (BRIEF)
  * Applications of KNN, Decision Tree, Random Forest, SVM, Logistic Regression
    
### Decision Trees
    
I have studied decision trees in Python during my 2017 summer REU in medical informatics. During this time my research was involved with **uncertainty** in machine learning problems, where the exact true label of a sample wasn't known, but the probability distribution of the sample over a discrete number of classes was known. This form of problem lead to the use of **Belief Decision Trees**, in which the decision node was not a specific class, but a probability distribution over all possible classes called a pignistic probability. For this reserach we compared the performance of this kind of classifier with **standard decision trees** as well, so I do have some expereience with common implementations as well. 

I also have previous experience with decision trees in R, though it was breif experience. I have learned that decision trees are usually binary decision trees (meaning a two way split at each node) even though that this is not the best way to split the input space. This is due to the ability to often easy interpretation by experts and even by ordinary users of this method. Decision trees are widely implemented in supervised machine learning approaches for this quality.

In reviewing decision trees I found a [video made by the Google Developers Youtube channel](https://www.youtube.com/watch?v=LDRbO9a6XPU&t=526s) extremely helpful in decifering **Gini impurity** and **information gain**.

 <img src="files/decision_trees.jpg" width="50%" alt="decision trees"> 
 
 I felt that I needed a better understanding of Gini impurity so I did a walkthrough of the [wikipedia page](https://en.wikipedia.org/wiki/Decision_tree_learning)
 
<img src="files/gini_impurity.jpg" width="50%" alt="gini impurity"> 

### Random Forests

I came across random forests a few times in my academic career, though have only implemented them before breifly in R for my aforementioned [Capstone Project](https://github.com/jakesauter/Molecular_Classification_Capstone).

In reviewing random foresets, I found [this video](https://www.youtube.com/watch?v=QHOazyP-YlM) by Siraj Raval helpful.

If we understand the standard decision tree well, random forests should come very easily as they are simple a collection of decision trees constructed on random subsets of the data that are used to majority vote for the most likey class of a new sample.

<img src="files/random_forests.jpg" width="50%" alt="random forests"> 


### Support Vector Machines (SVMs)

Again, I have come across SVMs a few times in my studies and research. I put them to use during my [last summer's research](https://github.com/jakesauter/lateralization_project), though have more thoroughly studied them guided by my interest during my [Capstone Project](https://github.com/jakesauter/Molecular_Classification_Capstone). 

During my Capstone project experience I devoted an entire week just to familiarizing myself more with SVMs, composing a [presentation](https://github.com/jakesauter/Molecular_Classification_Capstone/blob/master/files/Machine_Learning_Continued.pdf) of what I learned from Andrew NG's section on logistic regression and SVMs from his course on machine learning. 

One aspect of SVMs that I am not super familiar with is **kernel functions**, though I have encountered them a few times and understand the basics, being helping nonlinear functions to be learned, I am more than willing to learn them for the job at hand.  

Summarized well from [this quora answer](https://www.quora.com/What-are-kernels-in-machine-learning-and-SVM-and-why-do-we-need-them), **kernels** are the idea of summing functions that imitate similarity. These kernels can be used to make nonlinearly seperable data into a much simpler problem that is linearly seperable. Though SVMs can solve nonlinear problems, choosing the exact features to be used can be tricky in nonlinear scenarios, so linearlizing a problem with kernels is a huge advantage when paired with SVMs.


### Artificial Neural Networks (ANNs)

My experience with ANNs comes from my Sophomore year [semester-long indepedent research project](https://jakesauter.github.io/course-sites/csc466_project.html) in which I implemented a Common LISP program that could construct aribitrary architectues of a simple **feed-forward ANN**. This ANN was used for memory compression of a form of board game solutions found via **Rote learning**.

Since then, since I have greatfully been granted the position to which this repository was constructed for, I have begun a refresher on Nerual Networks and will continue to expand my knowledge on the topic. I will begin my learning where I always like to start if content is present, on the [3 Blue 1 Brown Youtube Channel](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1).

 <img src="files/neural_nets_intro.jpg" width="50%" alt="Refresher to the introduction of neural networks"> 

 <img src="files/neural_nets_calculations.jpg" width="50%" alt="progoation of one layer in a neural network", style="transform:rotate(90deg);"> 


### Incremental Response

I have not heard the term "Incremental Response" before this prompt, though I was intially struck by the idea of **gradient descent**, and how the solution of a problem can be updated incrementally to achieve the optimal solution. With a little looking around I found a [maketing training website](https://blogs.sas.com/content/subconsciousmusings/2013/07/12/how-incremental-response-modeling-can-help-you-reach-the-right-target-group-more-precisely/) that described incremental response as a sort of experimental design, **second order** effects are attempted to be minimized while model changes are **incrementally added** in order to truly judge their effects on the model. 
    
    Experience with statistical tests and procedures such as ANOVA, Chi-squared, correlation,
    regression, and time series.
    
As noted before the reason for my change of major to Applied Mathematics was to form a more solid mathematical background. Once I began my full-time math studies I found Statistics to be the most intersting (and most likely the most probable to be useful in my career path) of the topics I was studying, and thus let this guide me into taking more Statistics classes then required and performing a [**Capstone Project**](https://github.com/jakesauter/Molecular_Classification_Capstone) in Statistics applied to molecular genomics. 
   
### ANOVA

Also during my last semester at SUNY Oswego I had the pleasure of taking a **Non-parametric Statistics Course** in which we reviewed **ANOVA**, while covering implemetnations in the **R programming language**

Analaysis of Variance (ANOVA) is a statisitical test to **analyze the differences among three or more group means in a sample**. In order for the **parametric** version of ANOVA to be valid,
* The distribution of the **residuals** of the group means the values within each group must be Normally distributed.
* The variances of all of the groups are equal 
* No temporal or spatial (or any other in fact) trend is present
* Data values are independent and random

If some or none of these assumptions are met the **non-parametric Kruskal-Wallis test** may be applicable. 

Both of these test were covered in [Environmental Statistics HW 4](files/Environmental_Statistics_HW_4.pdf), also at this time I welcome you to look the the previous 3 homeworks that are also in the files directory, links below. 

[Environmental Statistics HW 1](files/Environmental_Statistics_HW_1.pdf)

[Environmental Statistics HW 2](files/Environmental_Statistics_HW_2.pdf)

[Environmental Statistics HW 3](files/Environmental_Statistics_HW_3.pdf)


### Chi-Squared Test

During the second semester of my Junior year I enrolled in Mathematical Statistics II in which we reviewed the mathematical foundations of many Statistical tests and concepts, Chi-Squared was included in these studies.

The Chi-Squared test is used to determine if there is a statistically significant difference in the expected versus observed sizes of groups for the Pearson Chi Square test, which is used for categorical data. The standard definition of statistical significance is applied here, with a common p-value of .05.

 <img src="files/chi_square_test_stat.png" width="50%" alt="chi square test staistic"> 
 
The Chi-Squared test can also be applied in the continuous case to determine if a sample from a normally distributed population has a particular varaince. The test statistic is the sum of squares about the sample mean, divided by the nominal value for the variance (i.e. the value to be tested as holding). This test statistic has a chi-squared distribution with n − 1 degrees of freedom.

<img src="files/one_pop_chi_sq.png" width="30%" alt="one population chi square"> 

 
The Chi-Squared test can also be used to assess how well a sample distribution fits a coninuous distribution such as the **Normal Distribution**. I found [this video](https://www.youtube.com/watch?v=HabIKLG92MQ) and the [following video](https://www.youtube.com/watch?v=OnCL2JlD86k) very helpful as a refresher for this concept. Essentially we bin the distrubtion and our test statistic involves the **expected area** in the interested area of the distribtion minun the **observed area**. This area comes from the **normalized data** and thus can be seen as **what percent of the data would we expect to see in a particular bin vs. what percent of data we observe in that particular bin**.

### Correlation 

Pearson Correlation Coefficient

<img src="files/correlation.png" width="75%" alt="correlation equation"> 


### Regression

During the previously mentioned non-parametric statistics course **simple linear regression** and **multiple linear regression** were also covered. These were implemented in R in class on simple data sets along with evaluating the **r-squared**, **adjusted r-squared** and **F-test** of the model.

### Time Series
    
I do not have much expereience with time series analysis though I have attended a talk in which a **recurrent neural network**(LSTMs) was used to predict results of sports games based on a sliding window of previous sports game results. I found an [intersting blog](https://blog.statsbot.co/time-series-prediction-using-recurrent-neural-networks-lstms-807fa6ca7f) of this topic that I will be sure to review more closely before the interview.
    
    Experience with survey sampling methodologies and data collection techniques.
   
In the now seemingly very useful non-parametric statistics course that I had the pleasure of taking I was also exposed to many different sampling methodologies as we must be aware of the bias that can be introduced with different sampling methods, and even some methods are only applicable if certain sampling techniques are performed due to assumptions that must be met for statisticall sound tests.

**Sampling Techniques**

* **Convenient Sampling** -- Generally accepted as not a good idea, collecting samples because they are easily available.
* **Voluntray Response Sampling** -- Also can be a problematic sampling method, collecting data from individuals who volunteer to answer. All of these individuals may share common characteristics, for example people who have less of a carbon footprint may be more likey to take a survey about their carbon footprint.
* **Probability Sampling** -- A sampling methodology in which randomness is used to reduce sampling bias.
* **Simple Random Sampling** -- Every sample is equally likely of being included in the study.
* **Multi-Stage Sampling** -- A simple random sample of a simple random sample. Such as randomly selecting the states to be in an environmental study then randomly sampling the state parks to be tested in those states.

**Observational Study** -- Measurements for the variable of interest are colected on the individual but there is no attempt to modify or influence the individuals. The main goal of an observational study is to compare and observe existing characteristics or groups. These kinds of studies are particularly useful when the variable of interest cannot be controlled or influenced.

**Experiment** -- A study in which a treatment or condition is imposed on individuals and the response is measure. The main goal of an experiment is to examine the effect of an intervention on a response variable. Experiments are particularly useful when investigating cause and effect. 

**Types of Observational Studies:**

* **Retrospective Study** -- A study where information is collected on an individuals past
* **Prospective Study** --  A study that collects current/future information on subjects at regular intervals
* **Cross-Sectional Study** -- Information is collected on individuals at **one** specific point in tie
* **Case-Control Study** -- A study where a collection of individuals witha certain characterisitc are measures (cases) and a collection of individuals without that condition (controls) are collected, then the the groups are **compared**
* **Cohort Study** -- A study that examines a group of homogenous individuals regularly over time. The main goal is to examine the emergence of a condition of interest over time. 

**Types of Experiments**

* **Randomized Comparative Experiment** -- A method of experiment where the effect of two or more treatments are compared and subjects are assinged to groups by random chance
* **Completely Randomized Design** -- All individuals in the experiments are assigned to treatment type completely at random (Equals sample sizes for treatment type is not required).
* **Block Design** --  Treatment type is randomly assigned to groups of individuals that are known to be similar in some way that is expected to impact the treatment response. 
* **Matched-Pairs Design** -- Pairs are chosen that are closely related by the characteristic of interst, with one of the individuals being assigned to each condition. Individuals are often paired with themselves in a temporal study

**Lurking variables** are variables that may have an impact on the response variable but were not considered during the experiment.

**Confounding factors** occur when the effect of one factor cannot be distinguished from the effect of another factor.

    Ability to lead small-sized teams.

During my time at iD tech I found myself leading others and organizing many internal situations. As a technical coordinator my job was to make sure that the camp ran smoothly, assisting technology issues in the camp and instructing students that needed more help in my free time.
    
    Applicants selected will be subject to a government security investigation and must meet eligibility requirements
    for access to classified information. U.S. citizenship is required.
    
I am a United States citizen. As I currently am employed part-time by FedEx Express, I have airport security clearance.


## Nice To Haves

    Experience with SciPy, NumPy and Pandas packages.
    
From my 2017 REU in Medical Informatics and other various times I have encountered these packages that make scientific computing much easier.

[SciPy](https://docs.scipy.org/doc/scipy/reference/) -- Tutorial for SciPy from official docs

From Scipy I have also used the **scipy.special.comb** combinations library, **scipy.stats.binom** binomial distribution library and **scipy.secial.product** for the cartesian product library. 

[NumPy](https://docs.scipy.org/doc/numpy/user/basics.html) -- NumPy basics from official docs

I have seen numpy around though have never done anything too computationally expenisve in python. I was wondering the advantages of using NumPy over standard python lists and [this stackoverflow answer](https://stackoverflow.com/questions/993984/what-are-the-advantages-of-numpy-over-regular-python-lists) explains that it **can reduce memory storage by a factor of 5**, while also making reading and writing operations quicker than standard python can. This difference in size comes from the flexibility of python lists as each element in the list is actually a **4 byte pointer**, pointing to at least a **16 byte object** (smallest possible python object), while numpy can store fixed precision uniform variable type value arrays.

Another advantage of NumPy is that elementwise operations can also be performed, removing the need of list comprehenesions for simple calculations.

Uses: 

> \>> import numpy as np  
> \>> list_a = np.array([1,2,3,4])  
> \>> list_b = np.array([2,3,4,5])  
> \>> \# dot product of two arrays  
> \>> list_a * list_b  
> \>> \# appending to a numpy array (appends list_b to list_a)  
> \>> np.append(list_a, list_b)  
> \>> \# multi-dimentsional arrays  
> \>> list_c = np.array([[1, 2, 3],  
                     [4, 5, 6],   
                     [7, 8, 9]])  
> \>> \# creating a numpy array of zeros with a predefined shape of   
> \>> \# two rows (lists) and 3 columns (entries in each list)    
> \>> list_d = np.zeros((2,3))   
> \>> \# saving a numpy array to file in binary format    
> \>> np.save('list_d.npy', list_d)    
> \>> \# reading that same array back from memory   
> \>> np.load('list_d.npy')   

[Pandas](https://pandas.pydata.org/) -- Official docs for Pandas library

I have seen the power of Pandas at work for CSV reading in python. Pandas data structures are also easily convertable into numpy arrays which are then very compatible with modern day libraries such as **Tensorflow**.

In [their own words](http://pandas.pydata.org/pandas-docs/stable/getting_started/overview.html) **pandas** is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real world data analysis in Python. Additionally, it has the broader goal of becoming the most powerful and flexible open source data analysis / manipulation tool available in any language. 

Interestingly, pandas is **built ontop of numpy**. Its main data structure for 2D+ data is the DataFrame.

Specific advantages that caught my eye with pandas data frames are

* Intelligent label-based slicing, fancy indexing, and subsetting of large data sets Intuitive merging and joining data sets
* Flexible reshaping and pivoting of data sets
* Robust IO tools for loading data from flat files (CSV and delimited), Excel files, databases, and saving / loading data from the ultrafast HDF5 format (This is where I have put pandas to use before)
* Time series-specific functionality: date range generation and frequency conversion, moving window statistics, moving window linear regressions, date shifting and lagging, etc.

As for Uses, Pandas provides a great [10 min introdution page](http://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html).

Uses: 

> \>> import numpy as np  
> \>> import pandas as pd  
> \>> df = pd.DataFrame({'A': 1.,  
                      'B': pd.Timestamp('20130102'),  
                      'C': pd.Series(1, index=list(range(4)), dtype='float32'),  
                      'D': np.array([3] * 4, dtype='int32'),  
                      'E': pd.Categorical(["test", "train", "test", "train"]),  
                      'F': 'foo'})  
> \>> \# or we can read in a data frame from a CSV  
> \>> df2 = pd.read_csv()  
> \>> \# print the first few rows and columns  
> \>> df2.head()  
> \>> \# convert the dataframe to a numpy nd array  
> \>> df.to_numpy()  
> \>> \# when accessing rows of the data, splicing has to be used even  
> \>> \# for only one row  
> \>> df[0:1]  
> \>> \# print data types in data frame  
> \>> df.dypes  
> \>> \# five number summary of each column  
> \>> df.describe()  
> \>> \# sort data by column  
> \>> df.sort_values('column_name', ascending=False)   
> \>> \# accessing data by column name  
> \>> df.column_name   
> \>> \# or  
> \>> df['column_name']   
> \>> \# we can do this with multiple columns too   
> \>> df[['column_1', 'column_2']]   
> \>> \# df.loc can slide from a row to a row and a list of columns   
> \>> df.loc[row_1:row_2, ['col_1','col_2']]   
> \>> \# just like R we can also use the : to access all rows or columns    
> \>> df.loc[:, ['col_1','col_2]']   
> \>> \# if we wanted every other column    
> \>> df.loc[:, df.columns[0::2]]   
> \>> \# filtering data by a threshold value   
> \>> df[df['col_name'] > threshold]   
> \>> df[df['month'].isin['Jun', 'July', 'Aug']]   
> \>> \# iterating through a data frame   
> \>> for index,row in df.iterrows():   
    print(index, row['col_1_name'], row['col_2_name'])  
> \>> \# finally we can write the data frame to a CSV   
> \>> df.to_csv('file_name.csv')  

    
[scikit-learn](https://scikit-learn.org/stable/) -- A library that I thought was worth mentioning to make machine learning data analysis quick and easy. 

scikit-learn has great documentation with syntax similar to the MATLAB prediction libraries I have used before.
    
    Experience with Theano, Torch, Caffe, Tensorflow, Leaf or Autumn.
    
I have had brief expereience with Tensorflow during my Sophomore year AI project, in which I used Tensorflow to test my ANN architecture with the **Adam Optimizer** to see if in the very best possible case of my implmentation that the architecture would work. 

[simple_mlp_tensorflow.py](files/simple_mlp_tensorflow.py)

[simple_feed_forward_net.py](files/simple_feed_forward_net.py)

[dobo_nn_continuous_feed_forward.py](files/dobo_nn_continuous_feed_forward.py)

    Experience with developing in a Linux environment.
    
I have been using Linux as my main operating system for close to 4 years now througout my Computer Science career. I have developed solely in Ubuntu during this whole period and find the ease of use for programming applications to be unparalled. I mainly used **Vim** as my editor of choice for about 2 years as I also did a lot of work on server, though now my editor of choice is **Gedit** with my configurations of plugins that help accelerate my workflow. I prefer these lightweight editors as I find larger editors to be clunky and many of the tools getting in the way of my development. 
    
    
    Knowledge of machine learning acceleration techniques.
    
When I think of "machine learning acceleration techniques" I am thinking of using the appropriate hardware for the job. I have made a few short programs on **Google Cloud** in which users have the option to make use of TPUs, GPUs or stadard chips. I have also found out that alongside acceleration, **persistance**, or the ability to pick up training a model where you have left it off, is one of the most important features of a tuned machine learning pipeline. Because of this I wrote a custom function to **parse trained neural nets** and reinitialize them in my Common Lisp implementation of ANNs.
    
    Knowledge of radio communication technologies, i.e., coursework, amateur radio, etc.
    
The only experience I have involving anything close to radio technologies is the theoretical concepts covered in Electromagnetics (Physics II) 

I have also covered **Partial Differential Equations and Orthogonal Functions** in a 400 level math class at SUNY Oswego. I have reviewed Fourier Analysis and have posted the whiteboards below.

 <img src="files/fourier_1.jpg" width="50%" alt="fourier transform"> 
 
 <img src="files/fourier_2.jpg" width="50%" alt="fourier transform"> 
 
 I was not satisified with assuming the rotational properties of e^x in the complex plane, so did some more investigating
 
 <img src="files/e_1.jpg" width="50%" alt="e"> 
 
 <img src="files/e_2.jpg" width="50%" alt="e"> 

    
    Knowledge of or past experience working within an agile environment.
    
As most of my work has been independent or small-team reserach, I have not had any active hands-on expereience with the agile environment. I am aware of basics of the agile system and am always up to learning new development processes.    
    
    Experience with writing government proposals.
    
I have no experience in writing government proposals.    
    
    Active security clearance.

I have an active Syracuse Hancock Airport security clearance, though no direct governmental security clearances.
