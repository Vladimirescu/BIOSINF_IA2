# ETTI, BIOSINF: *Artificial Intelligence II, Deep Learning Methods*, Spring 2026

This repository will contain all lectures, labs and project resources, code templates and (maybe) some solutions.

## Courses:

| **Nr.** | **Date** |       **Topic**       |**Materials** |
|:-------:|:--------:|:---------------------:|:-------------:|
|    1    |   3.03  |  _Intro & AI Ethics_  |  [slides](course/C0%20-%20Intro&Ethics.pdf)
|    2    |  10.03  | _Optimization Basics & DNNs_ | [slides optimization](course/C1%20-%20Linear%20Nets.pdf) + [slides DNNs](course/C3%20-%20DNNs.pdf) |
|    3    |  17.03   | _Convolutional Neural Networks_ | |
|    4    |  24.03   | _Recurrent Neural Networks_ |  |
|    5    |  31.03  | _Graph Neural Networks_ |  |
|    -    |  7.04  | **Project Presentations** | |    
|    -    |  9.04    | **Project Presentations** |  |  
|    -    |  **TBD**    | **Exam** |    |

## Labs:

| **Nr.** | **Date** |        **Topic**        | 
|:-------:|:--------:|:-----------------------:|
|    0    |    -     | _Lab Setup Guide_ |   
|    1    |   12.03  | _Intro_ +  _DNNs_ |   
|    2    |   19.03   | _CNNs_ |   
|    3    |   26.03  | _RNNs_ |   
|    4    |   2.04  | _CNNs_ + _RNNs_ / Transformers |  

## Project Rules:
### 5 March - Q&A session for Project details.
For (some) projects, students can work in teams of 2. Each team should choose a project until **13 March**, which implies writing your name on the spreadsheet present in Teams - files.
The requirements for completing your practical project, along with a list of available projects can be found [here](Project_IA2.pdf). 
We'll be available for discussions regarding the status of your project every **Thursday after Course / Lab.**

## Lab Homework Rules:

1. Each homework should composed of a single Jupyter Notebook, containing your code, results and explanations.
2. You could create the homework by continuing the lab notebook, but only what you add after will be scored.
3. If your homework requires any additional external files (e.g. `.py`), you'll store everything into a `.zip` archive of the form:
`Name_L[x].zip`, where `x` is the lab number.
4. Do not include any datasets in the archive -- all homeworks are based on the datasets discussed at lab, therefore it is
recommended to use the already defined `torch.utils.data.Dataset` with the appropriate path of your dataset folder. Or, if you consider necessary to use any external data, add the corresponding `wget` or `curl` commands into your notebook for downloading the additional files $+$ the corresponding pre-processing.
5. ❗**One should be able to run the homework notebook without any effort. Please make sure you include in your archive any 
local dependencies related to your notebook, as well as all neccessary ```!pip install ...``` if you use
additional packages.**
6. ❗**Before saving the network for submitting, make sure the outputs of all cells are visible after running.**
7. ❗**All your experiments & results should be accompanied by explanations attached in adjacent notebook cells.**

## Deadlines:

| **Name** | **Date** |
|:-------:|:--------:|
|  Lab Homework(s) Submission  |  6.04.2026, 23:59  |
| Project Topic Selection | 10.03.2026, 23:59, @ MS Teams Excel | 
|  Project Submission  |  6.04.2026, 23:59, @ Moddle Assignment  |
|  Project Presentations  |  7.04.2026 / 9.04.2026, @ ETTI  |
___

Contact us if you have any questions via:
- [ana_antonia.neacsu@upb.ro](mailto:ana_antonia.neacsu@upb.ro)
- [vlad.vasilescu2111@upb.ro](mailto:vlad.vasilescu2111@upb.ro)

or make a post on the courses' Team.

___

## Recommended References:
- [Convex Optimization – Boyd and Vandenberghe](https://stanford.edu/~boyd/cvxbook/)
- [Berkley Convex Optimization and Approximation course](https://ee227c.github.io)
- [EPFL Optimization for ML Course](https://github.com/epfml/OptML_course)
- [MIT Deep Learning Book - Ian Goodfellow, Yoshua Bengio and Aaron Courville](https://github.com/janishar/mit-deep-learning-book-pdf)
- [Neural Networks and Deep Learning - A Textbook](https://www.charuaggarwal.net/neural.htm)
- [Awesome Deep Learning - A curated list of awesome Deep Learning tutorials and projects](https://github.com/ChristosChristofidis/awesome-deep-learning)
