# ETTI, BIOSINF Course: *Artificial Intelligence II, Deep Learning Methods*, Spring 2024
___

This repository will contain all lectures, labs and project resources, code templates and (maybe) some solutions.

## Courses:

| **Nr.** | **Date** |       **Topic**       | **Materials** |
|:-------:|:--------:|:---------------------:|:-------------:|
|    1    |   6.03  |  _Intro & AI Ethics_  |   [slides](course/C1%20-%20Intro&Ethics.pdf)           |
|    2    |  -  | _Optimization Basics_ |   [slides](course/C2%20-%20LinearNets.pdf)            |
|    3    |  -   | _Dense Neural Networks_ | [slides](course/C3%20-%20DNNs.pdf)              |
|    4    |  -   | _Convolutional Neural Networks_|     [slides](course/C4%20-%20CNNs.pdf)          |
|    5    |   -  |_Recurrent Neural Networks_|          [slides](course/C5%20-%20RNNs.pdf)       |
|    6    |  -  |_Applications_ |               |
|    -    |   -    |    **Written Exam**    |      |

## Labs:

| **Nr.** | **Date** |        **Topic**        | **Materials** |
|:-------:|:--------:|:-----------------------:|:-------------:|
|    0    |    -     |     _Lab Setup Guide_   |   [guide + requirements](https://github.com/Vladimirescu/BIOSINF_IA2/tree/main/lab_setup) |
|    1    |   13.03  |           _DNNs_        |   [notebook + utils](https://github.com/Vladimirescu/BIOSINF_IA2/tree/main/L1)            |
|    2    |   -   |           _CNNs_      |   [notebook + utils](https://github.com/Vladimirescu/BIOSINF_IA2/tree/main/L2)            |
|    3    |   -  | _Other Applications of CNNs_  |   [2 x notebook + utils](https://github.com/Vladimirescu/BIOSINF_IA2/tree/main/L3)           |
|    4    |   -  |         _RNNs_          |       [notebook + utils](https://github.com/Vladimirescu/BIOSINF_IA2/tree/main/L4)        |

## Project Rules:

The requirements for completing your practical project, along with a list of available projects can be found [here](Project_IA2.pdf). 
We'll be available for discussions regarding the status of your project every ___ 17:00-18:00.

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
|  Lab Homework(s) Submission Deadline  |  23:59 on the Wednesday before next lab  |
|  Project Submission Deadline  |  -  |
|  Project Presentations  |  -  |
___

Contact us if you have any questions via:
- [ana_antonia.neacsu@upb.ro](mailto:ana_antonia.neacsu@upb.ro)
- [vlad.vasilescu2111@upb.ro](mailto:vlad.vasilescu2111@upb.ro)
- or make a post on the courses' Team

___

## Recommended References:
- [Convex Optimization – Boyd and Vandenberghe](https://stanford.edu/~boyd/cvxbook/)
- [Berkley Convex Optimization and Approximation course](https://ee227c.github.io)
- [EPFL Optimization for ML Course](https://github.com/epfml/OptML_course)
- [MIT Deep Learning Book - Ian Goodfellow, Yoshua Bengio and Aaron Courville](https://github.com/janishar/mit-deep-learning-book-pdf)
- [Neural Networks and Deep Learning - A Textbook](https://www.charuaggarwal.net/neural.htm)
- [Awesome Deep Learning - A curated list of awesome Deep Learning tutorials and projects](https://github.com/ChristosChristofidis/awesome-deep-learning)
