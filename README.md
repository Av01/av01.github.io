## Hey there, I'm Aakash Vora! 👋
Talk to me about: *Machine Learning | Artificial Intelligence | Data Science | Generative AI | Reinforcement Learning | Logic | Optimization | Software Engineering | Cloud Technologies | Funtional Programming | Programming Languages | Type Theory | Category Theory | Quantum Computing*


I'm a passionate MS in Computer Science student with a focus on Machine Learning and Data Science with a strong experience in Software Engineering. I thrive on turning data into actionable insights and creating intelligent solutions. Let's explore my journey:

## 🔧 Technical Skills
- **Languages**:
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![Julia](https://img.shields.io/badge/Julia-9558B2?style=for-the-badge&logo=julia&logoColor=white)
  ![Java](https://img.shields.io/badge/Java-007396?style=for-the-badge&logo=java&logoColor=white)
  ![C/C++](https://img.shields.io/badge/C/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
  ![SQL](https://img.shields.io/badge/SQL-4479A1?style=for-the-badge&logo=sql&logoColor=white)
  ![Haskell](https://img.shields.io/badge/Haskell-5D4F85?style=for-the-badge&logo=haskell&logoColor=white)
  ![JavaScript/TypeScript](https://img.shields.io/badge/JavaScript/TypeScript-3178C6?style=for-the-badge&logo=javascript&logoColor=white)

- **Frameworks**: 
  ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
    ![Tensorflow](https://img.shields.io/badge/Tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
    ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
    ![Pytorch](https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
    ![Spark](https://img.shields.io/badge/Spark-E25A1C?style=for-the-badge&logo=apache%20spark&logoColor=white)
    ![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
    ![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon%20aws&logoColor=white)
    ![Ansible](https://img.shields.io/badge/Ansible-EE0000?style=for-the-badge&logo=ansible&logoColor=white)
    ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
    ![NLTK](https://img.shields.io/badge/NLTK-FFC300?style=for-the-badge&logo=nltk&logoColor=white)
    ![Numpy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
    ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
    ![MatplotLib](https://img.shields.io/badge/MatplotLib-3776AB?style=for-the-badge&logo=python&logoColor=white)
    ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
    ![Hugging-face](https://img.shields.io/badge/Hugging--face-3178C6?style=for-the-badge&logo=huggingface&logoColor=white)
    ![Seaborn](https://img.shields.io/badge/Seaborn-00599C?style=for-the-badge&logo=python&logoColor=white)
    ![Ray](https://img.shields.io/badge/Ray-FFA000?style=for-the-badge&logo=ray&logoColor=white)
    ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

## 🎓 Education
- **MS in Computer Science** | *Arizona State University* | *Aug 2022 - May 2024*
- **B. Tech in Computer Engineering** | *SVNIT (NIT, Surat)* | *Jul 2015 - May 2019*

## 💼 Experience

- **Data Science Intern** | *Dell Inc.* | *May 2023 - Aug 2023*
- **Associate Software Engineer**  | *JP Morgan Chase & Co.* | *Jan 2022 - Jul 2022*
- **Software Engineer** | *JP Morgan Chase & Co.* | *Jul 2019 - Jan 2022*
- **Software Engineer Intern** | *JP Morgan Chase & Co.* | *May 2018 - Jul 2018*




### 🚀 Projects

#### [Text-based games using Rollout and Language Models](https://av01.github.io/LLMwithRollout.pdf) | *Python, Pytorch, Transformers*
- *Abstract:* Text-based games have been used for a long time to evaluate the abilities of agents in terms of textual comprehension and reasoning. However, solving these games still poses a challenge to the AI community. Also, they involve planning and decision-making aspects which further complicates the problem. We propose to explore the use of online training algorithms like rollout in conjunction with Large Language Models to address this issue. We believe that textual capabilities combined with the cost improvement property of rollout should provide good results. We test our hypothesis on TextWorld games and see promising results using our approach.
- *Key Results:* Developed a new approach for solving text-based games using Large Language Models, boosting reward accumulation to 89% with rollout-based GPT-J as compared to 10.18% reward accumulation of vanilla GPT-J.

#### [Hybrid Approach to Parallel Stochastic Gradient Descent](https://av01.github.io/HybridSGD.pdf) | *Python, Pytorch, Ray*
- *Abstract:* Stochastic Gradient Descent is used for large datasets to train models to reduce the training time. On top of that data parallelism is widely used as a method to efficiently train neural networks using multiple worker nodes in parallel. Synchronous and asynchronous approach to data parallelism is used by most systems to train the model in parallel. However, both of them have their drawbacks. We propose a third approach to data parallelism which is a hybrid between synchronous and asynchronous approaches, using both approaches to train the neural network. When the threshold function is selected appropriately to gradually shift all parameter aggregation from asynchronous to synchronous, we show that in a given time period our hybrid approach outperforms both asynchronous and synchronous approaches.
- *Key Results:* Developed a new algorithm for faster and distributed execution of stochastic gradient descent, a hybrid of sync and async implementations showing at least 5% accuracy boost on CIFAR-10 dataset for the same training time.

#### [Hybrid OOD Robust Classifier](https://av01.github.io/OODClassifier.pdf) | *Python, Pytorch, Flow Models*
- *Abstract:* Discriminative models are widely used for classification tasks and they produce reliable results as long as the samples fall under the distribution of their training set. However, for samples outside their training set, they confidently misclassify them to belong to one of the classes. This reduces the reliability of models when used for critical applications. We propose to use a hybrid model to overcome this flaw. The hybrid model will help determine samples that do not belong to any of the classes by modeling the joint probability of the sample and its class. The hybrid model will consist of a generative model alongside a discriminative classifier. The discriminative classifier will be trained on samples that are out of distribution but lies in proximity to in-distribution samples. Our hypothesis is that the hybrid model will be resistant to misclassifying out-of-distribution samples as in-distribution, which is correctly verified through experiments.
- *Key Results:* Created a robust classifier that addresses out-of-distribution data problems by combining flow-based generative models and CNN-based discriminators, achieving 97% accuracy on transformed MNIST and 65% on Omniglot OOD images.

#### [Formal Chain for Reasoning using Language Models](https://av01.github.io/LLMonNUMGlue.pdf) | *Python, Pytorch, Transformers*
- *Abstract:* Word problems involving arithmetic and, in general, numerical reasoning are still challenging problems in AI research. Various approaches have been taken to tackle these problems. However, the two broad categories outlined in the previous works are Symbolic approaches and End-to-end Language Models. Traditional NLP systems have shown that Symbolic approaches are robust but less powerful. Whereas it has been shown by (Mishra et al., 2022) that the end-to-end approach for GPT3 performs extremely poorly. We are focusing on combining the best of both worlds in the project. We test out Few-Shot techniques, or prompt engineering techniques, and present a novel approach, which we call, Formal Chain of Reasoning. Our method outperforms the best baselines and demonstrates the potential for general use for other tasks. Furthermore, we present an analysis of the performance of our techniques on a range of available GPT3 models like davinci, ada, curie, etc. For the latter part of the study, we present how we trained GPT2, GPT1, and T5 to obtain results comparable to some GPT3 models.
- *Key Results:* Improved arithmetic and numerical reasoning in language models (e.g., BERT, GPT-X, T5) by over 2x, reaching 87.9% accuracy on NumGLUE TASK 4 using GPT-3 text-davinci-003 model and achieving near-human accuracy. Enhanced GPT-2, 100x smaller than GPT-3, elevating accuracy to 27% from nonsensical output.

#### [Data-driven dynamic sensor selection in IoT](https://ieeexplore.ieee.org/abstract/document/8929471) | *Python, Tensorflow, DQN, CNN*
- *International Publication*: “Data-driven dynamic sensor selection in internet of things” IEEE TENCON 2019.
- *Abstract:* The advent of Internet Of Things has led to the problem of an explosive outburst of data. Because of this, there is a need of an advanced data acquisition and data reduction system. We present two approaches for data reduction by sensor selection, leveraging the power of Machine Learning. The first approach uses Principal Component Analysis and the second approach uses Reinforcement Learning to identify k significant sensors from the n total sensors that closely resemble the original data statistically. This approach is applied on the data available from sensors deployed in South Brazil, as well as on the data collected from a sensor field we set up in the Computer Engineering Department. The results from both the datasets show significant data reduction while maintaining the characteristics of the original data.
- *Key Results:* Innovatively solved NP-hard Sensor selection using RL, boosting accuracy and efficiency. Our approach, applicable to various goals like coverage and energy efficiency, reduces the need for active sensors by 50%.



Let's connect and collaborate on exciting projects! Feel free to reach out to me via [Email](mailto:voraakash01@gmail.com) or connect with me on [LinkedIn](https://www.linkedin.com/in/aakashvora/).

