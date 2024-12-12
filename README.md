# Applied Deep Learning on Graphs

<a href="https://www.packtpub.com/en-us/product/applied-deep-learning-on-graphs-9781835885970"><img src="https://content.packt.com/B22118/cover_image.jpg" alt="no-image" height="256px" align="right"></a>

This is the code repository for [Applied Deep Learning on Graphs](https://www.packtpub.com/en-us/product/applied-deep-learning-on-graphs-9781835885970), published by Packt.

**Leverage graph data for business applications using specialized deep learning architectures**

## What is this book about?
Learn how applied graph deep learning crafts intelligent systems by decoding data's interconnected nature with modern deep learning and see how sophisticated graph-based models yield precision, revealing fresh insights across diverse domains.

This book covers the following exciting features:
* Discover how to extract business value through a graph-centric approach
* Develop a basic understanding of learning graph attributes using machine learning
* Identify the limitations of traditional deep learning with graph data and explore * specialized graph-based architectures
* Understand industry applications of graph deep learning, including recommender systems and NLP
* Identify and overcome challenges in production such as scalability and interpretability
* Perform node classification and link prediction using PyTorch Geometric

If you feel this book is for you, get your [copy](https://www.amazon.com/Applied-Deep-Learning-Graphs-Architectures/dp/1835885969/ref=sr_1_1?crid=2Z2Y6LFPOKVDB&dib=eyJ2IjoiMSJ9.91J9RVvukFIvzCqpokDeeVk-nMR5f4a9uRhzImKdwSoyrQgFWS87c9RQn2T2cmbY3-qlzS-hvjRo_Hijjgk4qH9qVcEpBwaGZ5PIgSb3mjUtAcXWE0tUjQzp8sdFDtOLVjyk2fkPYqWereDvY7VvxFgtFGNKN4VaPeMs2CMPjEqqPYaODqLYscvULNXVEJRZJhIIvcVk4USDQWRJRn0tNdrvGEwlk4RKQG1DnHbJzm8.vXq3ttL8FiCTXS4uXkYrSYZjG0aSfWmdLSWXmoYo-mw&dib_tag=se&keywords=applied+deep+learning+on+graphs&qid=1733986109&sprefix=applied+deep+lear%2Caps%2C341&sr=8-1) today!
<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders. For example, chapter - 1.

The code will look like the following:
```
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
correct = float(pred[data.test_mask].eq(
    data.y[data.test_mask]).sum().item())
accuracy = correct / data.test_mask.sum().item()
print(f'Accuracy: {accuracy:.4f}')
```
### Create conda environment
```bash
conda create -n applied-graph-nn-book python=3.12
```

### Install dependencies
```bash
pip install -r requirements.txt
```

**Following is what you need for this book:**
For data scientists, machine learning practitioners, researchers delving into graph-based data, and software engineers crafting graph-related applications, this book offers theoretical and practical guidance with real-world examples. A foundational grasp of ML concepts and Python is presumed.

With the following software and hardware list you can run all code files present in the book (Chapter 1-12).

## Software and Hardware List
| Chapter | Software required | OS required |
| -------- | ------------------------------------ | ----------------------------------- |
| 1-12 | Jupyter Lab/Google Colab | Windows, macOS, or Linux |
| 1-12 | Python | Windows, macOS, or Linux |

## Related products
* Machine Learning with PyTorch and Scikit-Learn [[Packt]](https://www.packtpub.com/en-us/product/machine-learning-with-pytorch-and-scikit-learn-9781801819312) [[Amazon]](https://www.amazon.com/Machine-Learning-PyTorch-Scikit-Learn-learning/dp/1801819319/ref=sr_1_1?crid=1EESLDNK7IV5W&dib=eyJ2IjoiMSJ9.BkD7X7COBYyhwaZaMCtJpaM4K--E7Ec4kSwBMS35QrhFZ6mcKPl_xN9RQ7hPApSw-SepogxeOAMhYVQzYxnRdIkFmmcUO43ey2QuSptUGIC7dAvS0bvHkG108SxDf32DpnjBBaDq5YH7DCiB6gV005wjZyru5Ik2TpWYX3gd2RgmjZKzIfIOqHXhyG6oKDalHuW9Il07cpIy4X1F5vyOu2pWbcXntwubG7BhcAFGLsA.SswKqg2ayLCbNljSZJEkr8Y8GDCNpd7AQL11C0e57kg&dib_tag=se&keywords=Machine+Learning+with+PyTorch+and+Scikit-Learn&qid=1733986309&sprefix=machine+learning+with+pytorch+and+scikit-learn%2Caps%2C652&sr=8-1)

* Deep Learning with TensorFlow and Keras – 3rd edition [[Packt]](https://www.packtpub.com/en-us/product/deep-learning-with-tensorflow-and-keras-3rd-edition-9781803232911) [[Amazon]](https://www.amazon.com/Deep-Learning-TensorFlow-Keras-reinforcement/dp/1803232919/ref=sr_1_1?crid=2O2ZMJPCAKBX&dib=eyJ2IjoiMSJ9.sfePZ3Bi-xqZg2Ia9iYK0QH5lovGwcuAWOPwYo1Jus-K6auexMCQeTeKIx7IOOdLDWiYr7aKCUkfCae_ho8zDSEftbX248wjlaT0lsg-iSoOHokc2C3KMo7GCO5M1eTU.FDn3eD0Tn98ezYZrYbONuzl2haihA9EDTXarAM4fPFc&dib_tag=se&keywords=Deep+Learning+with+TensorFlow+and+Keras+%E2%80%93+3rd+edition&qid=1733986429&sprefix=deep+learning+with+tensorflow+and+keras+3rd+edition%2Caps%2C1184&sr=8-1)

## Get to Know the Authors
**Lakshya Khandelwal** holds a bachelor’s and master’s degree from IIT Kanpur in mathematics and computer science and has 8+ years of experience in building scalable machine learning products for multiple tech giants. He has worked as a lead ML engineer with Samsung, building natural language intelligence for the very fi rst version of Bixby. He has also worked as a data scientist with Adobe, developing search bid optimization solutions as part of the advertising cloud suite for major enterprises across the globe. In addition, he has led natural language and forecasting initiatives at Walmart, building next-generation AI products for millions of customers. Lakshya currently leads AI for AirMDR, building agentic AI for the cybersecurity domain.

**Subhajoy Das** is a staff data scientist with 7 years of experience under his belt. He graduated from IIT Kharagpur with a bachelor’s and master’s degree in mathematics and computing. Since then, he has worked in organizations at varying stages of growth: from fast-growing e-commerce start-ups such as Meesho to behemoths such as Adobe. He has driven several pivotal features in every company he has worked in, including building an end-to-end recommendation system for the Meesho app and curating interesting advertising using reinforcement learning-based optimizations in Adobe Advertising. He is currently working at Arista Networks, building AI-driven apps that are responsible for the cybersecurity of several Fortune 500 companies..
