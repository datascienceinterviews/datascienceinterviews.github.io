---
title: Deploying Machine Learning Models to Production
description: Complete guide to ML deployment - Docker containerization, Kubernetes orchestration, Flask/FastAPI APIs, AWS/GCP/Azure cloud deployment, CI/CD pipelines, and MLOps best practices.
---

# Home

<p align="center">
  <a href="https://dsprep.com/">
    <img src="https://repository-images.githubusercontent.com/275878203/13719500-bb75-11ea-8f3a-be2ffb87a6a2" width="120" alt="Go to website">
  </a>
</p>

## Introduction

This is a completely open-source platform for maintaining curated list of interview questions and answers for people looking and preparing for data science opportunities.

Not only this, the platform will also serve as one point destination for all your needs like tutorials, online materials, etc.

**This platform is maintained by you!** 🤗 You can help us by answering/ improving existing questions as well as by sharing any new questions that you faced during your interviews.

## Contribute to the platform

Contribution in any form will be deeply appreciated. 🙏

### Add questions

❓ Add your questions [here](https://github.com/datascienceinterviews/datascienceinterviews.github.io/issues). Please ensure to provide a detailed description to allow your fellow contributors to understand your questions and answer them to your satisfaction.

[![Add New question](https://img.shields.io/badge/Use%20label-Interview%20Questions-%3CCOLOR%3E.svg)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/issues/new)

🤝 Please note that as of now, you cannot directly add a question via a pull request. This will help us to maintain the **quality** of the content **for you**.

### Add answers/topics 

📝 These are the answers/topics that need your help at the moment

* [x] Add documentation for the project
* [ ] Online Material for Learning
* [ ] Suggested Learning Paths
* [ ] Cheat Sheets
    * [ ] Django
    * [ ] Flask
    * [ ] Numpy
    * [ ] Pandas
    * [ ] PySpark
    * [ ] Python
    * [ ] RegEx
    * [ ] SQL
* [ ] NLP Interview Questions
* [ ] Add python common DSA interview questions
* [ ] Add Major ML topics
    * [ ] Linear Regression 
    * [ ] Logistic Regression 
    * [ ] SVM 
    * [ ] Random Forest 
    * [ ] Gradient boosting 
    * [ ] PCA 
    * [ ] Collaborative Filtering 
    * [ ] K-means clustering 
    * [ ] kNN 
    * [ ] ARIMA 
    * [ ] Neural Networks 
    * [ ] Decision Trees 
    * [ ] Overfitting, Underfitting
    * [ ] Unbalanced, Skewed data
    * [ ] Activation functions relu/ leaky relu
    * [ ] Normalization
    * [ ] DBSCAN 
    * [ ] Normal Distribution 
    * [ ] Precision, Recall 
    * [ ] Loss Function MAE, RMSE 
* [ ] Add Pandas questions
* [ ] Add NumPy questions
* [ ] Add TensorFlow questions
* [ ] Add PyTorch questions
* [ ] Add list of learning resources

### Report/Solve Issues 

[![Issues](https://img.shields.io/github/issues/datascienceinterviews/datascienceinterviews.github.io)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/issues)

🔧 To report any issues find me on [LinkedIn](#maintained-by) or [raise an issue](https://github.com/datascienceinterviews/datascienceinterviews.github.io/issues) on GitHub.

🛠 You can also solve existing [issues on GitHub](https://github.com/datascienceinterviews/datascienceinterviews.github.io/issues) and create a pull request.

### Say Thanks

😊 If this platform helped you in any way, it would be great if you could share it with others.

[![](https://img.shields.io/badge/Share%20to-Linked%20In-blue?logo=linkedin&style=flat&labelColor=blue&color=black)](https://www.linkedin.com/sharing/share-offsite/?text=Check%20out%20this%20%F0%9F%91%87%20platform%20%F0%9F%91%87%20for%20data%20science%20content:&url=https://dsprep.com/)
[![](https://img.shields.io/badge/Share%20to-Twitter-blue?logo=twitter&style=flat&labelColor=black&color=blue)](https://twitter.com/intent/tweet?text=Check%20out%20this%20%F0%9F%91%87%20platform%20%F0%9F%91%87%20for%20data%20science%20content:%20%F0%9F%91%89%20https://dsprep.com/%20%F0%9F%91%88%20#data-science%20#machine-learning%20#interview-preparation)
[![](https://img.shields.io/badge/Share%20to-Facebook-blue?logo=facebook&style=flat&labelColor=black&color=blue)](https://www.facebook.com/sharer.php?s=100&p[title]=Free%20Data%20Science%20Preperation%20Platform&p[summary]=Check%20out%20this%20&p[url]=https%3A%2F%2Fdsprep.com%2F)

```
Check out this 👇 platform 👇 for data science content:
👉 https://dsprep.com/ 👈

#data-science #machine-learning #interview-preparation 
```

You can also star the repository on GitHub 
[![Stars](https://badgen.net/github/stars/datascienceinterviews/datascienceinterviews.github.io)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/stargazers) 
and watch-out for any updates
[![Watchers](https://badgen.net/github/watchers/datascienceinterviews/datascienceinterviews.github.io)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/watchers)

## Features 

* **🎨 Beautiful:** The design is built on top of most popular libraries like MkDocs and material which allows the platform to be responsive and to work on all sorts of devices – from mobile phones to wide-screens. The underlying fluid layout will always adapt perfectly to the available screen space.

* **🧐 Searchable:** almost magically, all the content on the website is searchable without any further ado. The built-in search – server-less – is fast and accurate in responses to any of the queries.

* **🙌 Accessible:**
    * **Easy to use:** 👌 The website is hosted on github-pages and is free and open to use to over 40 million users of GitHub in 100+ countries.
    * **Easy to contribute:** 🤝 The website embodies the concept of collaboration to the latter. Allowing anyone to add/improve the content. To make contributing easy, everything is written in MarkDown and then compiled to beautiful html.

## Setup

> No setup is required for usage of the platform

**Important:** *It is strongly advised to use virtual environment and not change anything in `gh-pages`*

### `Linux` Systems ![Linux](https://img.shields.io/badge/Linux-Systems-orange?logo=linux&style=plastic)

```shell
python3 -m venv ./venv

source venv/bin/activate

pip3 install -r requirements.txt
```

```shell
deactivate
```

### `Windows` Systems ![Windows](https://img.shields.io/badge/Windows-Systems-blue?logo=Windows&style=plastic)

```shell
python3 -m venv ./venv

venv\Scripts\activate

pip3 install -r requirements.txt
```

```shell
venv\Scripts\deactivate
```

### To install the latest

```shell
pip3 install mkdocs
pip3 install mkdocs-material
```  

### Useful Commands

* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.
* `mkdocs gh-deploy` - Use `mkdocs gh-deploy --help` to get a full list of options available for the `gh-deploy` command.
    Be aware that you will not be able to review the built site before it is pushed to GitHub. Therefore, you may want to verify any changes you make to the docs beforehand by using the `build` or `serve` commands and reviewing the built files locally.
* ~~`mkdocs new [dir-name]` - Create a new project.~~ No need to create a new project
    
### Useful Documents

* 📑 MkDocs: [https://github.com/mkdocs/mkdocs](https://github.com/mkdocs/mkdocs)

* 🎨 Theme: [https://github.com/squidfunk/mkdocs-material](https://github.com/squidfunk/mkdocs-material)

## FAQ

* Can I filter questions based on companies? 🤪

    As much as this platform aims to help you with your interview preparation, it is not a short-cut to crack one.
    Think of this platform as a practicing field to help you sharpen your skills for your interview processes. However, for your convenience we have sorted all the questions by topics for you. 🤓

    This doesn't mean that such feature won't be added in the future. 
    *"Never say Never"*
    
    But as of now there is neither plan nor data to do so. 😢

* Why is this platform free? 🤗

    Currently there is no major cost involved in maintaining this platform other than time and effort that is put in by every [contributor](https://github.com/datascienceinterviews/datascienceinterviews.github.io/graphs/contributors). 
    If you want to help you can [contribute here](#contribute-to-the-platform). 
    
    If you still want to pay for something that is free, we would request you to donate it to a charity of your choice instead. 😇

## Credits

### Maintained by

👨‍🎓 ***Kuldeep Singh Sidhu*** 

Github: [github/singhsidhukuldeep](https://github.com/singhsidhukuldeep)
`https://github.com/singhsidhukuldeep`

Website: [Kuldeep Singh Sidhu (Website)](http://kuldeepsinghsidhu.com)
`http://kuldeepsinghsidhu.com`

LinkedIn: [Kuldeep Singh Sidhu (LinkedIn)](https://www.linkedin.com/in/singhsidhukuldeep/)
`https://www.linkedin.com/in/singhsidhukuldeep/`

### Contributors

😎 The full list of all the contributors is available [here](https://github.com/datascienceinterviews/datascienceinterviews.github.io/graphs/contributors)

## Current Status

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/datascienceinterviews)
[![Website shields.io](https://img.shields.io/website?url=https%3A%2F%2Fdsprep.com%2F)](https://dsprep.com/)
[![GitHub pages status](https://img.shields.io/github/deployments/datascienceinterviews/datascienceinterviews.github.io/github-pages)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/deployments/activity_log?environment=github-pages)
[![GitHub up-time BOT](https://badgen.net/uptime-robot/month/ur967659-422c6e77bfb79bb6a47c642c)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/deployments/activity_log?environment=github-pages)
[![Commits](https://img.shields.io/github/last-commit/datascienceinterviews/datascienceinterviews.github.io)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/commits/master)
[![DependaBot](https://api.dependabot.com/badges/status?host=github&repo=datascienceinterviews/datascienceinterviews.github.io)](https://dependabot.com/)

[![Issues](https://img.shields.io/github/issues/datascienceinterviews/datascienceinterviews.github.io)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/issues)
[![Total Commits](https://badgen.net/github/commits/datascienceinterviews/datascienceinterviews.github.io)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/commits/master)
[![Contributors](https://badgen.net/github/contributors/datascienceinterviews/datascienceinterviews.github.io)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/graphs/contributors)
[![Forks](https://badgen.net/github/forks/datascienceinterviews/datascienceinterviews.github.io)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/network/members)
[![Stars](https://badgen.net/github/stars/datascienceinterviews/datascienceinterviews.github.io)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/stargazers)
[![Watchers](https://badgen.net/github/watchers/datascienceinterviews/datascienceinterviews.github.io)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/watchers)
[![Branches](https://badgen.net/github/branches/datascienceinterviews/datascienceinterviews.github.io)](https://github.com/datascienceinterviews/datascienceinterviews.github.io/branches)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)
[![repo- size](https://img.shields.io/github/repo-size/datascienceinterviews/datascienceinterviews.github.io)](https://github.com/datascienceinterviews/datascienceinterviews.github.io)
[![Followers](https://img.shields.io/github/followers/singhsidhukuldeep?style=plastic&logo=github)](https://github.com/singhsidhukuldeep?tab=followers)