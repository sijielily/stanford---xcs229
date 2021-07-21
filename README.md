# Introduction

This repo records my answers to all questions from the excercises of XCS229
(Summer 2021 and Autumn 2021). https://online.stanford.edu/courses/xcs229i-machine-learning

I tried to record all details in my scripts and pdf files. If you see any
mistake, please let me know by
[opening a new issue](https://github.com/sijielily/stanford-xcs229/issues/new?template=your-question-or-bug-report.md).


I find some of the homeworks in an earlier version
(https://see.stanford.edu/Course/CS229) of this course interesting, so I chose
to do some and placed the answers in the `previous_cs229` fold.



# Development

Create virtual environment:

```
conda env create --prefix venv -f env-conda.yml
```

Start the server

```
jupyter notebook --no-browser --ip 0.0.0.0
```

Export virtual environment:

```
conda env export --prefix venv > env-conda.yml
```



