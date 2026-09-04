---
title: Onyx Quant Bot
emoji: 📈
colorFrom: gray
colorTo: blue
sdk: docker
app_file: main.py
pinned: false
---

# Onyx Quantitative Research Platform

This repository is configured to run autonomously on Hugging Face Spaces using Docker.

The bot runs on Indian Standard Time (IST). It sleeps natively to consume ~0 RAM while hibernating, wakes up every 10 minutes during Indian market hours, and executes an intensive Machine Learning NLP EOD job at 6:00 PM IST.
