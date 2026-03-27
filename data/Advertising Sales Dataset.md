# Advertising Sales Dataset

## Description

The Advertising Sales dataset captures the sales revenue generated with respect to advertisement spending across multiple channels: **TV**, **Radio**, and **Newspaper**. It is used to understand how ad budgets across different media influence overall sales performance.

## File

- **Filename:** `Advertising Budget and Sales.csv`
- **Records:** 200 rows
- **Columns:** 5 (index + 3 features + 1 target)

## Features

| Column                    | Type  | Unit | Range       | Description                               |
| ------------------------- | ----- | ---- | ----------- | ----------------------------------------- |
| `TV Ad Budget ($)`        | Float | $    | 0.7 - 296.4 | Advertising budget spent on TV            |
| `Radio Ad Budget ($)`     | Float | $    | 0.3 - 49.6  | Advertising budget spent on Radio         |
| `Newspaper Ad Budget ($)` | Float | $    | 0.0 - 114.0 | Advertising budget spent on Newspapers    |
| `Sales ($)`               | Float | $    | 1.6 - 27.0  | Sales revenue generated (target variable) |

## Objective

- Understand the dataset and perform any necessary cleanup.
- Build **simple linear regression** models to predict sales with respect to a single feature.
- Build a **multiple linear regression** model using all advertising channels.
- Evaluate and compare models using metrics such as R², RMSE, and MAE.

## Source

- **Platform:** Kaggle
- **Link:** [Advertising Sales Dataset](https://www.kaggle.com/datasets/yasserh/advertising-sales-dataset/data)
