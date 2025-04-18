# Clothing Reviews Sentiment Analysis

## Описание проекта
Проект **анализирует отзывы клиентов** на товары из категории одежды и **определяет их тональность** — позитивную или негативную.

## Используемые технологии
- **PyTorch**
- **Pandas**
- **NumPy**
- **Huggingface Transformers**
- **scikit-learn**
- **spaCy**
- **NLTK**

## Данные
Используется датасет **Womens Clothing E-Commerce Reviews**.

## Быстрый старт

Запустить проект можно через Google Colab или локально.

### Запуск через Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danilkos00/sentiment-analysis/blob/main/sentiment.ipynb)

### Локальный запуск

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/danilkos00/sentiment-analysis.git
   cd sentiment-analysis

2. Установите зависимости:
    ```bash
    pip install -r requirements.txt -qq

3. В ноутбуке автоматически загружаются предобученные параметры модели:
    Ссылка для загрузки весов: 
        ```bash
        https://drive.google.com/uc?id=1snKee0oLYAKJ-F5sTFZmh7qpEZrNU-Xg

## Структура проекта
    sentiment-analysis/
    ├── data/           # Датасет
    ├── src/           # Исходный код модели и вспомогательные функции
    ├── requirements.txt # Зависимости проекта
    ├── sentiment.ipynb     # Jupyter-ноутбук
    └── README.md      # Описание проекта

