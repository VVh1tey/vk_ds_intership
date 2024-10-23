
# VK DS Internship

## Описание

Этот репозиторий предназначен для хранения решения тестового задания.
## Файловая структура

```plaintext
vk_ds_intership/
├── .venv (can’t be stored, must be created manually)
├── data/
│   ├── (some parquet/csv/xlsx/etc data files)
├── models/
│   ├── (saved models)
├── src/
│   ├── (.py files for my preprocessing module)
├── output/
│   ├── (prediction results)
├── .gitignore 
├── requirements.txt (required python libs)
├── eda+fe.ipynb (jupyter notebook with eda and feature generatibg)
├── modelling.ipynb (jupyter notebook with creating and fitting models)  
├──  models.zip (zipped best models) 
├── README.md
```

## Установка / использование

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/VVh1tey/vk_ds_intership.git
   ```

2. Перейдите в папку проекта:
   ```bash
   cd vk_ds_intership
   ```

3. Создайте виртуальное окружение:
   ```bash
   python -m venv .venv
   ```

4. Активируйте виртуальное окружение:
   - Для Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Для Mac/Linux:
     ```bash
     source .venv/bin/activate
     ```

5. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
5. Запустите скрипт:
   ```bash
   python main.py
   ```
