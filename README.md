# AI Energy Hackathon
## Датасет
Мы собрали и разметили собственный датасет в Roboflow
## Решение
Для нашего решения мы обучили модель YOLOv8m.

Для инференса мы используем технологию SAHI.
Данная библиотека разделяет картинку на некоторое количество слайсов,
на которых делается предсказание с помощью YOLOv8m.
## Запуск решения

```
pip install requirements.txt
cd src
python main.py name_of_dir_imgs name_of_dir_save_files
```