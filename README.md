Работа для вконтакте cv
Основная идея взять предобученную на видео модель, добавить туда свои слои для классификации на заставка/не заставка. В ходе работы с датасетом заметил, что многие видео размечены не правильно и некоторые файлы с видео битые. Для устранения проблемы видео у которых размечен
end<start убрал из выборки, а также убрал не открывающиеся или частично битые видео. Так же решил добавить k-means для небольшой классификации по длинне размеченной заставки т.к. утверждалось что длина будет короткой следовательно наиболее частотно представленная длинна должна иметь больший вес,
в качестве оптимизатора использовал ADAMW чтобы больше обучаться по важным признакам, а также т.к. я обучал на ноутбуке решил взять не сильно нагруженную предобученную модель из torhcvision  S3d т.к. она имеет не так много параметров и не так требовательна к ресурсам и выдает хорошую точность,
также решил заморозить ее веса чтобы не тратить время на обучение уже неплохо выделяемых признаков.
Т.К. делал во время сессии особо заморочиться времени не было. Также для ускорения обучения и основной проверки брал окрестность в 15 секунд вокруг центра разметки заставки для датасета.
К сожалению нормальные значение выбить по IOU не получилось либо взял слишком маленький датасет для трейна либо он все таки плохо размечен.
В model модель, в train обучение
