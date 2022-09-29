# Graph Recognizer

## Данные
https://disk.yandex.ru/d/9Bvedhtg6aGyCQ
 
___

## Постановка задачи

Требуется по пришедшему на вход изображению планарного графа распознать его список смежности, результатом являются:
- список смежности графа с изображения на входе
- изображение графа, восстановленное по полученному списку смежности

___

### Объект распознавания
Планарный граф

### Свойства распознаваемого объекта
- Граф планарный, связный, невзвешенный (меток весов нет)
- Граф должен полностью помещаться на изображении (должен присутствовать зазор не менее 5 пикселей между границами изображения и границами примитивов графа)
- Вершина - примитив (окружность) с текстом метки внутри
- Цвет примитива вершины - черный
- Фон примитива вершины совпадает с фоном изображения
- Шрифт текста метки Arial, не курсив, допустимые символы: A-Z
- Метка всегда состоит из 2 символов из списка допустимых
- Баундбокс метки полностью помещается внутри примитива вершины, кратчайшее расстояние от границы баундобкса до границы примитива вершины не менее 5 пикселей
- Цвет метки
> Один из двух вариантов будет выбран в дальнейшем в зависимости от выбранного способа решения:
> 1. Черный
> 2. Красный
- Ребра могут быть двух видов
> Один из двух вариантов будет выбран в дальнейшем в зависимости от выбранного способа решения:
> 1. Одна или две параллельные сплошные прямые
> 2. Одна сплошная прямая двух различных цветов (синий/зеленый)
- Ребра не пересекаются
- Прямая, соответствующая ребру, соединяющему две вершины, доходит до границ примитивов этих вершин, но не пересекает их (зазора быть не должно)

______


### Входные данные:
- Путь до файла с изображением графа
- Путь для выгрузки изображения восстановленного по списку смежности графа

### Требования к изображению на входе:
- На изображении присутствует только один граф, посторонних объектов быть не должно
- Разрешение не менее 300x400
- Формат .png
- Объект на изображении резкий, не смазанный
- Цвет фона - белый
- Объект должен быть хорошо различим на фоне
- Отсутствие теней
- Ребра и вершины должны быть различимы, метки читаемы

### Допустимые искажения:
- Зашумление изображения

___

### Выходные данные
- Список смежности распознанного графа (стандартный вывод)
> dict[str: dict[str: dict[str: str]]] - словарь, в котором ключ является строковой меткой вершины, а значение – словарём, состоящим из пар (метка смежной вершины, словарь со значенями веса и типа ребра).
- Изображение восстановленного по списку смежности графа (сохранить по указанному пути или, если путь не был указан, вывести в отдельное окно)

### Ограничения
- Для корректного вывода значение параметра веса ребра принимается за 1

