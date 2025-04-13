#Библиотека компьютерного зрения
import PIL.Image
import cv2
#Библиотека операционной системы
import os
# Библиотека распознавания лиц
import face_recognition
# Библиотека для кеширования данных
import pickle
# Библиотека для работы с фотоизображениями
from imutils import paths
# Библиотека для работы с окнами ос
import win32gui
# Библиотека для работы с массивами данных
import numpy

import PIL
import linecache
#глобальные перемнные

name_user = 'Test'
#размеры окна для imshow
width = 1280
height = 720
# Количество кадров для обучения
dataSet_kadr = 100
# Функция выбора источника видео
def video_source():
    print('1 веб камера')
    print('2 камера видеонаблюдения')
    print('3 камера телефона')
    print ('4 выход из  подпрограммы')
    num_source=input('Выберите источник видео : ')
    # используем конструкцию case
    match num_source:
        case '1':
            video_s=cv2.VideoCapture(0)
        case '2':
            video_s=cv2.VideoCapture("rtsp://wurs:5ttht5@192.168.1.60:554")
        case '3':
            video_s=cv2.VideoCapture("rtsp://192.168.1.186:8080/h264.sdp")
        case '4':
            return(4)
        case _:
            print('Не найдено устройство!')
            return(4)
    # возвращаемое значение функции
    return (video_s)

# функция создания пользователя
def create_User_folder(Name_User_Folder):
    # Получаем путь к папке где расположена программа
    path1 = os.path.dirname(os.path.abspath(__file__))
    # добавляем имя пользователя
    dirUser=path1+r'/dataSet/'+Name_User_Folder+r'/'
    # проверяем существует ли такая папка, если нет создаем
    os.makedirs(dirUser, exist_ok=True)    
    # возвращаем полный путь к папке пользователя
    return dirUser

# функция Демонстрация источников захвата видео 
def first_func():
    # используем глобальные переменные для размеров окна
    global width
    global height
    # флаг для запуска цикла
    run = True
    while run:
        # вызов функции выбора источники изображения
        video = video_source()
        # проверка возвращаемого значения, выход
        if video == 4:
            run=False
            break
        
        # Проверка, что открылось видео 
        if video.isOpened():
            print('ok')
        else:
            video.release()
            print ('Dont video')


        # Бесконечный цикл получения видеопотока
        while True:
            # читаем кадр из видеопотока
            hasFrame, im =video.read()
            # Проверяем удалось ли прочитать
            if hasFrame:
                # Изменяем размер изображения под размер окна
                resized_image = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)
                # отображаем видео
                cv2.imshow('Video', resized_image)
                # выводим окно на передний фон
                cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
                # cv2.setWindowProperty('Video', cv2.WINDOW_KEEPRATIO,1)
            else:
                # отладочное сообщение если кадр не прочитан
                print('Dont open video timeout')
                # освобождаем источник видео
                video.release()
            # Ожидаем нажатия клавиши для выхода  
            if cv2.waitKey(10) == ord ('q'):
                # освобождаем источник видеопотока
                video.release()
                # закрываем окно
                cv2.destroyAllWindows()
                break    
    # пустое возвращаемое значение 
    return()

# функция заведения пользователя в систему
def second_func():
    # используем глобальную переменную
    global name_user
    # получаем путь к папке с программой
    path = os.path.dirname(os.path.abspath(__file__))
    # указываем, что мы будем искать лица по примитивам Хаара
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # счётчик изображений
    i=0
    # расстояния от распознанного лица до рамки
    offset=50
    # запрашиваем имя пользователя
    name_user=input('Введите имя пользователя: ')
    # вызываем функцию создания папки пользователя
    dir_User = create_User_folder(name_user)
    # вызываем функцию выбора источника видеопотока
    video = video_source()
    # проверяем выбранный источник
    if video == 4:
        print('Не указан источник изображения')  
        return()
    # запускаем бесконечный цикл   
    while True:
        # читаем кадр из видеопотока
        hasFrame, im =video.read()
        # Проверяем, что считался кадр
        if hasFrame:
            print('Кадр прочитан')
        else:
            # отладочное сообщение если кадр не прочитан
            print('Dont open video timeout')
            # освобождаем источник видео
            video.release()

        # переводим всё в ч/б для скорости обработки
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        # настраиваем параметры распознавания и получаем лицо с камеры
        faces=detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(100, 100),flags=cv2.CASCADE_SCALE_IMAGE)
       
        # обрабатываем лица
        for(x,y,w,h) in faces:
            # выводим отладочную информацию о координатах обнаруженного лица
            print (x)
            print (y)
            print (w)
            print (h)
            # увеличиваем счетчик кадров
            i=i+1
            #  выводим количество полученных изображений
            print (i)
            # проверяем что края рамки не выходя за пределы окна
            if (x-offset>=0 and y-offset>=0):
                # записываем полученное изображение на диск в папку пользователя
                cv2.imwrite(dir_User+name_user +'.'+ str(i) + ".jpg", im[y-offset:y+h+offset,x-offset:x+w+offset])
            else:
                print('лицо не обнаружено')
                # рисуем квадрат вокруг лица
                cv2.rectangle(im,(x-offset,y-offset),(x+w+offset,y+h+offset),(225,0,0),2)
                resized_image = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)
                # показываем лицо
                cv2.imshow('color', resized_image)
                cv2.setWindowProperty('color', cv2.WND_PROP_TOPMOST, 1)
                # обязательное ожидание для раскадровки
                cv2.waitKey(10)
        # если лицо не найдено все равно показываем видеопоток 
        resized_image = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('color', resized_image)
        cv2.setWindowProperty('color', cv2.WND_PROP_TOPMOST, 1)
        # обязательное ожидание для раскадровки 
        cv2.waitKey(10)
        # количество необходимых кадров
        if i> dataSet_kadr:
            # освобождаем камеру
            video.release()
            # удалаяем все созданные окна
            cv2.destroyAllWindows()
            # останавливаем цикл
            break
        
    #выход из функции 
    return()


# функция обучение системы
def three_func():
    # используем глобальные переменные
    global name_user
    global width
    global height

    path = os.path.dirname(os.path.abspath(__file__))
    pathfortr = path+r'/trainer/'
    # создаём новый распознаватель лиц
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # указываем, что мы будем искать лица по примитивам Хаара
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # путь к датасету с фотографиями пользователя
     # указываем расположение папок с фотографиями пользователей
    path = path+r'/dataSet'
    # проверяем создан хоть один пользователь
    if len(os.listdir(path)) == 0:
        print('Пожалуйста создайте пользователя в системе')
    else:
        # вывод всех пользователей
        g = 1
        my_file = open(pathfortr+r'/users.txt', "w")
        for s in os.listdir(path):
            print (g, ' - ',s)
            print (g, ' - ',s, file=my_file)
            g += 1               
        my_file.close()        
        # выбор одного из пользователей
        # number_user=int(input('Введите номер пользователя из списка = '))
        # 
        # my_file = open(pathfortr+r'/users.txt', "r")
        # name_user = my_file.readlines()[number_user-1]
        # my_file.close()
        # 
    # получение пути к папки пользователя
    # path = path+'\\'+name_user+'\\'
    # вывод для отладки
    number_user = 0
    # списки картинок и подписей на старте пустые
    images = []
    labels = []
    print(path)
    # получаем путь к картинкам
    for user_fold in os.listdir(path):
        my_file = open(pathfortr+r'/users.txt', "r")
        add_full_path = path+r'/'+user_fold
        image_paths = [os.path.join(add_full_path, f) for f in os.listdir(add_full_path)]
        
        # Счетчик обработанных изображений
        i = 1
           
        name_user = my_file.readlines()[number_user]
        name_id = (name_user[0])
        name_user = name_user[6:-1]
        # перебираем все картинки в датасете 
        for image_path in image_paths:  
            # читаем картинку и сразу переводим в ч/б
            image_pil = PIL.Image.open(image_path).convert('L')
            # переводим картинку в numpy-массив
            image = numpy.array(image_pil, 'uint8')
            # получаем id пользователя из имени файла
            nbr = int(name_id)
            # определяем лицо на картинке
            print ('Обработано фотографий = ', i)
            i += 1
            faces = faceCascade.detectMultiScale(image)
            # если лицо найдено
            for (x, y, w, h) in faces:
                # добавляем его к списку картинок 
                images.append(image[y: y + h, x: x + w])
                # добавляем id пользователя в список подписей
                labels.append(nbr)
                # выводим текущую картинку на экран
                cv2.imshow(str(name_user), image[y: y + h, x: x + w])
                cv2.setWindowProperty(str(name_user), cv2.WND_PROP_TOPMOST, 1)
                # делаем паузу
                cv2.waitKey(10)
               
        cv2.destroyAllWindows()        
        number_user += 1
        my_file.close()   
    # Saving the trained faces and their respective ID's
    # in a model named as "trainer.yml".
    recognizer.train(images, numpy.array(labels))
    recognizer.save(pathfortr+"trainer.yml")

    
    
    return()

# функция Распознавание пользователя
def four_func():
    #  используем глобальные переменные
    global name_user
    global width
    global height
    # получаем путь к этому скрипту
    # if name_user == 'Test':
    #     print('Нет заведенных пользователей в систему')
    #     # если нет пользователей выход из функции
    #     return()
    path = os.path.dirname(os.path.abspath(__file__))
    # создаём новый распознаватель лиц
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # добавляем в него модель, которую мы обучили на прошлых этапах
    recognizer.read(path+r'/trainer/trainer.yml')
    # указываем, что мы будем искать лица по примитивам Хаара
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # получаем доступ к камере
    cam = video_source()
    # настраиваем шрифт для вывода подписей
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # запускаем цикл
    while True:
        # получаем видеопоток
        ret, im =cam.read()
        cv2.waitKey(10)
        # переводим его в ч/б
        if ret == False:
            print('Устройство недоступно повторите попытку')
            break
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        # определяем лица на видео
        faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) == 0:
            resized_image = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)     
            cv2.imshow('Face recognition',resized_image)
            # cv2.waitKey(10)
            cv2.setWindowProperty('Face recognition', cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(10) == ord ('q'):
               # освобождаем источник видеопотока
               cam.release()
               # закрываем окно
               cv2.destroyAllWindows()
               return() 
       
        # перебираем все найденные лица
        for(x,y,w,h) in faces:
            # получаем id пользователя
            nbr_predicted,coord = recognizer.predict(gray[y:y+h,x:x+w])
            # рисуем прямоугольник вокруг л≈ица
            cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
            # если мы знаем id пользователя
            print(nbr_predicted, '   = ',coord)
            if int(coord) < 60:
                my_file = open(path+r'/trainer/users.txt', "r")
                name_user = my_file.readlines()[nbr_predicted-1]
                name_user = name_user[6:-1]
                my_file.close()
            else:
                name_user = 'Unknown'
            
            # добавляем текст к рамке
            cv2.putText(im,name_user, (x,y+h),font, 1.1, (0,255,0))
            # выводим окно с изображением с камеры
            resized_image = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Face recognition',resized_image)
            cv2.setWindowProperty("Face recognition", cv2.WND_PROP_TOPMOST, 1)
            # делаем паузу
            if cv2.waitKey(10) == ord ('q'):
                # освобождаем источник видеопотока
                cam.release()
                # закрываем окно
                cv2.destroyAllWindows()
                return()   
               
    #     resized_image = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)     
    #     cv2.imshow('Not recognition',resized_image)
    #     cv2.waitKey(10)
    #     cam.release()
    # # закрываем окно
    # cv2.destroyAllWindows()
    #  выход из функции
    return()

#  основной цикл и начало программы
while True:
    # вывод пунктов меню
    print ('Основное меню консольного приложения:')
    print('1 Демонстрация источников захвата видео')
    print('2 Заведение пользователя в систему')
    print('3 Обучение системы')
    print('4 Распознавание пользователя')
    print('5 Выход из приложения')


    menu=input ('Выберите пункт меню: ')
    # конструкция case для списка условий
    match menu:
        case '1':
            first_func()
        case '2':
            second_func()
        case '3':
            three_func()
        case '4':
            four_func()
        case '5':
            exit()
        case _:
            print('Неправильный ввод!')

#  конец программы