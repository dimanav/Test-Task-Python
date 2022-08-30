import cv2
import configparser
from PIL import ImageColor

# инициализация
window_name = "Result"
face_cascade_db = cv2.CascadeClassifier('faces.xml') # обученая модель с лицами
bluring = True;

config = configparser.ConfigParser()  # создаём объекта парсера
config.read("config.ini")  # читаем конфиг

# считываем цвет и преобразовываем его в brg
color = config["Settings"]["color"]
color = ImageColor.getcolor(color, "RGB")
color = color[::-1]

#считываем источник изображения

source = config["Settings"]["source"]
if source.isdigit(): source = int(source)
cap = cv2.VideoCapture(source) 


# обрабатываем изображение
while True:
    success, frame = cap.read() # считываем изображение
    
    try:
        if success == False:
            raise Exception("Не удалось считать изображение")
    except Exception as e:
        print(e)
        
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # переводим в серый

    faces = face_cascade_db.detectMultiScale(frame_gray, 1.1, 5)  # меняем параметры для достижения BEST результата

    # рисуем обводку
    for (x, y, w, h) in faces:
        detect_face = cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=1)
        if bluring: detect_face[y:y + h, x:x + w] = cv2.medianBlur(frame[y:y + h, x:x + w], 35)  # размываем лица

    # полный экран
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # выводим изображение
    cv2.imshow(window_name, frame)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    if cv2.waitKey(1) & 0xff == ord('b'):
        bluring = not bluring
        
    
    
#закрываем приложение 
cap.release()
cv2.destroyAllWindows()
