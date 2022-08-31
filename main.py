import cv2
import configparser
import time
from PIL import ImageColor

# инициализация
window_name = "Result"
face_cascade_db = cv2.CascadeClassifier('faces.xml') # обученая модель с лицами
bluring = True;

prev_frame_time = 0
new_frame_time = 0

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

font = cv2.FONT_HERSHEY_SIMPLEX


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
    
    # считаем fps
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    fps = int(fps)
    fps = str(fps)
 
    # выводим счётчик кадров
    cv2.putText(frame, f"FPS:{fps}", (7, 60), font, 2, (0, 0, 0), 3, cv2.LINE_AA)

    # выводим изображение
    cv2.imshow(window_name, frame)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    if cv2.waitKey(1) & 0xff == ord('b'):
        bluring = not bluring
        
    
    
#закрываем приложение 
cap.release()
cv2.destroyAllWindows()
