import cv2
import cv2.data

# Cargar el clasificador de rostro
rostro_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara")
    exit()

contador = 1  # Contador para numerar los cuerpos detectados

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = rostro_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    contador = 1  # Reinicia el contador en cada frame

    for (x, y, w, h) in rostros:
        # Dibujar rectángulo del rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Simular zona de medio cuerpo extendiendo debajo del rostro
        body_top = y
        body_bottom = y + int(h * 3)
        body_left = x - int(w * 0.5)
        body_right = x + int(w * 1.5)
        body_width = body_right - body_left
        body_height = body_bottom - body_top

        cv2.rectangle(frame, (body_left, body_top), (body_right, body_bottom), (255, 255, 0), 2)

        # Mostrar información del cuerpo en la terminal
        print(f'cuerpo {contador}: x = {body_left}, y = {body_top}, ancho = {body_width}, alto = {body_height}')
        contador += 1

    cv2.imshow('Deteccion de medio cuerpo', frame)

    if cv2.waitKey(1) != -1:  # Cierra con cualquier tecla
        break

cap.release()
cv2.destroyAllWindows()
