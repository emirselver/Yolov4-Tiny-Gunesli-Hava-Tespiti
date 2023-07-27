import cv2
import numpy as np
import time

try:
    net = cv2.dnn.readNet("model/yolov4-tiny-obj_best.weights", "model/yolov4-tiny.cfg")
    classes = []
    with open("model/obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()

    output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv2.VideoCapture(0)
    starting_time = time.time()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        frame_id += 1
        if not ret:
            break

        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layer_names)

        confidences = []

        sun_detected = False
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:
                    sun_detected = True
                    confidences.append(int(confidence * 100))
                    break

        if sun_detected:
            conf = confidences[0]
            con = f"Dogruluk Orani: %{conf}"
            cv2.putText(frame, "Hava gunesli", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, con, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Hava gunesli degil", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        a = elapsed_time + 1

        cv2.rectangle(frame, (520, 10), (630, 35), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(frame, (365, 40), (630, 65), (70, 70, 70), cv2.FILLED)

        cv2.putText(frame, "FPS: " + str((round(fps, 1))), (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Cikis icin Q tusuna basiniz.", (370, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Hava Durumu", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

except Exception as ex:
    print(ex)
