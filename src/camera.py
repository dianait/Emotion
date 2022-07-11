#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from jinko_games_message.srv import jinko_games_message, jinko_games_messageRequest, jinko_games_messageResponse
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import time

class TEAGame(object):

    def __init__(self, answer):
        self.start = 0
        self.end = 0
        self.timeElapsed = 0.0
        self.bridge_object = CvBridge()
        self.image_sub = None
        self.MIN_MATCH = 10
        self.threshold = 0.8
        self.detector = cv2.ORB_create(1000)
        self.FLANN_INDEX_LSH = 6
        self.index_params= dict(algorithm = self.FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
        self.search_params=dict(checks=32)
        self.matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        self.answer = answer
        self.result = False
    
    def getResult(self):
        return [self.result, round(self.timeElapsed,2)]


    def check(self):
        self.image_sub = rospy.Subscriber("/raspicam_node/image", Image, self.camera_callback)

    def camera_callback(self, data):
        self.start = time.time()
        try:
            frame = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")

            self.processing(frame)
        except CvBridgeError as e:
            print(e)
        cv2.imshow("Imagen capturada por el robot", frame)
        cv2.waitKey(1)

    def processing(self, frame):

            pattern = cv2.imread("/home/diana/catkin_ws/src/GAMES/jinko_games/src/img/" + self.answer + ".png")
            pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

            # Pasamos la imagen de la webcam a B/N
            frameBN = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #  Detecta los puntos que deben coincidir y a partir de ahi crea los descriptores
            kp1, desc1 = self.detector.detectAndCompute(frameBN, None)
            kp2, desc2 = self.detector.detectAndCompute(pattern, None)
                
            # Matching descriptor vectors with a FLANN based matcher
            matches = self.matcher.knnMatch(desc1, desc2, 2)
                
            #  Filtrado a partir del umbral
            good_matches = [m[0] for m in matches \
                            if len(m) == 2 and m[0].distance < m[1].distance *  self.threshold]

            print('good matches:%d/%d' %(len(good_matches),len(matches)))

            if len(good_matches) > self.MIN_MATCH:
                
                # Los puntos que hacen match (que estan en good_matches ) de las dos imagenes 
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])

                # Encuentra transformaciones de perspectiva entre dos planos.
                # A partir de los keypoints que deben coincidir tiene encuenta si se gira, se inclina... la tarjeta 
                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                accuracy=float(mask.sum()) / mask.size
                print(accuracy)
                print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))
                # if mask.sum() > self.MIN_MATCH:
                if accuracy > 0.8:
                    self.end = time.time()
                    self.timeElapsed = (self.end - self.start)
                    print("El tiempo trascurrido es de ", str(self.timeElapsed))
                    self.image_sub.unregister()
                    self.result = True
                    return 
            return 
                  
def checkAnswer(request):
    """
    Callback del servicio /jinko_games_service
    @param request: informacion recibida del servicio /jinko_games_service
    @type request: jinko_games_messageRequest
    @return: Si ha encontrado la coincidencia o no a partir de GameTEA.check()
    @rtype: jinko_games_messageResponse
    """
    # Recogemos la respuesta correcta de la llamada al servicio
    answer = request.answer

    # Ceamos objeto de la clase TEAGame
    game = TEAGame(answer)
    # help(game)

    # Generamos el mensaje respuesta del servicio
    response = jinko_games_messageResponse()

    # Nos sucribimos a la camara y comprobamos si ha acertado
    game.check()

    # Tiempo de espera
    time.sleep(7)

    # Guardamos el resultado
    respuesta = game.getResult()

    # Dejamos de suscribirnos a la camara de Jinkobot
    game.image_sub.unregister() 

    # Guardamos el resultado en la respuesta al servidos
    response.success = respuesta[0]
    response.timeElapsed = str(respuesta[1])
    print("resultado y tiempo", str(response.success) + " => " + str(response.timeElapsed))
    return response

def main():
    rospy.init_node('Ros20OpenCV_image_converter', anonymous=True)
    rospy.Service('/jinko_games_service', jinko_games_message, checkAnswer)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Fin del programa")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()