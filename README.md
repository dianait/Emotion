# Feature matching with FLANN

- Una vez pasadas las dos imágenes a B/N (webcam y plantilla) nos valemos del objeto **ORB** (Oriented FAST and Rotated BRIEF) para detectar los _keypoints_ y descriptores a comparar de ambas imágenes
  [OpenCV: ORB (Oriented FAST and Rotated BRIEF)](https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html)

  ```python
  detector = cv2.ORB_create(1000);
  kp1, desc1 = detector.detectAndCompute(frame, None)
  kp2, desc2 = detector.detectAndCompute(pattern, None)
  ```

  [FLANN](https://stackoverflow.com/tags/flann/info) (fast approximate nearest-neighbor searches) es una librería que utliza un [algoritmo de búsqueda a partir de los vecinos más cercanos](https://es.wikipedia.org/wiki/K_vecinos_m%C3%A1s_pr%C3%B3ximos) encontrando coincidencias entre dos imágenes dadas

  ```python
  # Configuración del detector FLANN
  FLANN_INDEX_LSH = 6
  index_params= dict(algorithm = FLANN_INDEX_LSH,
                     table_number = 6,
                     key_size = 12,
                     multi_probe_level = 1)
  search_params=dict(checks=32)

  # Creación del detector
  matcher = cv2.FlannBasedMatcher(index_params,search_params)

  # Búsqueda de coincidencias a partir de los descriptores hallados con ORB
  matches = matcher.knnMatch(desc1, desc2, 2)
  ```

  Los resultados se filtrarán en base a un umbral

  ```python
  umbral = 0.7
  good_matches = [m[0] for m in matches \
                              if len(m) == 2 and m[0].distance < m[1].distance * umbral]

  # Esta línea creo que equivaldría a esto,
  # pero por alguna razón funciona mucho mejor el código de arriba:
  good_matches = []
  for m,n in matches:
      if m.distance < umbral * n.distance and len(m) == 2:
          good_matches.append(m)

  ```

  - Guarda los puntos dónde han habido coincidencias en dos vectores, uno para la imagen fuente y otra para la plantilla

  ```python
  # Los puntos que hacen match (que estan en good_matches ) de las dos imagenes
   src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
   dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
  ```

  - A partir de esta información crea una nueva matriz y una máscara que contempla los cambios de perspectiva con la función _findHomography_

  ```python
  mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
  ```

  - Después de ésto comprueba que la suma de las coincidencias ( teniendo en cuanta las transformaciones de perspectiva ) sea mayor que el número mínimo de coincidencias que queramos tener ( especificado por nosotros )

  ```python
  if mask.sum() > self.MIN_MATCH:
       return True
  ```

  ...y VOILÀ!
