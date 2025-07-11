import os
import cv2
import matplotlib.pyplot as plt
import numpy

from extractFeatures import FeatureExtractor

image_filename = 'cells.jpg'

image_filepath = os.path.join('assets', image_filename)

img = cv2.imread(image_filepath)


def extract_contours(img):
  if(img) is None:
    print('error: could not read the image')
  else:
    print('image loaded successfully')
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  image_preprocessed = cv2.GaussianBlur(gray_image, (5,5), 0)
  ret, binary_mask = cv2.threshold(image_preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  return contours


# def extract_features(contours, area=False, circularity=False, solidity=False, intensity=False, size=False, gray_image=''):
  calculated_areas = [cv2.contourArea(c) for c in contours]
  large_contours = []
  calculated_circularity = []
  calculated_solidity = []
  calculated_intensity = []
  average_area = numpy.mean(calculated_areas)

  if size == True:
    for i, area in enumerate(calculated_areas):
      if area >= average_area:
        large_contours.append(contours[i])

  if circularity == True:
    calculated_perimeters = [cv2.arcLength(c, True) for c in contours]
    for i, contour in enumerate(contours):
      cell_area = calculated_areas[i]
      cell_perimeter = calculated_perimeters[i]
      if cell_perimeter == 0:
        continue
      cell_circularity = (4 * numpy.pi * cell_area) / (cell_perimeter ** 2)
      calculated_circularity.append(cell_circularity)

  if solidity == True:
    for i, contour in enumerate(contours):
      cell_area = calculated_areas[i]

      convex_hull = cv2.convexHull(contour)

      hull_area = cv2.contourArea(convex_hull)

      if hull_area > 0:
        solidi = float(cell_area) / hull_area
      else:
        solidi = 0

      calculated_solidity.append(solidi)

  if intensity == True and gray_image != '':
    for i, contour in enumerate(contours):
      mask = numpy.zeros(gray_image.shape, dtype=numpy.uint8)
      cv2.drawContours(mask, [contour], -1, 255, -1)
      cell_intensity = cv2.mean(gray_image, mask=mask)[0]
      calculated_intensity.append(cell_intensity)

  result = {}
  if area:
    result['areas'] = calculated_areas
  if circularity:
    result['circularities'] = calculated_circularity
  if solidity:
    result['solidities'] = calculated_solidity
  if intensity:
    result['intensities'] = calculated_intensity
  if size:
    result['large_contours'] = large_contours

  return result


def display_large_cells(img):
  contours = extract_contours(img)

  features = FeatureExtractor(contours)

  large_contours = features._detect_large_cells()

  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  output_image = img_rgb.copy()

  cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
  cv2.drawContours(output_image, large_contours, -1, (0, 0, 255), 3)

  plt.imshow(output_image)
  plt.title("drawed contours")
  plt.axis('off')

  plt.tight_layout()
  plt.show()


display_large_cells(img)

