import numpy as np
import cv2 as cv


image_filename = 'cells.jpg'

image_filepath = os.path.join('assets', image_filename)

img = cv.imread(image_filepath)

blank_image = np.zeros((500,500, 3), dtype='uint8') #creating blank images


blank_image[:] = 0,255,0 # changing the color

cv.rectangle(blank_image, (0,0), (250, 500), (0,255,0), thickness=cv.FILLED) # drawing rectangle

cv.putText(blank_image, 'hello', (225,225), cv.FONT_HERSHEY_TRIPLEX, 1.2, (0,255,0), 2) # writing text on image

canny = cv.Canny(img, 125, 175) # detecting any edges in the image

dialated = cv.dilate(canny, (3,3), iterations=1) # dialating the edges

resized = cv.resize(img, (img.shape[1]/2, img.shape[0]/2), interpolation=cv.INTER_NEAREST) # resizing image

croped = img[50:200, 200:400] # croping part of image

def translate(img,x, y):
  transMat = np.float32([[1,0,x],[0,1,y]])
  dimen = (img.shape[1], img.shape[0])
  return cv.warpAffine(img, transMat, dimen)


translated = translate(img, 100, 100) # translating or shifting image without changing dimension first by x axis and then by y axis

def rotate(img, angle, rotPoint=None):
  (height, width) = img.shape[:2]

  if(rotPoint is None):
    rotPoint = (width//2, height//2)

  rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
  dimen = (width, height)

  return cv.warpAffine(img, rotMat, dimen)

rotated = rotate(img, 45) # rotating image by 45 deg

b,g,r = cv.split(img) # splitting color channel

blue = cv.merge([b, blank_image, blank_image]) # creating a three channel image

medianBlur = cv.medianBlur(img, 2) # applying median blur which is good for salt and pepper noises

bilateral = cv.bilateralFilter(img, 5, 15, 15) # reducing noises better than median blur
