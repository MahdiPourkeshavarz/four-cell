import os
import cv2
import matplotlib.pyplot as plt
import numpy
from functools import cached_property


class FeatureExtractor:
  def __init__(self, contours):
    self.contours = contours

  @cached_property
  def areas(self):
      return [cv2.contourArea(c) for c in self.contours]

  @cached_property
  def average_area(self):
      return numpy.mean(self.areas)

  @cached_property
  def perimeters(self):
      return [cv2.arcLength(c, True) for c in self.contours]

  def _detect_large_cells(self):
    large_contours = []

    for i, contour in enumerate(self.contours):
        if self.areas[i] >= self.average_area:
          large_contours.append(contour)
    return large_contours

  def _get_circularity(self):
        circularity_values = []
        for i, p in enumerate(self.perimeters):
            if p == 0:
                circularity_values.append(0)
                continue
            area = self.areas[i]
            circularity = (4 * numpy.pi * area) / (p ** 2)
            circularity_values.append(circularity)
        return circularity_values

  def _get_solidity_value(self):
    solidity_values = []

    for i, contour in enumerate(self.contours):
      cell_area = self.areas[i]

      convex_hull = cv2.convexHull(contour)

      hull_area = cv2.contourArea(convex_hull)

      if hull_area > 0:
        solidi = float(cell_area) / hull_area
      else:
        solidi = 0

      solidity_values.append(solidi)
    return solidity_values

  def _get_intensity_value(self, gray_image):
    intensity_values = []

    for i, contour in enumerate(self.contours):
      mask = numpy.zeros(gray_image.shape, dtype=numpy.uint8)
      cv2.drawContours(mask, [contour], -1, 255, -1)
      cell_intensity = cv2.mean(gray_image, mask=mask)[0]
      intensity_values.append(cell_intensity)
    return intensity_values


