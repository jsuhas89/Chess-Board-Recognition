/** 
 * \file         conversion.h
 * \author       Alain Lehmann <lehmann at vision.ee.ethz.ch>
 * \version      $Id: conversion.h 338 2009-09-24 09:40:55Z lehmanal $
 * \brief        <+A brief description+>
 * \par Description
 *   <+TODO a brief description+>
 * \date         2009-09-24 created
 */
#ifndef CONVERSION_H
#define CONVERSION_H

/** ensure 3-color channels -> RGB */
CImg<float> RGB(CImg<float> img) { return img.resize(-100,-100,-100,3); }
/** ensure 1-color channels -> GRAY */
CImg<float> GRAY(CImg<float> img) { 
  return img.get_norm_pointwise(1)/(float)img.dimv();
}

#endif /* CONVERSION_H */
