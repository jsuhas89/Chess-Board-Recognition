/** 
 * \file         overlay.h
 * \author       Alain Lehmann <lehmann at vision.ee.ethz.ch>
 * \version      $Id: overlay.h 338 2009-09-24 09:40:55Z lehmanal $
 * \date         2009-09-24 created
 */
#ifndef OVERLAY_H
#define OVERLAY_H

/** overlay some drawing to a grey-scale image.
 * \param img initial gray scale image
 * \param highlight thing to overlay on img
 * \param color     on which color chanel {0,1,2} --> {red, green, blue}
 */
CImg<float> overlay (CImg<float> img, CImg<float>& highlight, int color=0) {
  img.resize(-100,-100,-100,3);
  cimg_mapXY(img, x, y) { 
    img(x,y,0,color) = max(img(x,y,0,color),highlight(x,y)); 
  }
  return img;
}

#endif /* OVERLAY_H */
