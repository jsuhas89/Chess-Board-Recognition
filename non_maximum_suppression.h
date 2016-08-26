/** 
 * \file         non_maximum_suppression.h
 * \author       Alain Lehmann <lehmann at vision.ee.ethz.ch>
 * \version      $Id: non_maximum_suppression.h 338 2009-09-24 09:40:55Z lehmanal $
 * \date         2009-09-23 created
 */
#ifndef NON_MAXIMUM_SUPPRESSION_H
#define NON_MAXIMUM_SUPPRESSION_H

#include <vector>
#include <CImg.h>
using namespace cimg_library;
using namespace std;

/** a vector of pixel coordinates.
 * Usage:
 *    unsigned i;
 *    int x, y;
 *    TVectorOfPairs nonmax;
 *    nonmax.push_back (make_pair(x,y));   // adding new pixel coordinates:
 *    x = nonmax[i].first;                 // get x-coordinate of i-th pixel
 *    y = nonmax[i].second;                // get y-coordinate of i-th pixel
 */
typedef std::vector<std::pair<int,int> > TVectorOfPairs;

/** apply non-maximum suppression
 * \param input: some float image
 * \param nonmax: a list of (x,y)-tuple of maxima
 * \param thresh: ignore those with too small response
 * \param halfwidth: halfwidth of the neighbourhood size
 */
void non_maximum_suppression (CImg<float>& img, TVectorOfPairs& nonmax,
                              float thresh, int halfwidth) {
  nonmax.clear();
  for (int y=halfwidth; y<img.dimy()-halfwidth; y++) {
  for (int x=halfwidth; x<img.dimx()-halfwidth; x++) {
    float value = img(x,y);
    if (value<thresh) { continue; }

    bool ismax = true;
    for (int ny=y-halfwidth; ny<=y+halfwidth; ny++) {
    for (int nx=x-halfwidth; nx<=x+halfwidth; nx++) {
      ismax = ismax && (img(nx,ny)<=value);
    }}
    if (!ismax) continue;
    
    nonmax.push_back (make_pair(x,y));
  }}
}

#endif /* NON_MAXIMUM_SUPPRESSION_H */