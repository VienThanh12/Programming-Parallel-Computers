/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
#include<math.h>
#include <vector>
#include <algorithm>
#include <iostream>

void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {

  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
      int count = 0;
      std::vector<float> neighbors;
      for (int b = std::max(y - hy, 0); b < std::min(y + hy + 1, ny); b++) {
          for(int a = std::max(x - hx, 0); a < std::min(x + hx + 1, nx); a++) {
              neighbors.push_back(in[a + b * nx]);
              count++;
          }
      }
      std::nth_element(neighbors.begin(), neighbors.begin() + count / 2, neighbors.end());
      if(count % 2 == 0) {
          std::nth_element(neighbors.begin(), neighbors.begin() + count / 2 - 1, neighbors.end());
          out[x + y * nx] = (neighbors[count / 2] + neighbors[count / 2 - 1]) / 2.0f;
      } else {
          out[x + y * nx] = neighbors[count / 2];
      }
    }
  }
}
