#include <bits/stdc++.h>

#include <cmath>
#include <complex>
#include <eigen3/Eigen/Core>

#include "compute.hpp"

using namespace std;
using namespace Eigen;

int main() {
  dVector q = linspace(0, 10, 7);
  dVector x = linspace(0, 10, 4);
  dVector y = linspace(0, 10, 4);
  cout << computeIq(q, x, y) << endl;
}