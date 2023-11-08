#include <iostream>

#include "ClusterConfig.hpp"

int main() {
  ClusterConfig::ClusterConfig cc(10, 10, 10);
  double q[4] = {4.5, 5.5, 6.5, 7.5};
  double i[4];
  cc.computeIq(q, i);
  for (int j = 0; j < 4; j++) {
    std::cout << i[j] << std::endl;
  }
  return 0;
}