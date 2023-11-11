
#include <cmath>
#include <complex>
#include <eigen3/Eigen/Core>
#include <iostream>

template <typename T>
using Array = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using Vector = Eigen::Vector<T, Eigen::Dynamic>;

using Complex = std::complex<double>;
using dArray = Eigen::ArrayXXd;
using cArray = Eigen::ArrayXXcd;
using dVector = Eigen::VectorXd;
using cVector = Eigen::VectorXcd;

#define RAD(deg) (deg * M_PI / 180.0)

dVector linspace(double start, double end, int n) {
  dVector arr;
  arr.resize(n);
  double dx = (end - start) / (n - 1);
  for (int i = 0; i < n; i++) {
    arr(i) = start + dx * i;
  }
  return arr;
}

template <typename T>
T exp(T x) {
  T y;
  y.resize(x.size());
  for (int i = 0; i < x.size(); i++) {
    y(i) = exp(x(i));
  }
  return y;
}

dVector computeIq(const dVector &q, const dVector &x, const dVector &y) {
  int nq = q.size(), nc = x.size();
  dVector Iq;
  Iq.resize(nq);
  for (int i = 0; i < nq; i++) {
    double qi = q(i);
    dVector theta = RAD(linspace(0, 360, 360));
    dVector qx = qi * theta.array().cos();
    dVector qy = qi * theta.array().sin();
    cArray delta =
        (x * qx.transpose() + y * qy.transpose()).array();  // nc x 360
    cArray f = (Complex(0, 1) * delta.array()).exp();       // nc x 360
    // std::cout << f << std::endl;
    for (int j = 0; j < 360; j++) {
      Iq(i) += (f.col(j) * f.col(j).conjugate()).sum().real();
    }
    Iq(i) /= (360 * nc);
  }
  return Iq;
}