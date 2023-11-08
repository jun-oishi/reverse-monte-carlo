#ifndef __CLUSTERCONFIG_H__
#define __CLUSTERCONFIG_H__

#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <type_traits>

namespace ClusterConfig {

using string = std::string;
using complex = std::complex<double>;

const double COS60 = 0.5;
const double SIN60 = 0.86602540378443864676372317075294;

class ClusterConfig {
 public:
  ClusterConfig(int n, int lx, int ly);
  ~ClusterConfig();
  void move();
  void undo();
  void save(string dist);
  double *computeIq(double arr_q[], double arr_Iq[]);

 private:
  static const double MAX_DENSITY;           // 格子座標での最大密度
  static const double LATTICE_A, CLUSTER_R;  // [nm]
  static const double Q_MIN, Q_MAX;          // [nm^-1]
  static const int STEP_A[6], STEP_B[6];
  const int N_CLUSTERS;
  const int LX, LY;  // 格子座標での領域サイズ

  int *arr_a, *arr_b;     // 格子座標での座標
  double *arr_x, *arr_y;  // [nm] 座標
  int last_moved, last_a, last_b;

  int is_overlap(int a, int b, int i);
};

const double ClusterConfig::MAX_DENSITY = 1 / 12.0;
const double ClusterConfig::LATTICE_A = 0.321;
const double ClusterConfig::CLUSTER_R = 0.355;
const double ClusterConfig::Q_MIN = 4.5;
const double ClusterConfig::Q_MAX = 7.5;
const int ClusterConfig::STEP_A[6] = {1, 0, -1, -1, 0, 1};
const int ClusterConfig::STEP_B[6] = {0, 1, 1, 0, -1, -1};

/**
 * @brief ランダムな配置で初期化する
 * @param n クラスター数
 * @param lx 格子座標での領域サイズ
 * @param ly 格子座標での領域サイズ
 */
ClusterConfig::ClusterConfig(int n, int lx, int ly)
    : N_CLUSTERS(n), LX(lx), LY(ly) {
  arr_a = new int[N_CLUSTERS];
  arr_b = new int[N_CLUSTERS];
  arr_x = new double[N_CLUSTERS];
  arr_y = new double[N_CLUSTERS];
  for (int i = 0; i < n; i++) {
    arr_a[i] = rand() % LX;
    arr_b[i] = rand() % LY;
    if (is_overlap(arr_a[i], arr_b[i], i)) {
      i--;
      continue;
    }
    arr_x[i] = (arr_a[i] + arr_b[i] * COS60) * LATTICE_A;
    arr_y[i] = arr_b[i] * SIN60 * LATTICE_A;
  }
}

ClusterConfig::~ClusterConfig() {
  delete[] arr_a;
  delete[] arr_b;
  delete[] arr_x;
  delete[] arr_y;
}

/**
 * @brief クラスターを移動させる
 */
void ClusterConfig::move() {
  int i = rand() % N_CLUSTERS;
  int direction = rand() % 6;
  int a = (arr_a[i] + STEP_A[direction] + LX) % LX;
  int b = (arr_b[i] + STEP_B[direction] + LY) % LY;
  if (is_overlap(a, b, i)) {
    move();
  }
  last_moved = i;
  last_a = arr_a[i];
  last_b = arr_b[i];
  arr_a[i] = a;
  arr_b[i] = b;
  arr_x[i] = (a + b * COS60) * LATTICE_A;
  arr_y[i] = b * SIN60 * LATTICE_A;
}

/**
 * @brief クラスターの移動を取り消す
 */
void ClusterConfig::undo() {
  arr_a[last_moved] = last_a;
  arr_b[last_moved] = last_b;
  arr_x[last_moved] = (last_a + last_b * COS60) * LATTICE_A;
  arr_y[last_moved] = last_b * SIN60 * LATTICE_A;
}

/**
 * @brief クラスターの座標をファイルに保存する
 * @param dist ファイル名
 */
void ClusterConfig::save(string dist) {
  FILE *fp = fopen(dist.c_str(), "w");
  fprintf(fp, "n_clusters:%d, Lx:%d, Ly:%d\n", N_CLUSTERS, LX, LY);
  fprintf(fp, "a b x y\n");
  for (int i = 0; i < N_CLUSTERS; i++) {
    fprintf(fp, "%d %d %lf %lf\n", arr_a[i], arr_b[i], arr_x[i], arr_y[i]);
  }
  fclose(fp);
}

/**
 * @brief クラスターの座標から散乱強度を計算する
 * @param arr_q 波数配列
 * @param nq 波数配列の要素数
 * @return double* 散乱強度
 */
double *ClusterConfig::computeIq(double arr_q[], double arr_Iq[]) {
  int n = std::extent<decltype(arr_q, 0)>::value;  // <- 何故か0になる
  for (int i = 0; i < n; i++) {
    arr_Iq[i] = 0;
    double q = arr_q[i];
    for (int j = 0; j < 180; j++) {
      double theta = j * M_PI / 180;
      double qx = q * cos(theta);
      double qy = q * sin(theta);
      complex f = complex(0, 0);
      for (int k = 0; k < N_CLUSTERS; k++) {
        f += exp(complex(0, 1) * (qx * arr_x[k] + qy * arr_y[k]));
        std::cout << f << "->" << abs(f) << std::endl;
      }
      arr_Iq[i] += pow(abs(f), 2);
    }
  }
  // 規格化
  double sum = 0;
  for (int i = 0; i < n; i++) {
    if (Q_MIN < arr_q[i] && arr_q[i] < Q_MAX) {
      sum += arr_Iq[i];
    }
  }
  for (int i = 0; i < n; i++) {
    arr_Iq[i] /= sum;
  }
  return arr_Iq;
}

/**
 * @brief i番目までのクラスタと重なっているかどうかを判定する
 * @param a 格子座標
 * @param b 格子座標
 * @param i i番目までとの重なりを判定する
 */
int ClusterConfig::is_overlap(int a, int b, int i) {
  for (int j = 0; j < i; j++) {
    if (a == arr_a[j] && b == arr_b[j]) {
      return 1;
    }
  }
  return 0;
}

}  // namespace ClusterConfig

#endif