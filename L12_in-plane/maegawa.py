import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import os
import time
import argparse
import tqdm
import re

mpl.use("Agg")
COS60 = np.cos(np.pi / 3)
SIN60 = np.sin(np.pi / 3)


class LPSO_RMCSetting:
    LATTICE_A = 0.321  # for alpha-Mg [nm]
    CLUSTER_R = 0.355  # for L12 cluster [nm]
    CLUSTER_D_SQ = (CLUSTER_R * 2)**2  # [nm^2]
    Q_MIN = 4.5  # 規格化、評価に使う最小のq[nm^-1]
    Q_MAX = 7.5  # 規格化、評価に使う最大のq[nm^-1]
    CLUSTER_MAX_DENSITY = 1 / 12  # クラスタの最大密度
    A_STEP = (1, 0, -1, -1, 0, 1)  # 1ステップでのa軸方向の移動
    B_STEP = (0, 1, 1, 0, -1, -1)  # 1ステップでのb軸方向の移動
    KP = 100 # 残差が増える遷移の許容確率のパラメタ(大きいほど許容確率を下げる)

    def __init__(self, n_cluster, Lx, Ly):
        self.N_CLUSTER = n_cluster
        self.LX = Lx
        self.LY = Ly
        # overlapの判定に使う
        self.Lax = self.LX * self.LATTICE_A # a軸周期ずれのx幅
        self.Lbx = self.LY * self.LATTICE_A * COS60 # b軸周期ずれのx幅
        self.Lby = self.LY * self.LATTICE_A * SIN60 # b軸周期ずれのy幅
        if n_cluster > self.LX * self.LY * self.CLUSTER_MAX_DENSITY:
            raise ValueError("too many clusters")

        self.a, self.b = np.empty(n_cluster, dtype=int), np.empty(n_cluster, dtype=int)
        self.x, self.y = np.empty(n_cluster), np.empty(n_cluster)

        failure = True
        for i in range(10):
            try:
                self.config()
                failure = False
                break
            except RuntimeError:
                continue
        if failure:
            raise RuntimeError("random initialization failed")
        self.n_cycle = 0
        self.time_run = 0.0

    def config(self):
        """クラスタの初期配置を決定する"""
        bar = tqdm.tqdm(total=self.N_CLUSTER)
        bar.set_description("initializing")
        for i in range(0, self.N_CLUSTER):
            a, b = np.random.randint(0, self.LX), np.random.randint(0, self.LY)
            count = 0
            while self.is_overlap(a, b, i_fin=i) and count < self.LX * self.LY:
                count += 1
                a, b = np.random.randint(0, self.LX), np.random.randint(0, self.LY)
            if count == self.LX * self.LY:
                raise RuntimeError("random initialization failed")
            self.a[i], self.b[i] = a, b
            self.x[i], self.y[i] = self.ab2xy(a, b)
            bar.update(1)
        return

    def ab2xy(self, a, b):
        """格子座標系の座標a,bをxy直交座標系の座標に変換する"""
        x = (a + b * COS60) * self.LATTICE_A  # x座標[nm]
        y = b * SIN60 * self.LATTICE_A  # y座標[nm]
        return x, y

    def is_overlap(self, a:int, b:int, *, i_ini:int=0, i_fin:int=-1) -> bool:
        """添え字iからj-1までのクラスタと重なりがあるか判定する"""
        i_fin = self.N_CLUSTER if i_fin < 0 else i_fin
        if i_ini == i_fin:
            return False
        x, y = self.ab2xy(a, b)
        arr_x = self.x[i_ini:i_fin]
        arr_y = self.y[i_ini:i_fin]
        arr_x = np.concatenate([
            arr_x-self.Lax+self.Lbx, arr_x+self.Lbx, arr_x+self.Lax+self.Lbx,
            arr_x-self.Lax, arr_x, arr_x+self.Lax,
            arr_x-self.Lax-self.Lbx, arr_x-self.Lbx, arr_x+self.Lax-self.Lbx
        ])
        arr_y = np.concatenate([
            arr_y+self.Lby, arr_y+self.Lby, arr_y+self.Lby,
            arr_y, arr_y, arr_y,
            arr_y-self.Lby, arr_y-self.Lby, arr_y-self.Lby
        ])
        # 以下4行はあってもなくても結果は同じでn_cluster=48だと速度もほぼ同じ(n大だとあったほうが速い?)
        # x_filter = (-2*self.CLUSTER_R<=arr_x) * (arr_x < self.Lax+2*self.CLUSTER_R)
        # y_filter = (-2*self.CLUSTER_R<=arr_y) * (arr_y < self.Lby+2*self.CLUSTER_R)
        # arr_x = arr_x[x_filter*y_filter]
        # arr_y = arr_y[x_filter*y_filter]
        overlap = (arr_x-x)**2 + (arr_y-y)**2 < self.CLUSTER_D_SQ
        return np.any(overlap)

    def loadExpData(self, src):
        """実験データを読み込む"""
        data = np.loadtxt(src)
        self.EXP_Q = data[:, 0]
        self.N_POINTS = self.EXP_Q.shape[0]
        sum = np.sum(data[(self.Q_MIN < self.EXP_Q) & (self.EXP_Q < self.Q_MAX), 1])
        self.EXP_IQ = data[:, 1] / sum  # 所定範囲の和で規格化

    def loadConfig(self, src):
        """初期配置を読み込む"""
        data = np.loadtxt(src, skiprows=4, delimiter=",", dtype=int)
        if data.shape[0] != self.N_CLUSTER:
            raise ValueError("number of clusters is different")
        if data[:,0].max() >= self.LX or data[:,1].max() >= self.LY:
            raise ValueError("cluster position is out of range")
        self.a, self.b = data[:, 0], data[:, 1]
        self.x, self.y = self.ab2xy(self.a, self.b)

    def computeIq(self):
        """現在のクラスタ配置に基づいてI(q)を計算する"""
        n = 180  # クラスタの向きを平均するための分割数
        iq = np.empty_like(self.EXP_Q)

        # 各原子の位置ベクトル shape=(n_cluster)
        rv = self.x + 1j * self.y
        mq = self.EXP_Q  # shape=(n_points,)
        rotate = np.array(
            np.exp(1j * np.linspace(0, 2 * np.pi, n, endpoint=False))
        )  # shape=(n,)
        qv = np.outer(mq, rotate)  # shape=(n_points,n)
        # f[i,j,k]:i番目のクラスタのj番目のq、k番目の方向における複素振幅
        f = np.array(
            [np.exp(1j * (r * qv.conjugate()).real) for r in rv]
        )  # shape=(n_cluster,n_points,n)
        i = np.abs(np.sum(f, axis=0)) ** 2  # shape=(n_points,n)
        iq = np.mean(i, axis=1)  # shape=(n_points,)

        iq = iq / np.sum(iq[(self.Q_MIN < self.EXP_Q) & (self.EXP_Q < self.Q_MAX)])
        return iq

    def residual(self, iq):
        """計算されたI(q)と実験データとの差を計算する"""
        i_com = iq[(self.Q_MIN < self.EXP_Q) & (self.EXP_Q < self.Q_MAX)]
        i_exp = self.EXP_IQ[(self.Q_MIN < self.EXP_Q) & (self.EXP_Q < self.Q_MAX)]
        return np.sum((i_com - i_exp) ** 2)

    def move(self):
        """ランダムにクラスタを選んで移動する"""
        i = np.random.randint(0, self.N_CLUSTER)
        new_a, new_b = self.a[i], self.b[i]
        direction = np.random.randint(0, 6)
        new_a += self.A_STEP[direction]
        new_b += self.B_STEP[direction]
        new_a = new_a % self.LX
        new_b = new_b % self.LY

        if self.is_overlap(new_a, new_b, i_fin=i) or self.is_overlap(new_a, new_b, i_ini=i+1):
            # 重なる場合はやり直し
            self.move()
        else:
            # 戻すために記録
            self.last_moved, self.last_moved_before = i, (self.a[i], self.b[i])
            self.a[i], self.b[i] = new_a, new_b
            self.x[i], self.y[i] = self.ab2xy(new_a, new_b)
        return

    def undo(self):
        """最後に動かしたクラスタを元に戻す"""
        self.a[self.last_moved], self.b[self.last_moved] = self.last_moved_before
        self.x[self.last_moved], self.y[self.last_moved] = self.ab2xy(
            *self.last_moved_before
        )

    def run(self, n_cycle, log_interval=-1, saveConfig=False, saveDir=""):
        """シミュレーションを実行する
        log_intervalで指定した回数ごとにlogを取る
        最初と最後の値は必ず入る
        """
        t_ini = time.time()
        iq = self.computeIq()
        residual = self.residual(iq)
        if log_interval < 0:
            log_interval = n_cycle
        residual_log = np.empty(1 + n_cycle // log_interval)
        log_cycle = np.empty_like(residual_log, dtype=int)
        residual_log[0] = residual
        log_cycle[0] = 0

        bar = tqdm.tqdm(total=n_cycle)
        bar.set_description("optimizing")
        i = 0
        try:
            for i in range(1, n_cycle + 1):  # tqdmで進捗を表示
                self.move()
                new_iq = self.computeIq()
                new_residual = self.residual(new_iq)
                # new_residual->大で確率が小さくなるように
                p = np.exp(-self.KP * (new_residual - residual) / residual)
                if np.random.uniform(0, 1) < p:
                    iq = new_iq
                    residual = new_residual
                else:
                    self.undo()
                if i % log_interval == 0:
                    residual_log[i // log_interval] = residual
                    log_cycle[i // log_interval] = i
                    if saveConfig:
                        self.n_cycle = i
                        self.time_run = time.time() - t_ini
                        self.saveConfig(saveDir + f"/{i}cycle_config.dat")
                bar.update(1)
        except KeyboardInterrupt:
            print(f"interrupted at {i} cycle")
            self.undo()
            residual_log = residual_log[: i // log_interval + 1]
            log_cycle = log_cycle[: i // log_interval + 1]
            n_cycle = i-1

        # log_intervalによらず最後の値を入れる
        residual_log[-1] = residual
        log_cycle[-1] = n_cycle
        self.time_run = time.time() - t_ini
        self.n_cycle = n_cycle
        return log_cycle, residual_log

    def showScatter(self, ax: Axes, title=""):
        ax.scatter(self.x, self.y)
        vortex = self.ab2xy(
            np.array([0, 0, self.LX, self.LX, 0]), np.array([0, self.LY, self.LY, 0, 0])
        )
        ax.plot(vortex[0], vortex[1])
        if title:
            ax.set_title(title)
        return ax

    def saveConfig(self, dist, overwrite=False):
        if not overwrite and os.path.exists(dist):
            raise FileExistsError(f"{dist} already exists")
        header = (
            f"n_cluster: {self.N_CLUSTER}, Lx: {self.LX}, Ly: {self.LY}\n"
            + f"Q_MIN: {self.Q_MIN}, Q_MAX: {self.Q_MAX}, cluster_r: {self.CLUSTER_R}, lattice_a: {self.LATTICE_A}, cluster_max_density: {self.CLUSTER_MAX_DENSITY}\n"
            + f"{self.n_cycle} cycles run in {self.time_run:.6f} sec.\n"
            + "a_coord, b_coord"
        )
        np.savetxt(
            dist, np.array([self.a, self.b]).T, header=header, fmt="%d", delimiter=", "
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help="path to experimental data")
    parser.add_argument(
        "-n", "--n_cluster", type=int, default=20, help="number of clusters (default 48)"
    )
    parser.add_argument("-x", "--lx", type=int, default=24, help="x size of lattice")
    parser.add_argument("-y", "--ly", type=int, default=24, help="y size of lattice")
    parser.add_argument(
        "-c", "--n_cycle", type=int, default=300, help="number of cycles"
    )
    parser.add_argument(
        "-i", "--log_interval", type=int, default=1, help="log interval (default 1)"
    )
    parser.add_argument(
        "-d",
        "--detail_log",
        action="store_true",
        help="log configuration while running",
    )
    parser.add_argument(
        "-ow",
        "--overwrite",
        action="store_true",
        help="if set, overwrite existing files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/result_" + time.strftime("%Y%m%d%H%M"),
        help="output directory",
    )
    parser.add_argument(
        "-l", "--load", type=str, default="", help="load initial configuration"
    )
    parser.add_argument("-s", "--seed", type=int, default=1, help="random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)
    n_cluster, Lx, Ly = args.n_cluster, args.lx, args.ly
    n_cycle, log_interval = args.n_cycle, args.log_interval
    src, dist_dir = args.src, args.output
    overwrite, saveConfig = args.overwrite, args.detail_log
    initial_config = args.load

    t_ini = time.time()
    rmc = LPSO_RMCSetting(n_cluster, Lx, Ly)
    if initial_config:
        rmc.loadConfig(initial_config)
    rmc.loadExpData(src)
    q = rmc.EXP_Q
    iq_before = rmc.computeIq()

    os.makedirs(dist_dir, exist_ok=overwrite)
    rmc.saveConfig(dist_dir + "/initial_config.dat", overwrite=overwrite)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    rmc.showScatter(axes[0, 0], title="cluster arrangement (before)")

    log_cycle, residual_log = rmc.run(
        n_cycle, log_interval=log_interval, saveConfig=saveConfig, saveDir=dist_dir
    )
    iq_after = rmc.computeIq()

    print(f"{rmc.n_cycle} cycles run in {time.time() - t_ini:.6f} [s]")

    rmc.showScatter(axes[1, 0], title=f"cluster arrangement (after {rmc.n_cycle} cycles)")

    ax = axes[0, 1]
    ax.semilogy(log_cycle, residual_log)
    ax.set_title("residual history")
    ax.set_xlabel("cycle")
    ax.set_ylabel("residual")

    ax = axes[1, 1]
    ax.plot(q, rmc.EXP_IQ, label="exp.")
    ax.plot(q, iq_before, label="before")
    ax.plot(q, iq_after, label=f"after {rmc.n_cycle} cycles")
    ax.axvline(rmc.Q_MIN, color="gray", linestyle="--")
    ax.axvline(rmc.Q_MAX, color="gray", linestyle="--")
    ax.set_title("SAXS profile")
    ax.set_xlabel("q[nm^-1]")
    ax.legend()

    fig.tight_layout()

    fig.savefig(dist_dir + "/graph.png")
    rmc.saveConfig(dist_dir + "/final_config.dat", overwrite=overwrite)
    np.savetxt(
        dist_dir + "/iq.dat",
        np.array([q, rmc.EXP_IQ, iq_before, iq_after]).T,
        header="q[nm^-1], exp., before, after",
        fmt="%.6e",
        delimiter=", ",
    )
