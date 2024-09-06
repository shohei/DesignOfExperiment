import numpy as np
from pymoo.util.misc import stack
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 問題設定（自作関数）
class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]),
                        )
    def _evaluate(self, X, out, *args, **kwargs):
        f1 = X[:,0]**2 + X[:,1]**2
        f2 = (X[:,0]-1)**2 + X[:,1]**2
        g1 = 2*(X[:, 0]-0.1) * (X[:, 0]-0.9) / 0.18
        g2 = - 20*(X[:, 0]-0.4) * (X[:, 0]-0.6) / 4.8
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])

# パレート解の設定
# パレート解が既知の場合に使用（テスト用）
def func_pf(flatten=True, **kwargs):
    f1_a = np.linspace(0.1**2, 0.4**2, 100)
    f2_a = (np.sqrt(f1_a) - 1)**2
    f1_b = np.linspace(0.6**2, 0.9**2, 100)
    f2_b = (np.sqrt(f1_b) - 1)**2
    a, b = np.column_stack([f1_a, f2_a]), np.column_stack([f1_b, f2_b])
    return stack(a, b, flatten=flatten)
def func_ps(flatten=True, **kwargs):
    x1_a = np.linspace(0.1, 0.4, 50)
    x1_b = np.linspace(0.6, 0.9, 50)
    x2 = np.zeros(50)
    a, b = np.column_stack([x1_a, x2]), np.column_stack([x1_b, x2])
    return stack(a,b, flatten=flatten)
    
class MyTestProblem(MyProblem):
    def _calc_pareto_front(self, *args, **kwargs):
        return func_pf(**kwargs)
    def _calc_pareto_set(self, *args, **kwargs):
        return func_ps(**kwargs)

if __name__ == "__main__":
    # 問題の定義
    problem = MyTestProblem()
    
    # アルゴリズムの初期化（NSGA-IIを使用）
    algorithm = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(eta=20),
        eliminate_duplicates=True
    )
    
    # 終了条件（40世代）
    termination = get_termination("n_gen", 40)
    
    # 最適化の実行
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)
    
    # 結果の可視化
    ps = problem.pareto_set(use_cache=False, flatten=False)
    pf = problem.pareto_front(use_cache=False, flatten=False)
    
    # 入力変数空間
    # plot = Scatter(title = "Design Space", axis_labels="x")
    # plot.add(res.X, s=30, facecolors='none', edgecolors='r')
    # if ps is not None:
    #     plot.add(ps, plot_type="line", color="black", alpha=0.7)
    # plot.do()
    # plot.apply(lambda ax: ax.set_xlim(-0.5, 1.5))
    # plot.apply(lambda ax: ax.set_ylim(-2, 2))
    # plot.show()
    
    # 目的関数空間
    plot = Scatter(title = "Objective Space")
    plot.add(res.F)
    if pf is not None:
        plot.add(pf, plot_type="line", color="black", alpha=0.7)
    plot.show()
