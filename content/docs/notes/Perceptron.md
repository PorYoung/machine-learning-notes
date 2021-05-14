# Perceptron

{{< katex />}}
Perceptron is a kind of binary classification model.

<!--more-->

## Loss Function

$$
 L ( \omega, b ) =\sum_{i \in M}-y_{i} \frac{\left|\omega x_{i}+b\right|}{\|\omega\|}
$$

## Target

$$
 \min L ( \omega, b )
$$

## Method

### 随机梯度下降法

目标梯度

$$
 \nabla_{\omega} L=-\sum_{x_{i} \in M} y_{i} x_{i}
$$

$$
 \nabla_{b} L=-\sum_{x_{i} \in M} y_{i}
$$

选取初值$\omega_0$, $b_0$

迭代式

$$
 \omega \leftarrow \omega+\eta \cdot y_{i} x_{i}
$$

$$
 b \leftarrow b+\eta \cdot y_{i} x_{i}
$$

初值选取和迭代过程误分类点选取不同，结果不同。

## Convergence

存在$\gamma>0$，有

$$
 y_{i}\left ( \omega_{opt} \cdot x_{i}+b_{opt}\right ) \geqslant \gamma
$$

感知机学习算法收敛，$R=\max\_{1 \leqslant i \leqslant N}\left\|x\*{i}\right\|$，对误分类次数$k$，有

$$
 k \leq\left ( \frac{R}{\gamma}\right )^{2}
$$

## Form of duality

取初值：$\omega=\mathit{0}, \alpha=\mathit{0}, b=\mathit{0}$

学习到的$\omega$和$b$的形式为

$$
 \omega=\sum_{i=1}^{N} \alpha_{i} \mathcal{y}_{i} x_{i}
$$

$$
 b=\sum_{i=1}^{N} \alpha_{i} x_{i}
$$

误分类判据变为

$$
 y_{i}\left(\sum_{j=1}^{N} \alpha_{j} y_{i} x_{j} \cdot x_{i}+b\right) \leq 0
$$

迭代式

$$
 \alpha_{i} \leftarrow \alpha_{i}+\eta
$$

$$
 b_{i} \leftarrow b+\eta y_{i}
$$

$Gram$矩阵 $= \left[x_{i} \cdot x_{j}\right]_{N \times N}$

## Example

{{< details "Show an Example: fit iris data " >}}
{{% code file="Perceptron_test.md" md="true" %}}
{{< /details >}}

## Code

{{< details "Show Code" >}}
{{% code file="Perceptron.py" lang="py" %}}
{{< /details >}}
