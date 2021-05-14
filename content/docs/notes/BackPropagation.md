# BackPropagation

{{< katex />}}

<!--more-->

## Single-hidden Layer Feedforward Neural Network

{{<columns>}}

![newworks](https://i.loli.net/2021/05/14/DkUCvadouE8mQW7.png)

<--->

|              | input                                       | output                                           |
| ------------ | ------------------------------------------- | ------------------------------------------------ |
| output layer | $\beta_j=\sum_{h=1}^{q} \omega_{h j} b_{h}$ | $\hat{y}_{j}^{k}=f\left(\beta_j-\theta_j\right)$ |
| hidden layer | $\alpha_h=\sum_{i=1}^{d} v_{i h} x_{i}$     | $b_{n}=f\left(\alpha_{i h}-\gamma_{h}\right)$    |
| input layer  | $x_i$                                       | .                                                |

{{</columns>}}

## LossFunction

累计误差$E_k$，目标$min E_k$

$$
 E_{k}=\frac{1}{2} \sum_{j=1}^{l}\left(\hat{y}_{j}^{k}-y_{j}\right)^{2}
$$

## Iterative Equations[^1]

隐层到输出层连接边权值变化

$$
 \Delta \omega_{h}=-\eta \frac{\partial E_{k}}{a \omega_{k j}}
$$

$$
 \begin{aligned}
 \frac{\partial E_{k}}{\partial \omega_{h j}} &=\frac{\alpha E_{k}}{\partial \hat{y}_{j}^{k}} \cdot \frac{\partial \hat{y}_{j}^{k}}{\partial \beta_{j}} \cdot \frac{\partial \beta_{j}}{\partial \omega_{h j}} \newline
 &=\left(\hat{y}_{j}^{k}-y_{j}^{k}\right) f^{\prime}\left(\beta_{j}-\theta_{j}\right) \cdot b_{h}
 \end{aligned}
$$

输出层阈值变化

$$
 \Delta \theta_{j}=-\eta \frac{\partial E_k}{\partial \theta_{j}}
$$

$$
 \begin{aligned} \frac{\partial E_k}{\partial \theta_{j}} &=\frac{\partial E_{k}}{\partial \hat{y}_{j}^{k}} \cdot \frac{\partial \hat{y}_{j}^{k}}{\partial \theta_j} \newline
 &=\left(\hat{y}_{j}^{k}-y_{j}^{k}\right) \cdot f^{\prime}\left(\beta_{i}-\theta_{j}\right) \cdot(-1)=g_{j}
 \end{aligned}
$$

输入层到隐层连接边权值变化

$$
 \Delta v_{i h}=-\eta \frac{\partial E k}{\partial V_{i h}}
$$

$$
\begin{aligned}
 \frac{\partial E_k}{\partial v_{i h}}&=\sum_{j=1}^{l} \frac{\alpha E_{k}}{\partial \hat{y}_{j}^{k}} \cdot \frac{\partial \hat{y}_{j}^{k}}{\partial \beta_{j}} \cdot \frac{\partial \beta_{j}}{\partial b_{h}} \cdot \frac{\partial b_{h}}{\partial \alpha_{h}} \cdot \frac{\partial \alpha_{h}}{\partial v_{i h}} \newline
 &=\sum_{j=1}^{k}\left(\hat{y}_{j}^{k}-y_{5}^{k}\right) \cdot f^{\prime}\left(\beta_{j}-\theta_{j}\right) \cdot \frac{\partial \beta_{j}}{\partial b_{j}} \cdot \frac{\partial b_{h}}{\partial \alpha_{h}} \cdot \frac{\partial \alpha_{h}}{\partial v_{i h}} \newline
 &=\left[\sum_{j=1}^{l}\left(\hat{y}_{j}^{k}-y_{j}^{k}\right) \cdot f^{\prime}\left(\beta_{j}-\theta_{j}\right) \cdot \omega_{h j}\right] \cdot \frac{\partial b_{h}}{\partial \alpha_{h}} \cdot \frac{\partial \alpha_{h}}{\partial v_{i h}}
 \end{aligned}
$$

{{<hint info >}}
通常可以根据经验公式 $m=log_2 ( n )$， ( $m$为隐层节点数，$n$为输入层节点数 ) 得到隐层应节点数。
{{</hint>}}

## Examples

{{< details "Show an Example: fit $0.5*(cos(x)+1)$" >}}

- 测试
- `max_iter=10000, error=0.0001, same_error_times=10`
- `iterated 10000/10000 times, error 0.2785599852590231.`

![result](https://i.loli.net/2021/05/14/JQSwWaP6RHD4Xn7.png)
{{< /details >}}

{{< details "Show an Example: fit $0.5*cos(x_1)*sin(x_2)$ " >}}
{{% code file="BP_test.md" md="true"%}}
{{< /details >}}

## Code

{{< details "Show Code" >}}
{{% code file="BP.py" lang="py"%}}
{{< /details >}}

[^1]: [南瓜书 PumpkinBook](https://datawhalechina.github.io/pumpkin-book/#/chapter5/chapter5)
