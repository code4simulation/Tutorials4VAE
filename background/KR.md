# Variational Autoencoder (VAE) 분석 및 수학적 배경

이 문서는 Variational Autoencoders (VAE)에 대한 포괄적인 분석을 제공하며, 수학적 기초, PyTorch를 사용한 구현 세부 사항, 그리고 MNIST 데이터셋에 대한 결과 분석을 다룹니다.

## 1. VAE 소개

VAE는 딥러닝과 확률 모델을 결합한 생성 모델이다. VAE는 입력 데이터를 잠재 공간(latent space)의 **확률 분포**로 매핑한다. 이를 통해 VAE는 잠재 공간에서 샘플링을 수행하여 새로운 데이터를 생성할 수 있다.

## 2. 수학적 배경

이 섹션에서는 VAE를 이해하기 위해 필요한 수학적 개념들을 기초부터 순서대로 다룹니다.

### 2.1 베이즈 정리 (Bayes' Theorem)
베이즈 정리는 사전 지식을 바탕으로 사건의 확률을 갱신하는 방법을 설명한다. 일반적인 형태는 다음과 같다:

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

VAE의 관점에서 변수들은 다음과 같이 대응된다:
*   $A \rightarrow z$ (잠재 변수)
*   $B \rightarrow x$ (관측 데이터)

즉, VAE에 맞춰 베이즈 정리를 재구성하면 다음과 같다:

$$
p(z|x) = \frac{p(x|z)p(z)}{p(x)}
$$

*   **$p(z|x)$ (사후 확률, Posterior)**:
    데이터 $x$가 관찰되었을 때, 해당 데이터가 어떤 잠재 변수 $z$에서 기인했을지에 대한 확률이다. 복잡한 이미지 $x$로부터 이미지를 생성했을 법한 실제 $z$는 무엇인지 추론하는 **인코더(Encoder)** 과정에 해당한다. 생성 모델링의 궁극적인 계산 목표이나, 분모인 $p(x)$의 계산 난이도로 인해 직접적인 유도가 어렵다.

*   **$p(x|z)$ (우도, Likelihood)**:
    특정 잠재 변수 $z$로부터 관측 데이터 $x$가 생성될 확률이다. 즉, 추상적인 개념 $z$가 주어졌을 때, 이것이 실제 이미지 $x$로 어떻게 구체화되는가?"를 나타낸다. VAE에서는 이를 **디코더(Decoder)** 신경망으로 모델링하며, 모델 학습이 진행됨에 따라 이 확률값은 최대화된다.

*   **$p(z)$ (사전 확률, Prior)**:
    데이터 $x$를 관찰하기 전, 잠재 변수 $z$가 가질 것으로 가정하는 분포이다. VAE에서는 통상적으로 잠재 공간의 각 차원이 독립적이고 표준 정규 분포를 따른다고 가정한다 ($p(z) \sim \mathcal{N}(0, I)$). 이는 잠재 공간을 규칙적이고 연속적으로 유지하려는 **'구조적 제약(Regularization)'**으로 작용한다.

*   **$p(x)$ (증거, Evidence)**:
    관측된 데이터 $x$가 발생할 전체 확률(**Marginal Likelihood**)이다. 이는 모든 가능한 잠재 변수 $z$에 대해 우도와 사전 확률의 곱을 적분한 값과 같다 ($p(x) = \int p(x|z)p(z)dz$). 고차원 잠재 공간에서는 모든 $z$에 대한 적분이 불가능(**Intractable**)하므로, VAE는 이 값을 직접 계산하는 대신 **ELBO(Evidence Lower Bound)**를 최대화하는 우회적인 방법을 채택한다.

### 2.2 최대 우도 추정 (Maximum Likelihood Estimation)
최대 우도 추정(MLE)은 통계학에서 모델의 파라미터를 추정하는 직관적이고 강력한 방법론이다. 핵심 아이디어는 **현재 관측된 데이터가 발생할 확률이 가장 높도록 모델의 파라미터를 조정하는 것**이다. 생성 모델의 궁극적인 목표는 데이터셋(예: MNIST)의 분포 $p_{data}(x)$를 모델 $p_\theta(x)$가 근사하는 것이다. 만약 모델이 데이터 분포를 완벽하게 학습한다면, 실제 데이터와 구별 불가능한 새로운 샘플을 생성할 수 있다.

수식으로 표현하면, 관측 데이터 $x$에 대해 $p_\theta(x)$를 최대화하는 파라미터 $\theta$를 찾는 최적화 문제가 된다:

$$
\theta^* = \text{argmax}_\theta \sum_{i=1}^N \log p_\theta(x^{(i)})
$$

여기서 $p(x)$를 베이즈 정리의 '증거(Evidence)'가 아닌 **'우도(Likelihood)'**라고 칭하는 이유는 관점의 차이 때문이다. 베이즈 정리에서는 파라미터가 고정된 상태에서 관측값의 확률을 논하지만, MLE에서는 데이터 $x$를 고정하고 파라미터 $\theta$를 변화시키며 "해당 파라미터가 데이터를 설명하기에 얼마나 적합한가(Likely)"를 평가하기 때문이다.

또한, 단순 확률 $p_\theta(x)$ 대신 **로그 확률(Log-Likelihood)**을 최대화하는 방식을 취한다. 이는 두 가지 실용적 이유에 기인한다. 첫째, 확률값($0 \le p \le 1$)의 연속적인 곱셈은 0으로 수렴하는 언더플로우(Underflow)를 유발하나, 로그를 취하면 곱셈이 덧셈으로 변환되어 **수치적 안정성**을 확보할 수 있다. 둘째, 덧셈 연산의 미분은 곱셈 연산보다 현저히 용이하여 **계산 효율성**이 증대된다. 로그 함수는 단조 증가 함수이므로, 이를 최대화하는 것은 원본 우도를 최대화하는 것과 수학적으로 동치이다.

이 값이 최대가 될 때, 모델 분포는 실제 데이터 분포와 가장 유사해진다. 이는 수학적으로 실제 분포와 모델 분포 사이의 KL Divergence를 최소화하는 것과 동일한 의미를 갖는다.

하지만 **VAE와 같은 잠재 변수 모델(Latent Variable Model)에서는 난관이 존재한다.**
$p_\theta(x)$를 계산하기 위해서는 모든 가능한 잠재 변수 $z$에 대한 적분($p_\theta(x) = \int p_\theta(x|z)p(z)dz$)이 수행되어야 하나, 이는 계산 불가능(Intractable)하다. 따라서 우도를 직접 최대화하는 대신, 우도의 **하한(Lower Bound, ELBO)**을 최대화하는 우회적 방법을 사용한다.

### 2.3 쿨백-라이블러 발산 (Kullback-Leibler Divergence, KLD)
두 확률 분포 $q(x)$와 $p(x)$의 차이를 측정하는 비대칭적인 지표이다. VAE에서는 근사 분포 $q$가 실제 분포 $p$와 얼마나 유사한지를 정량화하는 데 사용된다.

**정의**

$$
D_{KL}(q \parallel p) = \int q(x) \log \frac{q(x)}{p(x)} dx = \mathbb{E}_{q} \left[ \log \frac{q(x)}{p(x)} \right]
$$

(여기서 적분 $\int q(x) (...) dx$는 확률 분포 $q(x)$에 대한 기댓값 $\mathbb{E}_q[...] $와 동일합니다)

**젠센 부등식 (Jensen's Inequality)**

젠센 부등식은 볼록 함수(convex function)에 대해 기댓값의 함수값이 함수값의 기댓값보다 작거나 같음을 나타내는 부등식이다.

*   함수 $f(x)$ 가 **볼록(convex)** 할 때 : $\mathbb{E}[f(x)] \ge f(\mathbb{E}[x])$
*   함수 $f(x)$ 가 **오목(concave)** 할 때 : $\mathbb{E}[f(x)] \le f(\mathbb{E}[x])$

본 유도 과정에서는 $-\log(x)$ 함수를 사용한다. 로그 함수 자체는 오목 함수이나, 음의 부호가 붙은 $-\log x$는 아래로 볼록한 **볼록 함수**이므로 젠센 부등식을 적용할 수 있다.

**비음수성(Non-negativity) 증명**
$D_{KL}$은 항상 0 이상의 값을 가진다 (Gibbs' inequality). 증명 과정은 다음과 같다:

$$
\begin{aligned}
D_{KL}(q \parallel p) &= \mathbb{E}_q \left[ -\log \frac{p(x)}{q(x)} \right] \\
&\ge -\log \left( \mathbb{E}_q \left[ \frac{p(x)}{q(x)} \right] \right) \quad (\text{Jensen Inequality}) \\
&= -\log \left( \int q(x) \frac{p(x)}{q(x)} dx \right) \\
&= -\log \left( \int p(x) dx \right) \\
&= -\log(1) = 0
\end{aligned}
$$

위 유도 과정의 마지막 단계에서 $\int p(x) dx = 1$이 되는 것은 확률 밀도 함수(Probability Density Function, PDF)가 갖는 핵심적인 성질인 **정규화 조건(Normalization Condition)**에 기인합니다. 확률은 표본 공간 내에서 발생 가능한 모든 사건의 상대적 빈도를 0과 1 사이의 값으로 정의한 것이며, "전체 표본 공간 내에서 어떤 사건이든 하나는 반드시 일어난다"는 공리에 따라 모든 가능한 결과에 대한 확률의 총합(또는 적분)은 반드시 1이 되어야 합니다. 따라서 $p(x)$가 유효한 확률 분포를 따른다면, 그 형태와 관계없이 전 구간에 대한 적분값은 항상 1로 수렴하게 됩니다.

### 2.4 생성 과정 (The Generative Process)
VAE는 데이터 $x$가 잠재 변수 $z$로부터 생성된다고 가정하는 잠재 변수 모델이다:
1.  **Prior**: $z \sim p_\theta(z)$ (통상적으로 표준 정규분포 가정)
2.  **Likelihood**: $x \sim p_\theta(x|z)$ (디코더 신경망을 통해 모델링)

사후 확률 $p_\theta(z|x)$를 계산하는 것이 목표이나, $p(x)$의 계산 불가능성(Intractability)으로 인해 직접적인 추론이 불가능하다.

### 2.5 변분 추론 (Variational Inference)
계산 불가능한 사후 확률 $p_\theta(z|x)$를 근사하기 위해, 다루기 쉬운 분포 $q_\phi(z|x)$ (인코더)를 도입한다. 변분 추론의 목표는 $q_\phi$와 $p_\theta$ 사이의 KL Divergence, 즉 $D_{KL}(q_\phi \parallel p_\theta)$를 최소화하는 파라미터 $\phi$를 찾는 것이다.

### 2.6 Evidence Lower Bound (ELBO) 상세 유도
로그 우도 $\log p_\theta(x)$를 최대화하는 문제는 ELBO를 최대화하는 문제로 치환될 수 있다. 유도 과정은 다음과 같다.

$$
\begin{aligned}
\log p_\theta(x) &= \log \int p_\theta(x, z) dz \\
&= \log \int p_\theta(x, z) \frac{q_\phi(z|x)}{q_\phi(z|x)} dz \\
&= \log \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \frac{p_\theta(x, z)}{q_\phi(z|x)} \right] \quad (\text{적분을 기댓값으로 변환}) \\
&\ge \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \right] \quad (\text{Jensen Inequality}) \\
&= \text{ELBO}
\end{aligned}
$$

**다른 방식의 유도 (KL Divergence와의 관계):**

$$
\begin{aligned}
\log p_\theta(x) &= \int q_\phi(z|x) \log p_\theta(x) dz \\
&= \int q_\phi(z|x) \log \left( \frac{p_\theta(x, z)}{p_\theta(z|x)} \cdot \frac{q_\phi(z|x)}{q_\phi(z|x)} \right) dz \\
&= \underbrace{\int q_\phi \log \frac{p_\theta(x, z)}{q_\phi} dz}_{\text{ELBO}} + \underbrace{\int q_\phi \log \frac{q_\phi}{p_\theta(z|x)} dz}_{D_{KL}(q_\phi || p_\theta(z|x))}
\end{aligned}
$$

**최종 ELBO 식**

$$
\begin{aligned}
\text{ELBO} &= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z) + \log p(z) - \log q_\phi(z|x)] \\
&= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \parallel p(z))
\end{aligned}
$$

*   첫 번째 항: **Reconstruction Error** (데이터 복원 오차)
*   두 번째 항: **Regularization** (근사 분포 $q_\phi$와 사전 분포 $p(z)$ 간의 차이)

### 2.7 재파라미터화 트릭 (The Reparameterization Trick)

**샘플링의 미분 불가능성**

VAE의 인코더는 잠재 변수 $z$의 분포 파라미터인 평균 $\mu$와 분산 $\sigma^2$를 출력하고, 이 분포 $\mathcal{N}(\mu, \sigma^2)$에서 $z$를 샘플링하여 디코더에 전달한다.
문제는 **"샘플링(Sampling)"** 과정이 확률적(Stochastic)이어서 미분이 불가능하다는 점이다. 무작위로 추출된 노드 $z$를 통해서는 오차 역전파(Backpropagation)가 인코더의 파라미터 $\phi$로 전달될 수 없으며, 결과적으로 체인 룰(Chain Rule)이 단절된다.

**해결책: 무작위성의 분리**

재파라미터화 트릭의 핵심은 **$z$를 결정론적(Deterministic) 부분과 확률적(Stochastic) 부분으로 분리하는 것**이다.
$z$를 분포에서 직접 샘플링하는 대신, 외부의 보조 노이즈 변수 $\epsilon$을 도입하여 $z$를 다음과 같이 재정의한다:

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

여기서:
*   $\mu, \sigma$: 인코더(신경망)의 출력 (결정론적, $\phi$에 대해 미분 가능)
*   $\epsilon$: 표준 정규분포에서 샘플링한 노이즈 (상수 취급, 미분 불필요)
*   $\odot$: 요소별 곱 (Element-wise product)

이제 $z$는 $\mu$와 $\sigma$에 대한 **함수**로 표현된다. 따라서 $z$를 $\mu$와 $\sigma$로 미분하는 것이 가능해지며($\frac{\partial z}{\partial \mu}=1, \frac{\partial z}{\partial \sigma}=\epsilon$), 역전파 알고리즘이 인코더까지 원활하게 수행될 수 있다. 이는 VAE의 End-to-End 학습을 가능케 하는 필수적인 기법이다.

### 2.8 조건부 VAE (Conditional VAE, CVAE)

기존 VAE는 생성 능력이 뛰어나지만, **생성 대상을 임의로 제어할 수 없다**는 한계가 존재한다. 잠재 공간에서의 무작위 샘플링으로 인해 특정 숫자(예: 7)의 생성을 보장할 수 없기 때문이다. 이를 보완하기 위해 제안된 모델이 **Conditional VAE (CVAE)**이다.

**핵심 아이디어: 조건(Condition) $c$의 도입**

CVAE는 인코더와 디코더의 입력에 **조건 정보 $c$ (예: 클래스 레이블)**를 추가한다. 이를 통해 모델은 주어진 조건 하에서의 데이터 생성 분포를 학습하게 된다.

**수식적 모델링**

모든 확률 분포는 조건 $c$에 종속되는 형태로 변경된다.

1.  **인코더 (Encoder)**: $q_\phi(z|x, c)$
    *   입력: 이미지 $x$와 레이블 $c$의 결합 (Concatenation)
    *   출력: 잠재 변수 $z$의 분포 파라미터 ($\mu, \sigma$)
    *   의미: "조건 $c$(숫자 7)를 만족하는 이미지 $x$가 잠재 공간의 어떤 좌표 $z$에 위치하는가?"

2.  **디코더 (Decoder)**: $p_\theta(x|z, c)$
    *   입력: 잠재 변수 $z$와 레이블 $c$의 결합
    *   출력: 이미지 $x$
    *   의미: "잠재 변수 $z$와 조건 $c$(숫자 7)를 결합하여 이미지 $x$를 생성하라."

3.  **목적 함수 (CVAE Loss)**:
    ELBO 수식에 조건 $c$가 추가된다.
    $$
    \log p(x|c) \ge \mathbb{E}_{q(z|x,c)}[\log p(x|z,c)] - D_{KL}(q(z|x,c) || p(z|c))
    $$
    *   통상적으로 사전 확률 $p(z|c)$는 조건 $c$와 무관하게 표준 정규 분포 $\mathcal{N}(0, I)$로 가정한다. 이는 클래스에 관계없이 잠재 공간의 위상적 구조를 일정하게 유지하기 위함이다.

**구조적 구현 (Implementation Detail)**

실제 구현 시에는 주로 **One-Hot Encoding**된 레이블 벡터를 이미지 또는 잠재 변수와 **결합(Concatenate)**하는 방식을 사용한다.

*   **인코더 입력**: `[Batch, 784]` (이미지) + `[Batch, 10]` (레이블) $\rightarrow$ `[Batch, 794]`
*   **디코더 입력**: `[Batch, 2]` (잠재 변수) + `[Batch, 10]` (레이블) $\rightarrow$ `[Batch, 12]`

이러한 구조적 확장을 통해 CVAE는 특정 클래스의 데이터를 의도적으로 생성할 수 있는 제어권(Controllability)을 확보한다.

## 3. PyTorch 구현 및 분석

본 절에서는 PyTorch 프레임워크를 활용하여 VAE와 CVAE를 구현하고 MNIST 데이터셋에 대해 학습한 내용을 다룬다. `vae_mnist.py`에 구현된 주요 모델 구성 요소에 대한 상세 분석은 다음과 같다.

### 3.1 모델 아키텍처 (Model Architecture)

#### 인코더 (Encoder)
인코더는 입력 이미지를 잠재 공간의 분포 파라미터인 평균($\mu$)과 로그 분산($\log \sigma^2$)으로 매핑하는 역할을 수행한다.
*   **VAE**: $28 \times 28$ 크기의 이미지를 평탄화(Flatten)하여 입력으로 사용한다.
*   **CVAE**: 이미지 벡터와 One-Hot 인코딩된 레이블 벡터를 **결합(Concatenate)**하여 입력으로 사용한다.

```python
# CVAE Encoder Implementation
class CVAE(nn.Module):
    def __init__(self, latent_dim: int = 2, num_classes: int = 10):
        super(CVAE, self).__init__()
        # 입력 차원: 이미지(784) + 클래스 레이블(10)
        self.fc1 = nn.Linear(28 * 28 + num_classes, 400)
        self.fc2_mu = nn.Linear(400, latent_dim)
        self.fc2_logvar = nn.Linear(400, latent_dim)

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        # 이미지(x)와 레이블(c)을 결합
        inputs = torch.cat([x, c], 1)
        h1 = F.relu(self.fc1(inputs))
        return self.fc2_mu(h1), self.fc2_logvar(h1)
```

#### 재파라미터화 (Reparameterization Trick)
역전파(Backpropagation)가 가능하도록 무작위성을 분리하는 핵심 메커니즘이다. 로그 분산(`logvar`)을 사용하는 이유는 분산이 항상 양수여야 한다는 수학적 제약을 자연스럽게 만족시키기 위함이다 ($\sigma = e^{0.5 \times \log\sigma^2}$).

```python
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)  # 표준편차 복원
        eps = torch.randn_like(std)    # 표준정규분포에서 노이즈 샘플링
        return mu + eps * std          # z = mu + epsilon * sigma
```

#### 디코더 (Decoder)
잠재 변수 $z$로부터 원본 이미지 공간으로의 복원을 수행한다.
*   **CVAE**: 인코더와 동일하게 잠재 변수 $z$와 조건 레이블 $c$를 결합하여 입력받는다. 이를 통해 모델은 조건 $c$에 부합하는 이미지를 생성하도록 유도된다.

```python
    def decode(self, z: torch.Tensor, c: torch.Tensor):
        # 잠재 변수(z)와 레이블(c)을 결합
        inputs = torch.cat([z, c], 1)
        h3 = F.relu(self.fc3(inputs))
        return torch.sigmoid(self.fc4(h3)) # 픽셀 값을 0~1 사이 확률로 출력
```

### 3.2 손실 함수 (Loss Function)와 KL Annealing

손실 함수는 **Reconstruction Loss (BCE)**와 **Regularization Loss (KLD)**의 합으로 정의된다.
특히 초기 학습 안정화를 위해 **KL Annealing** 기법이 적용되었다. `beta` 값은 0에서 시작하여 1까지 점진적으로 증가하며, 이는 학습 초기에 모델이 데이터 복원(Reconstruction)에 집중하도록 유도한다.

```python
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # 1. Reconstruction Loss: 입력과 복원 이미지 간의 차이 (Binary Cross Entropy)
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction='sum')
    
    # 2. KL Divergence: 잠재 분포와 표준정규분포 간의 차이
    # 공식: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Beta를 곱해 KLD의 영향력을 조절 (Annealing)
    return bce + beta * kld
```

### 3.3 가중치 초기화 (Weight Initialization)

Latent Space의 초기 분포가 발산하는 것을 방지하기 위해, 잠재 변수와 연결된 레이어(`fc2_mu`, `fc2_logvar`)에 대해 **특수한 초기화**가 도입되었다. 이를 통해 초기 $z$ 분포가 표준 정규 분포 $N(0, I)$에 근사하도록 설정된다.

```python
    def initialize_weights(self):
        # ... (일반 레이어는 Xavier 초기화) ...
        
        # Latent 파라미터는 매우 작은 값으로 초기화
        # 결과적으로 mu ~= 0, logvar ~= 0 (sigma ~= 1)이 됨
        nn.init.normal_(self.fc2_mu.weight, 0, 0.01)
        nn.init.constant_(self.fc2_mu.bias, 0)
        nn.init.normal_(self.fc2_logvar.weight, 0, 0.01)
        nn.init.constant_(self.fc2_logvar.bias, 0)
```

## 4. 결과 (Results)

### 4.1 잠재 공간 분포 (Latent Space Distribution)
다음 플롯은 훈련된 VAE의 2D 잠재 공간을 시각화한 결과이다. 각 점은 테스트 세트의 MNIST 숫자를 나타내며, 클래스 레이블에 따라 색상으로 구분된다.

**관찰 결과**
*   **클러스터링 (Clustering)**: 비지도 학습임에도 불구하고, 동일한 숫자끼리 잠재 공간 상에서 군집을 형성함이 관찰된다.
*   **유사성 (Similarity)**: 형태가 유사한 숫자들(예: 9와 7, 3과 8)이 인접한 영역에 위치하는 경향을 보인다.
*   **정규화 (Regularization)**: KL Divergence 정규화 효과로 인해 전체적인 분포 형태가 구형(표준 정규 분포)에 가깝게 유지된다.

### 4.2 생성된 숫자 (Latent Space Walk)
잠재 공간의 2D 메쉬 그리드(Grid) 상에서 좌표를 샘플링하고 디코딩하여, 잠재 변수의 변화에 따른 생성 이미지의 변화를 관찰하였다.

**관찰 결과**
*   **연속성 (Continuity)**: 잠재 공간의 축을 따라 이동함에 따라 생성된 숫자의 형태가 부드럽게 변형된다.
*   **매니폴드 학습 (Manifold Learning)**: 이는 VAE가 숫자 데이터의 연속적이고 의미 있는 잠재 매니폴드를 성공적으로 학습했음을 시사한다.
*   **중심과 외곽 (Center vs Edge)**: 분포의 중심부(0,0 부근)는 정형화된 평균적인 숫자를 생성하는 반면, 외곽으로 갈수록 스타일이 극단적이거나 변형된 형태의 숫자가 생성된다.
