# Variational Autoencoder (VAE) 분석 및 수학적 배경

이 문서는 Variational Autoencoders (VAE)에 대한 포괄적인 분석을 제공하며, 수학적 기초, PyTorch를 사용한 구현 세부 사항, 그리고 MNIST 데이터셋에 대한 결과 분석을 다룹니다.

## 1. VAE 소개

VAE는 딥러닝과 확률 모델을 결합한 생성 모델입니다. VAE는 입력 데이터를 잠재 공간(latent space)의 **확률 분포**로 매핑합니다. 이를 통해 VAE는 이 잠재 공간에서 샘플링하여 새로운 데이터를 생성할 수 있습니다.

## 2. 수학적 배경

이 섹션에서는 VAE를 이해하기 위해 필요한 수학적 개념들을 기초부터 순서대로 다룹니다.

### 2.1 베이즈 정리 (Bayes' Theorem)
베이즈 정리는 사전 지식을 바탕으로 어떤 사건의 확률을 계산하는 방법을 설명합니다. 일반적인 형태는 다음과 같습니다:

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

VAE의 관점에서 이 변수들을 다음과 같이 대응시킬 수 있습니다:
*   $A \rightarrow z$ (잠재 변수)
*   $B \rightarrow x$ (관측 데이터)

즉, VAE에 맞춰 베이즈 정리를 다시 쓰면 다음과 같습니다:

$$
p(z|x) = \frac{p(x|z)p(z)}{p(x)}
$$

*   **$p(z|x)$ (사후 확률, Posterior)**:
    데이터 $x$를 관찰한 후, 이 데이터가 어떤 잠재 변수 $z$에서 기인했을지에 대한 확률입니다. 복잡한 이미지 $x$를 보고 "이 이미지를 생성했을 법한 실제 $z$는 무엇인가?"를 추론하는 **인코더(Encoder)** 과정입니다. 생성 모델링에서 우리가 궁극적으로 계산하고 싶은 값이지만, 분모인 $p(x)$ 때문에 직접 계산이 어렵습니다.

*   **$p(x|z)$ (우도, Likelihood)**:
    특정 잠재 변수(개념) $z$로부터 관측 데이터 $x$가 생성될 확률입니다. 즉, "추상적인 개념 $z$가 주어졌을 때, 이것이 실제 이미지 $x$로 어떻게 구체화되는가?"를 나타냅니다. VAE에서는 이를 **디코더(Decoder)** 신경망으로 모델링하며, 모델이 학습될수록 이 확률값이 높아집니다.

*   **$p(z)$ (사전 확률, Prior)**:
    데이터 $x$를 관찰하기 전, 잠재 변수 $z$가 어떤 분포를 가질 것이라는 **우리의 믿음(Assumption)**입니다. VAE에서는 보통 잠재 공간의 모든 차원이 서로 독립적이고 표준 정규 분포를 따른다고 가정합니다 ($p(z) \sim \mathcal{N}(0, I)$). 이는 잠재 공간을 규칙적이고 연속적으로 유지하려는 **'구조적 제약'** 역할을 합니다.

*   **$p(x)$ (증거, Evidence)**:
    관측된 데이터 $x$가 나타날 전체 확률(**Marginal Likelihood**)입니다. 이는 모든 가능한 잠재 변수 $z$에 대해 우도와 사전 확률의 곱을 합산(적분)한 값입니다 ($p(x) = \int p(x|z)p(z)dz$). 잠재 공간이 고차원일수록 적분이 불가능(**Intractable**)해지기 때문에, VAE는 이 값을 직접 구하는 대신 **ELBO(하한)**를 사용하여 우회합니다.

### 2.2 최대 우도 추정 (Maximum Likelihood Estimation)
최대 우도 추정은 통계학에서 모델의 파라미터를 추정하는 직관적이고 강력한 방법입니다. 그 핵심 아이디어는 **현재 관측된 데이터가 나올 확률이 가장 높도록 모델의 파라미터를 조정하는 것**입니다. 생성 모델의 궁극적인 목표는 데이터셋(예: MNIST 숫자 이미지)의 분포를 모델이 학습하는 것입니다. 만약 모델이 데이터의 분포 $p_\theta(x)$를 완벽하게 학습한다면, 모델은 실제 데이터와 구별할 수 없는 새로운 샘플을 생성할 수 있습니다.

수식으로 표현하면, 관측된 데이터 $x$에 대해 $p_\theta(x)$를 최대화하는 파라미터 $\theta$를 찾는 것입니다:

$$
\theta^* = \text{argmax}_\theta \sum_{i=1}^N \log p_\theta(x^{(i)})
$$

여기서 $p(x)$를 베이즈 정리의 '증거(Evidence)'가 아닌 **'우도(Likelihood)'**라고 부르는 이유는 관점의 차이 때문입니다. 베이즈 정리에서는 파라미터가 고정된 상태에서 관측값의 확률을 다루지만, MLE에서는 데이터 $x$를 고정하고 파라미터 $\theta$를 변화시키며 "해당 파라미터가 데이터를 설명하기에 얼마나 그럴듯한가(Likely)"를 평가하기 때문입니다.

또한, 우리는 단순 확률 $p_\theta(x)$ 대신 **로그 확률(Log-Likelihood)**을 최대화합니다. 이는 두 가지 실용적인 이유 때문입니다. 첫째, 확률값($0 \le p \le 1$)을 계속 곱하면 수치가 0으로 수렴하는 언더플로우(Underflow) 문제가 발생하는데, 로그를 취하면 곱셈이 덧셈으로 바뀌어 이를 방지할 수 있습니다 (**수치적 안정성**). 둘째, 곱셈 연산보다 덧셈 연산의 미분이 훨씬 계산하기 쉽습니다 (**계산의 편의성**). 로그 함수는 단조 증가 함수이므로, 로그 우도를 최대화하는 것은 원본 우도를 최대화하는 것과 수학적으로 동일합니다.

이 값이 최대가 될 때, 모델은 실제 데이터 분포와 가장 유사해집니다. 수학적으로는 실제 분포와 모델 분포 사이의 KL Divergence를 최소화하는 것과 같습니다.

하지만 **VAE와 같은 잠재 변수 모델(Latent Variable Model)에서는 문제가 존재합니다.**
$p_\theta(x)$를 계산하려면 모든 가능한 잠재 변수 $z$에 대해 적분해야 하는데 ($p_\theta(x) = \int p_\theta(x|z)p(z)dz$), 이 적분 계산이 불가능(intractable)합니다. 따라서 우도를 직접 최대화하는 대신, 우도의 **하한(Lower Bound, ELBO)**을 최대화하는 우회적인 방법을 사용합니다.

### 2.3 쿨백-라이블러 발산 (Kullback-Leibler Divergence, KLD)
두 확률 분포 $q(x)$와 $p(x)$가 얼마나 다른지를 측정하는 지표입니다. VAE에서는 근사 분포 $q$와 실제 분포 $p$ 사이의 차이를 줄이는 데 사용됩니다.

**정의:**

$$
D_{KL}(q \parallel p) = \int q(x) \log \frac{q(x)}{p(x)} dx = \mathbb{E}_{q} \left[ \log \frac{q(x)}{p(x)} \right]
$$

(여기서 적분 $\int q(x) (...) dx$는 확률 분포 $q(x)$에 대한 기댓값 $\mathbb{E}_q[...] $와 동일합니다)

**젠센 부등식 (Jensen's Inequality) 이란?**

젠센 부등식은 볼록 함수(convex function)의 기댓값과 기댓값의 함수값 사이의 관계를 나타냅니다.

*   함수 $f(x)$ 가 **볼록(convex)** 할 때 : $\mathbb{E}[f(x)] \ge f(\mathbb{E}[x])$
*   함수 $f(x)$ 가 **오목(concave)** 할 때 : $\mathbb{E}[f(x)] \le f(\mathbb{E}[x])$

여기서 우리는 $-\log(x)$ 함수를 사용합니다. 로그 함수는 오목 함수이지만, 마이너스가 붙은 $-\log x$는 아래로 볼록한 **볼록 함수**입니다. 따라서 젠센 부등식을 적용할 수 있습니다.

**비음수성 증명:**
$D_{KL}$은 항상 0 이상입니다. ($-\log$는 볼록 함수)

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
VAE는 데이터 $x$가 잠재 변수 $z$로부터 생성된다고 가정합니다:
1.  **Prior**: $z \sim p_\theta(z)$ (보통 표준 정규분포)
2.  **Likelihood**: $x \sim p_\theta(x|z)$ (디코더 신경망)

우리는 $p_\theta(z|x)$를 알고 싶지만, 앞서 언급했듯 $p(x)$를 구할 수 없어 계산이 불가능합니다.

### 2.5 변분 추론 (Variational Inference)
계산 불가능한 $p_\theta(z|x)$ 대신, 다루기 쉬운 근사 분포 $q_\phi(z|x)$ (인코더)를 도입합니다. 목표는 $q_\phi$를 $p_\theta$에 최대한 가깝게 만드는 것, 즉 $D_{KL}(q_\phi \parallel p_\theta)$를 최소화하는 것입니다.

### 2.6 Evidence Lower Bound (ELBO) 상세 유도
우리는 $\log p_\theta(x)$를 최대화하는 것을 목표로 합니다. 이를 위해 식을 다음과 같이 변형할 수 있습니다.

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

**최종 ELBO 식:**

$$
\begin{aligned}
\text{ELBO} &= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z) + \log p(z) - \log q_\phi(z|x)] \\
&= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \parallel p(z))
\end{aligned}
$$
*   첫 번째 항: **Reconstruction Error**
*   두 번째 항: **Regularization** (Prior $p(z)$와 $q_\phi$의 차이)

### 2.7 재파라미터화 트릭 (The Reparameterization Trick)

**샘플링의 문제점**

VAE의 인코더는 평균 $\mu$와 분산 $\sigma^2$를 출력합니다. 그리고 이 분포에서 잠재 변수 $z$를 샘플링하여 디코더에 전달합니다.
$$\mathcal{N}(\mu, \sigma^2)$$
문제는 **"샘플링(Sampling)"**이라는 과정 자체가 미분이 불가능한(non-differentiable) 무작위 연산이라는 점입니다.

신경망을 학습시키려면 오차 역전파(Backpropagation)를 통해 경사(Gradient)가 흘러가야 합니다. 하지만 $z$가 무작위로 뽑힌 값이라면, 이 무작위 노드를 통과해서 인코더의 파라미터($\phi$)로 미분값을 전달할 수 없습니다. 즉, 체인 룰(Chain Rule)이 끊기게 됩니다.

**해결책: 무작위성의 분리**

재파라미터화 트릭의 핵심 아이디어는 **$z$를 결정론적(deterministic) 부분과 확률적(stochastic) 부분으로 분리하는 것**입니다.
$z$를 직접 샘플링하는 대신, 외부에서 무작위 노이즈 $\epsilon$을 주입하여 $z$를 생성합니다.

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

여기서:
*   $\mu, \sigma$: 인코더(신경망)의 출력 (결정론적, 미분 가능)
*   $\epsilon$: 표준 정규분포에서 샘플링한 노이즈 (상수 취급, 미분 불필요)
*   $\odot$: 요소별 곱 (Element-wise product)

이제 $z$는 $\mu$와 $\sigma$에 대한 **함수**가 되었습니다. 따라서 $z$를 $\mu$와 $\sigma$로 미분할 수 있게 되었고($\frac{\partial z}{\partial \mu}=1, \frac{\partial z}{\partial \sigma}=\epsilon$), 역전파가 인코더까지 막힘없이 흐를 수 있게 됩니다. 이것이 VAE 학습을 가능하게 하는 핵심 테크닉입니다.

### 2.8 조건부 VAE (Conditional VAE, CVAE)

기존의 VAE는 강력한 생성 모델이지만, 한 가지 결정적인 한계가 있습니다. 바로 **"생성할 대상을 제어할 수 없다"**는 점입니다. 예를 들어, 숫자 7을 생성하고 싶어도 VAE는 잠재 공간에서 무작위로 샘플링하기 때문에 7이 나올지 9가 나올지 보장할 수 없습니다. 이를 해결하기 위해 제안된 것이 **Conditional VAE (CVAE)**입니다.

**핵심 아이디어: 조건(Condition) $c$의 주입**

CVAE는 인코더와 디코더 모두에게 **조건 정보 $c$ (예: 숫자 레이블)**를 추가로 입력받습니다. 이를 통해 모델은 주어진 조건 하에서 데이터를 생성하고 잠재 변수를 매핑하는 법을 학습합니다.

**수식적 변화:**

모든 확률 분포가 조건 $c$에 종속되도록 변경됩니다.

1.  **인코더 (Encoder)**: $q_\phi(z|x, c)$
    *   입력: 이미지 $x$ + 레이블 $c$ (Concatenation)
    *   출력: 잠재 변수 $z$의 분포 파라미터 ($\mu, \sigma$)
    *   의미: "숫자 7($c$)인 이미지 $x$를 잠재 공간의 어디($z$)에 매핑해야 하는가?"

2.  **디코더 (Decoder)**: $p_\theta(x|z, c)$
    *   입력: 잠재 변수 $z$ + 레이블 $c$
    *   출력: 이미지 $x$
    *   의미: "잠재 변수 $z$와 숫자 7($c$)이라는 정보를 가지고 이미지 $x$를 그려라."

3.  **목적 함수 (CVAE Loss)**:
    ELBO 식에도 조건 $c$가 추가됩니다.
    $$
    \log p(x|c) \ge \mathbb{E}_{q(z|x,c)}[\log p(x|z,c)] - D_{KL}(q(z|x,c) || p(z|c))
    $$
    *   보통 Prior $p(z|c)$는 $c$와 무관하게 표준 정규 분포 $\mathcal{N}(0, I)$로 가정합니다. 즉, 어떤 숫자를 그리든 잠재 공간의 분포 자체는 동일한 형태를 유지하도록 합니다.

**구조적 차이점 (Implementation Detail):**

실제 구현(PyTorch)에서는 주로 **One-Hot Encoding**된 레이블 벡터를 이미지나 잠재 변수와 **결합(Concatenate)**하여 사용합니다.

*   **인코더 입력**: `[Batch, 784]` (이미지) + `[Batch, 10]` (레이블) $\rightarrow$ `[Batch, 794]`
*   **디코더 입력**: `[Batch, 2]` (잠재 변수) + `[Batch, 10]` (레이블) $\rightarrow$ `[Batch, 12]`

이처럼 간단한 구조 변경만으로도, CVAE는 우리가 원하는 특정 숫자(클래스)를 정확하게 생성해낼 수 있는 강력한 제어 능력을 갖게 됩니다.

## 3. PyTorch 구현 및 분석

## 3. PyTorch 구현 및 분석

우리는 PyTorch를 사용하여 VAE와 CVAE를 구현하고 MNIST 데이터셋에 대해 훈련했습니다. 다음은 `vae_mnist.py`에 구현된 주요 모델 구성 요소에 대한 상세 분석입니다.

### 3.1 모델 아키텍처 (Model Architecture)

#### 인코더 (Encoder)
인코더는 이미지를 입력받아 잠재 공간의 파라미터인 평균($\mu$)과 로그 분산($\log \sigma^2$)을 출력합니다.
*   **VAE**: $28 \times 28$ 이미지를 평탄화(Flatten)하여 입력받습니다.
*   **CVAE**: 이미지와 One-Hot 인코딩된 레이블을 **결합(Concatenate)**하여 입력받습니다.

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
역전파(Backpropagation)가 가능하도록 무작위성을 분리하는 핵심 부분입니다. 로그 분산(`logvar`)을 사용하는 이유는 분산이 항상 양수여야 한다는 제약을 자연스럽게 만족시키기 위함입니다 ($\sigma = e^{0.5 \times \log\sigma^2}$).

```python
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)  # 표준편차 복원
        eps = torch.randn_like(std)    # 표준정규분포에서 노이즈 샘플링
        return mu + eps * std          # z = mu + epsilon * sigma
```

#### 디코더 (Decoder)
잠재 변수 $z$로부터 원본 이미지를 복원합니다.
*   **CVAE**: 인코더와 마찬가지로 $z$와 레이블 $c$를 결합하여 입력받습니다. 이를 통해 모델은 "어떤 숫자($c$)를 그려야 하는지" 알 수 있습니다.

```python
    def decode(self, z: torch.Tensor, c: torch.Tensor):
        # 잠재 변수(z)와 레이블(c)을 결합
        inputs = torch.cat([z, c], 1)
        h3 = F.relu(self.fc3(inputs))
        return torch.sigmoid(self.fc4(h3)) # 픽셀 값을 0~1 사이 확률로 출력
```

### 3.2 손실 함수 (Loss Function)와 KL Annealing

손실 함수는 **Reconstruction Loss (BCE)**와 **Regularization Loss (KLD)**의 합으로 정의됩니다.
특히 초기 학습 안정화를 위해 **KL Annealing** 기법이 적용되어 있습니다. `beta` 값은 0에서 시작하여 1까지 점진적으로 증가하며, 이는 초기에 모델이 복원(Reconstruction)에 집중하도록 돕습니다.

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

Latent Space의 초기 분포가 엉뚱한 곳으로 튀는 것을 방지하기 위해, 잠재 변수와 연결된 레이어(`fc2_mu`, `fc2_logvar`)에 대해 **특수한 초기화**를 적용했습니다. 이를 통해 초기 $z$ 분포가 표준 정규 분포 $N(0, I)$에 가깝게 시작하도록 유도합니다.

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
다음 플롯은 훈련된 VAE의 2D 잠재 공간을 시각화한 것입니다. 각 점은 테스트 세트의 MNIST 숫자를 나타내며, 클래스 레이블에 따라 색상으로 구분됩니다.

**관찰**:
*   숫자들이 훈련 중에 레이블 정보가 사용되지 않았음에도(비지도 학습) 잠재 공간에서 뚜렷한 클러스터를 형성합니다.
*   유사한 숫자들(예: 9와 7, 3과 8)은 서로 가깝게 위치하는 경향이 있습니다.
*   KL Divergence 정규화로 인해 분포가 대략적으로 표준 정규 분포를 따릅니다.

### 4.2 생성된 숫자 (Latent Space Walk)
잠재 공간의 2D 메쉬 그리드에서 점을 샘플링하고 디코딩하여 새로운 숫자를 생성할 수 있습니다.

**관찰**:
*   잠재 공간의 축을 따라 이동함에 따라 생성된 숫자가 하나에서 다른 하나로 부드럽게 변합니다.
*   이는 VAE가 숫자 데이터의 연속적이고 의미 있는 매니폴드를 학습했음을 보여줍니다.
*   플롯의 중심(0,0 부근)은 일반적이고 평균적인 모양의 숫자에 해당하며, 가장자리는 더 극단적인 스타일의 변형을 나타냅니다.
