# Variational Autoencoder (VAE): 이론적 배경 및 심층 분석

## 목차

1.  [서론: 확률론적 생성 모델 VAE](#1-서론-확률론적-생성-모델-VAE)
    *   1.1. [결정론적 모델과 확률적 모델의 차이](#11-결정론적-모델과-확률적-모델의-차이)
    *   1.2. [VAE 모델의 원리 (매니폴드 학습)](#12-vae-모델의-작동-원리)
2.  [수학적 배경](#2-수학적-배경)
3.  [구현 및 구조적 특징 (Implementation)](#3-구현-및-구조적-특징-implementation)
4.  [PyTorch 구현 상세 및 분석](#4-pytorch-구현-상세-및-분석)
5.  [확장: 조건부 VAE (Conditional VAE, CVAE)](#5-확장-조건부-vae-conditional-vae-cvae)
6.  [실험 결과 및 분석 (Results)](#6-실험-결과-및-분석-results)

---

## 1. 서론: 확률론적 생성 모델 VAE

### 1.1. 결정론적 모델과 확률적 모델의 차이
Autoencoder (AE)는 입력 데이터 $x$를 압축된 잠재 벡터 $z$로 매핑한 후, 이를 다시 $x'$으로 복원하는 과정을 학습한다. 이 과정은 결정론적으로 특정 입력 $x$는 항상 동일한 고정된 좌표의 잠재 벡터 $z$로 매핑된다. 이는 데이터의 압축과 차원 축소에는 효과적이나, 새로운 데이터를 생성하는 데는 근본적인 한계를 가진다. 잠재 공간이 불연속적일 수 있으며, 의미 있는 데이터가 존재하지 않는 빈 공간에서 샘플링할 경우 해독 불가능한 결과가 나올 수 있기 때문이다. 반면, Variational Autoencoder (VAE)는 데이터를 특정 좌표 점이 아닌 확률 분포로 매핑한다. 이는 입력 데이터를 잠재 공간 상의 점이 아닌 영역으로 해석함을 의미하며, 생성 모델로서의 핵심적인 차별점이다.

### 1.2. VAE 모델의 작동 원리
VAE는 잠재 공간의 불연속성을 해결하고, 고차원 데이터 공간에 내재된 저차원 구조인 매니폴드(Manifold)를 학습함으로써 연속적인 생성 모델링을 가능하게 한다. 여기서 '매니폴드를 학습한다'는 것은 다음과 같은 의미를 갖는다. 이미지와 같은 고차원 데이터(예: $28 \times 28 = 784$ 차원)는 전체 공간상에 무작위로 흩어져 있는 것이 아니라, 실제로는 훨씬 낮은 차원의 부분 공간에 밀집해 있다는 가정을 전제로 한다. VAE는 확률적 매핑을 통해 이 복잡하게 꼬여 있는 데이터의 저차원 구조를 찾아내어, 이를 평탄하고 연속적인 잠재 공간(Latent Space)으로 펴주는 변환을 학습한다. 이로 인해 잠재 공간 내에서의 미세한 이동이 생성된 이미지 상에서의 부드러운 형태 변화로 이어지게 된다.

---

## 2. 수학적 배경

VAE를 깊이 이해하기 위해서는 베이즈 정리와 변분 추론에 대한 이해가 필수적이다.

### 2.1. 문제의 정의: 베이즈 정리와 난제

VAE의 근본적인 목적은 우리가 가진 데이터셋의 분포를 가장 잘 설명할 수 있는 모델을 학습하는 것이다. 통계학에서는 이를 최대 우도 추정(Maximum Likelihood Estimation, MLE)이라 부르는데, 이는 "현재 관측된 데이터 $x$가 발생했을 확률이 가장 높도록 모델의 파라미터를 조정하는 것"을 의미한다. 이때 데이터 $x$가 모델에 의해 생성될 전체 확률 $p(x)$를 주변 우도(Marginal Likelihood) 또는 증거(Evidence)라 부른다. (후술할 $p(x|z)$인 '우도(Likelihood)'와 구별된다.) VAE의 학습 목표는 이 증거(Evidence)를 최대화하는 것이지만, 직접 계산이 불가능하기 때문에 베이즈 정리를 도입하여 우회적인 방법을 사용한다.

$$
p(z|x) = \frac{p(x|z)p(z)}{p(x)}
$$

*   **$p(z|x)$ (사후 확률, Posterior)**:
    관측된 데이터 $x$가 주어졌을 때, 해당 데이터를 생성했을 것으로 추정되는 잠재 변수 $z$의 분포를 의미한다. 이는 복잡한 데이터 공간에서 추출된 $x$로부터 잠재적 특징을 추출하는 인코더 과정에 해당한다. 생성 모델링의 핵심적인 계산 목표이나, 분모인 $p(x)$의 계산 복잡도로 인해 직접적인 도출이 불가능하다.

*   **$p(x|z)$ (우도, Likelihood)**:
    특정 잠재 변수 $z$로부터 관측 데이터 $x$가 생성될 확률을 나타낸다. 즉, 추상적인 잠재 표현 $z$가 실제 이미지 $x$로 어떻게 구체화되는지를 모델링하며, VAE에서는 이를 디코더 신경망으로 구현한다. 학습 과정에서 이 우도 값을 최대화하는 방향으로 모델 파라미터가 최적화된다.

*   **$p(z)$ (사전 확률, Prior)**:
    데이터 $x$를 관찰하기 전, 잠재 변수 $z$가 기본적으로 따를 것이라고 가정하는 확률 분포이다. VAE에서는 일반적으로 잠재 공간의 각 차원이 독립적인 표준 정규 분포를 따른다고 가정하며, 이는 잠재 공간을 규칙적이고 연속적으로 유지하려는 구조적 제약 역할을 수행한다.

*   **$p(x)$ (증거, Evidence / 주변 우도, Marginal Likelihood)**:
    관측된 데이터 $x$가 발생할 전체 확률을 의미하며, 모든 가능한 잠재 변수 $z$에 대해 우도와 사전 확률의 곱을 적분한 값($p(x) = \int p(x|z)p(z)dz$)과 같다. 고차원 잠재 공간에서는 이 적분 계산이 불가능(Intractable)하므로 사후 확률 $p(z|x)$를 직접 구할 수 없다. 이를 직접 계산하는 대신 ELBO(Evidence Lower Bound)를 최대화하는 우회 경로를 선택한다.

### 2.2. 변분 추론 (Variational Inference)
직접 계산이 불가능한 $p(z|x)$를 대신하기 위해, 다루기 쉬운 근사 분포 $q_{\phi}(z|x)$를 도입한다. 이 $q_{\phi}$는 신경망(인코더)으로 모델링된다. 변분 추론의 목표는 근사 분포 $q_{\phi}(z|x)$를 실제 사후 확률 $p(z|x)$에 최대한 가깝게 만드는 것, 즉 두 분포 사이의 쿨백-라이블러 발산 (Kullback-Leibler Divergence, KLD)를 최소화하는 파라미터 $\phi$를 찾는 것이다.

### 2.3. ELBO (Evidence Lower Bound) 상세 유도
로그 우도 $\log p(x)$를 최대화하는 문제는 수학적 전개를 통해 ELBO를 최대화하는 문제로 치환된다.

$$
\begin{aligned}
\log p(x) &= \log \int p(x, z) dz = \log \int p(x, z) \frac{q_\phi(z|x)}{q_\phi(z|x)} dz \\
&= \log \mathbb{E}_{q_\phi(z|x)} \left[ \frac{p(x, z)}{q_\phi(z|x)} \right] \\
&\ge \mathbb{E}_{q_\phi(z|x)} \left[ \log \frac{p(x, z)}{q_\phi(z|x)} \right] \quad (\because \text{Jensen's Inequality})
\end{aligned}
$$

부등식의 우변이 바로 **ELBO**이다. 이를 다시 정리하면 다음과 같다.

$$
\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
$$

즉, $\log p(x)$를 최대화하는 것은 ELBO를 최대화하는 것과 같으며, 이는 **Reconstruction Error를 줄이고(첫 번째 항)**, **잠재 분포가 사전 분포와 유사해지도록 정규화(두 번째 항)**하는 과정이다.

### 2.4. Kullback-Leibler Divergence (KLD)
KLD는 두 확률 분포의 차이를 측정하는 지표로, 항상 0 이상의 값을 가진다(Gibbs' inequality).
$$ D_{KL}(q \parallel p) = \mathbb{E}_{q} \left[ \log \frac{q(x)}{p(x)} \right] $$
VAE에서는 인코더가 출력하는 분포 $q_\phi(z|x)$가 사전 분포 $p(z)$인 표준 정규분포 $\mathcal{N}(0, I)$와 얼마나 가까운지를 측정하여 Loss에 반영한다.

---

## 3. 구현 및 구조적 특징 (Implementation)

이론적 모델을 실제 신경망으로 구현하기 위해서는 미분 불가능한 샘플링 과정을 해결하는 구조적 기법이 필요하다. 아래의 다이어그램은 VAE의 전체적인 아키텍처와 데이터 흐름, 특히 재파라미터화 트릭의 위치를 명확히 보여준다.

![VAE Structure](./vae_structure.png)
*그림 1. Variational Autoencoder의 구조 및 재파라미터화 트릭 도식*

### 3.1. 구조적 특징 분석
위 그림을 통해 VAE의 핵심적인 세 가지 구성 요소를 확인할 수 있다.

1.  **확률적 인코더 (Probabilistic Encoder, 좌측)**:
    *   그림의 $p_\theta(z|x) \approx q_\phi(z|x)$ 부분에 해당한다.
    *   일반적인 오토인코더와 달리, 인코더는 잠재 벡터 $z$를 직접 출력하지 않는다. 대신, 데이터가 따르는 가우시안 분포의 파라미터인 **평균($\mu$)**과 **분산($\sigma$)**을 추정하여 출력한다.

2.  **재파라미터화 트릭 (Reparameterization Trick, 중앙)**:
    *   그림 중앙의 $z = \mu + \sigma \odot \epsilon$ 연산 과정이다.
    *   만약 $z$를 분포 $N(\mu, \sigma)$에서 직접 샘플링한다면, 이는 미분 불가능한 확률적(Stochastic) 연산이 되어 역전파(Backpropagation)가 끊기게 된다.
    *   이를 해결하기 위해 확률적 요소인 노이즈 $\epsilon$을 외부 입력(Auxiliary Input)으로 분리하였다. $\epsilon \sim \mathcal{N}(0, I)$은 상수처럼 취급되므로, 네트워크는 $\mu$와 $\sigma$에 대해 미분 가능해지며 그라디언트가 인코더로 원활하게 흐를 수 있다.

3.  **확률적 디코더 (Probabilistic Decoder, 우측)**:
    *   그림의 $p_\theta(x|z)$ 부분이다.
    *   샘플링된 $z$를 입력받아 원본 데이터 $x$를 재구성한다.

---

## 4. PyTorch 구현 상세 및 분석

### 4.1. VAE 모델 아키텍처
다음은 `vae_mnist.py`에 구현된 VAE 클래스이다.

```python
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 2):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc2_mu = nn.Linear(400, latent_dim)
        self.fc2_logvar = nn.Linear(400, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28 * 28)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        mu, logvar = self.fc2_mu(h1), self.fc2_logvar(h1)
        z = self.reparameterize(mu, logvar)
        
        h3 = F.relu(self.fc3(z))
        recon_x = torch.sigmoid(self.fc4(h3))
        return recon_x, mu, logvar
```

### 4.2. 가중치 초기화 (Weight Initialization)
잠재 공간의 초기 분포가 발산하는 것을 방지하고 학습 초기 안정성을 확보하기 위해, 잠재 변수와 연결된 레이어(`fc2_mu`, `fc2_logvar`)에 대해 **특수한 초기화**를 적용하였다.

```python
def initialize_weights(self):
    # 일반 레이어: Xavier 초기화
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            
    # Latent 파라미터: 매우 작은 값으로 초기화하여 N(0, I)에 근사하게 시작
    nn.init.normal_(self.fc2_mu.weight, 0, 0.01)
    nn.init.normal_(self.fc2_logvar.weight, 0, 0.01)
```

### 4.3. 손실 함수 (Loss Function)와 KLD 유도
손실 함수는 **Reconstruction Loss**와 **Regularization Loss (KLD)**의 합이다. 또한, 초기 학습 시 KLD가 지나치게 커지는 것을 막기 위해 `beta` 값을 서서히 증가시키는 **KL Annealing**을 적용하였다.

```python
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # 1. Reconstruction Loss (Binary Cross Entropy)
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction='sum')
    
    # 2. KL Divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return bce + beta * kld
```

**참고: KLD 항의 수식적 유도**
두 가우시안 분포 $\mathcal{N}(\mu, \sigma^2)$와 $\mathcal{N}(0, 1)$ 사이의 KLD는 다음과 같이 유도된다.
$$
\begin{aligned}
D_{KL}(q \parallel p) &= \mathbb{E}_{q} [\log q(z) - \log p(z)] \\
&= \int \mathcal{N}(z; \mu, \sigma^2) \log \frac{\mathcal{N}(z; \mu, \sigma^2)}{\mathcal{N}(z; 0, 1)} dz \\
&= -\frac{1}{2} \sum \left( 1 + \log(\sigma^2) - \mu^2 - \sigma^2 \right)
\end{aligned}
$$
이 수식은 코드의 `kld` 계산 로직과 정확히 일치한다.

---

## 5. 확장: 조건부 VAE (Conditional VAE, CVAE)

기존 VAE의 "생성 대상을 제어할 수 없다"는 한계를 극복하기 위해 **CVAE**가 제안되었다.

### 5.1. 핵심 아이디어
인코더와 디코더 모두에게 **조건 정보 $c$ (예: 숫자 클래스 레이블)**를 추가로 입력받는다. 이를 통해 모델은 주어진 조건 하에서의 데이터 생성 분포 $p(x|z, c)$를 학습하게 된다.

### 5.2. PyTorch 구현 코드
이미지 벡터와 One-Hot 인코딩된 레이블 벡터를 **결합(Concatenate)**하여 입력으로 사용한다.

```python
class CVAE(nn.Module):
    def __init__(self, latent_dim=2, num_classes=10):
        super(CVAE, self).__init__()
        # 입력 차원 확장: 784 + 10
        self.fc1 = nn.Linear(28*28 + num_classes, 400)
        # ... (중략) ...

    def encode(self, x, c):
        # 이미지와 레이블 결합
        inputs = torch.cat([x, c], dim=1)
        h1 = F.relu(self.fc1(inputs))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def decode(self, z, c):
        # 잠재 변수와 레이블 결합
        inputs = torch.cat([z, c], dim=1)
        # ... (중략) ...
```

---

## 6. 실험 결과 및 분석 (Results)

### 6.1. 잠재 공간 분포 (Latent Space Distribution)
훈련된 VAE의 2D 잠재 공간을 시각화한 결과, 비지도 학습임에도 불구하고 **동일한 숫자끼리 군집(Cluster)을 형성**함을 확인하였다. 유사한 형태의 숫자(예: 3과 8)는 인접한 영역에 위치하며, KL 정규화로 인해 전체적인 분포는 원점 중심의 구형을 띤다.

### 6.2. 생성된 숫자 (Latent Space Walk)
잠재 공간의 축을 따라 좌표를 이동하며 이미지를 생성했을 때, 숫자의 형태가 **부드럽고 연속적으로 변형**되는 것을 관찰할 수 있다. 이는 VAE가 데이터의 불연속적인 공간이 아닌, 연속적인 매니폴드를 성공적으로 학습했음을 시사한다.
