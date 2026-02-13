# Variational Autoencoder (VAE) 분석 및 수학적 배경

이 문서는 Variational Autoencoders (VAE)에 대한 포괄적인 분석을 제공하며, 수학적 기초, PyTorch를 사용한 구현 세부 사항, 그리고 MNIST 데이터셋에 대한 결과 분석을 다룹니다.

## 1. VAE 소개

Variational Autoencoders (VAE)는 딥러닝과 확률적 그래픽 모델을 결합한 강력한 생성 모델입니다. 입력 데이터를 고정된 벡터로 매핑하는 기존의 오토인코더와 달리, VAE는 입력 데이터를 잠재 공간(latent space)의 **확률 분포**로 매핑합니다. 이를 통해 VAE는 이 잠재 공간에서 샘플링하여 현실적인 새로운 데이터를 생성할 수 있습니다.

## 2. 수학적 배경

### 2.1 베이즈 정리와 VAE (Bayes' Theorem and VAE)
베이즈 정리는 사전 지식을 바탕으로 어떤 사건의 확률을 계산하는 방법을 설명합니다. 일반적인 형태는 다음과 같습니다:

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$


VAE의 문맥에서 이 변수들을 다음과 같이 매핑할 수 있습니다:
*   $A \rightarrow z$ (잠재 변수 / Code)
*   $B \rightarrow x$ (관측 데이터 / Image)

VAE에 맞춰 베이즈 정리를 다시 쓰면:

$$
p(z|x) = \frac{p(x|z)p(z)}{p(x)}
$$

*   **$p(z|x)$ (사후 확률, Posterior)**: 데이터 $x$가 주어졌을 때 잠재 변수 $z$의 확률입니다. 우리가 알아내고자 하는 것(인코더)입니다.
*   **$p(x|z)$ (우도, Likelihood)**: 잠재 변수 $z$가 주어졌을 때 데이터 $x$의 확률입니다. 이것은 디코더에 해당합니다.
*   **$p(z)$ (사전 확률, Prior)**: 잠재 변수의 가정된 분포입니다 (보통 표준 정규분포 $\mathcal{N}(0, I)$).
*   **$p(x)$ (증거, Evidence)**: 데이터 자체의 확률입니다. 이는 $\int p(x|z)p(z)dz$로 계산됩니다.

VAE의 핵심 문제는 분모인 **증거 $p(x)$**가 모든 가능한 $z$에 대한 적분을 포함하고 있어 계산이 불가능(intractable)하다는 점입니다. 이로 인해 사후 확률 $p(z|x)$를 직접 계산할 수 없으며, 이를 해결하기 위해 $p(z|x)$를 $q_\phi(z|x)$로 근사하는 **변분 추론(Variational Inference)** (2.3절)이 등장하게 됩니다.

### 2.2 생성 과정 (The Generative Process)
우리는 데이터 $x$가 잠재 변수 $z$로부터 다음의 두 단계 과정을 통해 생성된다고 가정합니다:
1.  사전 분포(prior distribution) $p_\theta(z)$에서 잠재 벡터 $z$를 샘플링합니다. 일반적으로 $p_\theta(z) = \mathcal{N}(0, I)$로 가정합니다.
2.  조건부 분포(conditional distribution) $p_\theta(x|z)$에서 관측값 $x$를 생성합니다.

목표는 진정한 사후 분포(true posterior) $p_\theta(z|x)$를 추정하는 것이지만, 증거(evidence) $p_\theta(x) = \int p_\theta(x|z)p_\theta(z)dz$의 적분 계산이 불가능하여 직접 구하는 것은 어렵습니다.

### 2.3 변분 추론 (Variational Inference)
이 계산 불가능성을 해결하기 위해 신경망(**인코더**)으로 파라미터화된 근사 사후 분포(approximate posterior) $q_\phi(z|x)$를 도입합니다. 우리는 $q_\phi(z|x)$가 진정한 사후 분포 $p_\theta(z|x)$와 최대한 가까워지기를 원합니다. 이를 위해 두 분포 사이의 쿨백-라이블러(KL) 발산(Kullback-Leibler divergence)을 최소화합니다:

$$
D_{KL}(q_\phi(z|x) || p_\theta(z|x))
$$

### 2.4 Evidence Lower Bound (ELBO) 상세 유도

우리의 목표는 데이터 $x$의 우도(Likelihood)인 $p_\theta(x)$를 최대화하는 것입니다. 하지만 $p_\theta(x) = \int p_\theta(x|z)p(z)dz$ 적분은 계산이 불가능(intractable)합니다. 따라서 로그 우도(Log-likelihood)의 하한(Lower Bound)을 최대화하는 방식을 사용합니다.

1.  **로그 우도의 변형**
    변분 분포(Variational distribution) $q_\phi(z|x)$를 도입하여 식을 변형합니다.
    $$
    \log p_\theta(x) = \int q_\phi(z|x) \log p_\theta(x) dz
    $$
    ($\int q_\phi(z|x)dz = 1$ 이므로 곱해도 값은 변하지 않음)

2.  **베이즈 정리 및 분수 형태 도입**
    $$
    \log p_\theta(x) = \int q_\phi(z|x) \log \left( \frac{p_\theta(x, z)}{p_\theta(z|x)} \cdot \frac{q_\phi(z|x)}{q_\phi(z|x)} \right) dz
    $$

3.  **항의 분리 (ELBO와 KL Divergence)**
    로그 성질을 이용하여 식을 분리합니다.
    $$
    \begin{aligned}
    \log p_\theta(x) &= \int q_\phi(z|x) \log \left( \frac{p_\theta(x, z)}{q_\phi(z|x)} \cdot \frac{q_\phi(z|x)}{p_\theta(z|x)} \right) dz \\
    &= \underbrace{\int q_\phi(z|x) \log \left( \frac{p_\theta(x, z)}{q_\phi(z|x)} \right) dz}_{\text{ELBO}} + \underbrace{\int q_\phi(z|x) \log \left( \frac{q_\phi(z|x)}{p_\theta(z|x)} \right) dz}_{D_{KL}(q_\phi(z|x) || p_\theta(z|x))}
    \end{aligned}
    $$

4.  **ELBO의 재구성**
    여기서 ELBO 항만 다시 정리하면 우리가 아는 최종 식이 나옵니다.
    $$
    \begin{aligned}
    \text{ELBO} &= \int q_\phi(z|x) \log \frac{p_\theta(x|z)p(z)}{q_\phi(z|x)} dz \\
    &= \int q_\phi(z|x) \log p_\theta(x|z) dz + \int q_\phi(z|x) \log \frac{p(z)}{q_\phi(z|x)} dz \\
    &= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
    \end{aligned}
    $$
    - 첫 번째 항: **Reconstruction Error** (데이터를 잘 복원하도록 학습)
    - 두 번째 항: **Regularization** (잠재 변수 $z$의 분포가 사전 분포 $p(z)$와 유사해지도록 학습)

### 2.5 Kullback-Leibler Divergence (KLD)의 비음수성 증명

KL Divergence가 항상 0 이상임($D_{KL} \ge 0$)을 젠센 부등식(Jensen's Inequality)을 통해 증명합니다.

1.  **정의**
    $$ D_{KL}(q||p) = \int q(x) \log \frac{q(x)}{p(x)} dx = \mathbb{E}_q \left[ -\log \frac{p(x)}{q(x)} \right] $$

2.  **젠센 부등식 (Jensen's Inequality)**
    함수 $f$가 볼록 함수(convex function)일 때, $\mathbb{E}[f(X)] \ge f(\mathbb{E}[X])$ 가 성립합니다.
    $-\log(x)$는 볼록 함수이므로 젠센 부등식을 적용할 수 있습니다.

3.  **증명 과정**
    $$
    \begin{aligned}
    D_{KL}(q||p) &= \mathbb{E}_q \left[ -\log \frac{p(x)}{q(x)} \right] \\
    &\ge -\log \left( \mathbb{E}_q \left[ \frac{p(x)}{q(x)} \right] \right) \quad (\text{Jensen's Inequality}) \\
    &= -\log \left( \int q(x) \frac{p(x)}{q(x)} dx \right) \\
    &= -\log \left( \int p(x) dx \right) \\
    &= -\log(1) \\
    &= 0
    \end{aligned}
    $$
    따라서, **$D_{KL}(q||p) \ge 0$** 이 성립합니다.

### 2.6 재파라미터화 트릭 (The Reparameterization Trick)
경사 하강법을 사용하여 네트워크를 훈련하려면 $z$의 샘플링 과정을 통해 역전파(backpropagate)를 수행해야 합니다. 하지만 샘플링은 미분 불가능한 연산입니다. **재파라미터화 트릭**은 $z$를 무작위 노이즈 변수 $\epsilon$과 인코더 출력($\mu, \sigma$)의 결정론적 변환으로 표현하여 이 문제를 해결합니다:

$$
z = \mu + \sigma \odot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)
$$

이로써 샘플링 과정이 $\phi$에 대해 미분 가능해집니다.

## 3. PyTorch 구현 및 분석

우리는 PyTorch를 사용하여 VAE를 구현하고 MNIST 데이터셋에 대해 훈련했습니다. 모델은 다음과 같이 구성됩니다:
*   **인코더 (Encoder)**: $28 \times 28$ 이미지를 잠재 공간의 평균 $\mu$와 로그 분산 $\log(\sigma^2)$ 벡터로 매핑합니다.
*   **디코더 (Decoder)**: $z$를 샘플링하고 이미지를 재구성합니다.
*   **손실 함수 (Loss)**: 이진 교차 엔트로피 (Reconstruction) + KL Divergence.

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
