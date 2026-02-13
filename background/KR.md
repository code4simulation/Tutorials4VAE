# Variational Autoencoder (VAE) 분석 및 수학적 배경

이 문서는 Variational Autoencoders (VAE)에 대한 포괄적인 분석을 제공하며, 수학적 기초, PyTorch를 사용한 구현 세부 사항, 그리고 MNIST 데이터셋에 대한 결과 분석을 다룹니다.

## 1. VAE 소개

Variational Autoencoders (VAE)는 딥러닝과 확률적 그래픽 모델을 결합한 강력한 생성 모델입니다. 입력 데이터를 고정된 벡터로 매핑하는 기존의 오토인코더와 달리, VAE는 입력 데이터를 잠재 공간(latent space)의 **확률 분포**로 매핑합니다. 이를 통해 VAE는 이 잠재 공간에서 샘플링하여 현실적인 새로운 데이터를 생성할 수 있습니다.

## 2. 수학적 배경

이 섹션에서는 VAE를 이해하기 위해 필요한 수학적 개념들을 기초부터 순서대로 다룹니다.

### 2.1 베이즈 정리 (Bayes' Theorem)
베이즈 정리는 사전 지식을 바탕으로 어떤 사건의 확률을 계산하는 방법을 설명합니다. 일반적인 형태는 다음과 같습니다:

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

VAE의 문맥에서 이 변수들을 다음과 같이 매핑할 수 있습니다:
*   $A \rightarrow z$ (잠재 변수)
*   $B \rightarrow x$ (관측 데이터)

VAE에 맞춰 베이즈 정리를 다시 쓰면:

$$
p(z|x) = \frac{p(x|z)p(z)}{p(x)}
$$

*   **$p(z|x)$ (사후 확률, Posterior)**: 데이터 $x$가 주어졌을 때 잠재 변수 $z$의 확률입니다. (인코더)
*   **$p(x|z)$ (우도, Likelihood)**: 잠재 변수 $z$가 주어졌을 때 데이터 $x$의 확률입니다. (디코더)
*   **$p(z)$ (사전 확률, Prior)**: 잠재 변수의 가정된 분포입니다 (보통 $\mathcal{N}(0, I)$).
*   **$p(x)$ (증거, Evidence)**: 데이터 자체의 확률입니다.

### 2.2 최대 우도 추정 (Maximum Likelihood Estimation)
**"왜 우리는 우도(Likelihood)를 최대화해야 할까요?"**

최대 우도 추정(MLE)은 통계학에서 모델의 파라미터를 추정하는 가장 직관적이고 강력한 방법입니다. 그 핵심 아이디어는 **"현재 관측된 데이터가 나올 확률이 가장 높도록 모델의 파라미터를 조정하는 것"**입니다.

생성 모델(Generative Model)의 궁극적인 목표는 우리가 가진 데이터셋(예: MNIST 숫자 이미지)의 분포를 모델이 학습하는 것입니다. 만약 모델이 데이터의 분포 $p_\theta(x)$를 완벽하게 학습한다면, 모델은 실제 데이터와 구별할 수 없는 새로운 샘플을 생성할 수 있습니다.

수식으로 표현하면, 관측된 데이터 $x$에 대해 $p_\theta(x)$를 최대화하는 파라미터 $\theta$를 찾는 것입니다:

$$
\theta^* = \text{argmax}_\theta \sum_{i=1}^N \log p_\theta(x^{(i)})
$$

이 값이 최대가 될 때, 모델은 실제 데이터 분포와 가장 유사해집니다 (수학적으로는 실제 분포와 모델 분포 사이의 KL Divergence를 최소화하는 것과 같습니다).

하지만 **VAE와 같은 잠재 변수 모델(Latent Variable Model)에서는 문제가 있습니다.**
$p_\theta(x)$를 계산하려면 모든 가능한 잠재 변수 $z$에 대해 적분해야 하는데($p_\theta(x) = \int p_\theta(x|z)p(z)dz$), 이 적분 계산이 불가능(intractable)합니다. 따라서 우리는 우도를 직접 최대화하는 대신, 우도의 **하한(Lower Bound, ELBO)**을 최대화하는 우회적인 방법을 사용하게 됩니다.

### 2.3 쿨백-라이블러 발산 (Kullback-Leibler Divergence, KLD)
두 확률 분포 $q(x)$와 $p(x)$가 얼마나 다른지를 측정하는 지표입니다. VAE에서는 근사 분포 $q$와 실제 분포 $p$ 사이의 차이를 줄이는 데 사용됩니다.

**정의:**
$$
D_{KL}(q \parallel p) = \mathbb{E}_q \left[ \log \frac{q(x)}{p(x)} \right] = \int q(x) \log \frac{q(x)}{p(x)} dx
$$

**젠센 부등식 (Jensen's Inequality) 이란?**
젠센 부등식은 볼록 함수(convex function)의 기댓값과 기댓값의 함수값 사이의 관계를 나타냅니다.
*   함수 $f(x)$가 **볼록(convex)**할 때: $\mathbb{E}[f(x)] \ge f(\mathbb{E}[x])$
*   함수 $f(x)$가 **오목(concave)**할 때: $\mathbb{E}[f(x)] \le f(\mathbb{E}[x])$

여기서 우리는 $-\log(x)$ 함수를 사용합니다. 로그 함수($\log x$)는 오목 함수이지만, 마이너스가 붙은 $-\log x$는 아래로 볼록한 **볼록 함수**입니다. 따라서 젠센 부등식을 적용할 수 있습니다.

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

### 2.4 생성 과정 (The Generative Process)
VAE는 데이터 $x$가 잠재 변수 $z$로부터 생성된다고 가정합니다:
1.  **Prior**: $z \sim p_\theta(z)$ (보통 표준 정규분포)
2.  **Likelihood**: $x \sim p_\theta(x|z)$ (디코더 신경망)

우리는 $p_\theta(z|x)$를 알고 싶지만, 앞서 언급했듯 $p(x)$를 구할 수 없어 계산이 불가능합니다.

### 2.5 변분 추론 (Variational Inference)
계산 불가능한 $p_\theta(z|x)$ 대신, 다루기 쉬운 근사 분포 $q_\phi(z|x)$(인코더)를 도입합니다. 목표는 $q_\phi$를 $p_\theta$에 최대한 가깝게 만드는 것, 즉 $D_{KL}(q_\phi || p_\theta)$를 최소화하는 것입니다.

### 2.6 Evidence Lower Bound (ELBO) 상세 유도
우리는 $\log p_\theta(x)$를 최대화하고 싶습니다. 식을 변형해 봅시다.

$$
\begin{aligned}
\log p_\theta(x) &= \log \int p_\theta(x, z) dz \\
&= \log \int p_\theta(x, z) \frac{q_\phi(z|x)}{q_\phi(z|x)} dz \\
&= \log \mathbb{E}_{q_\phi} \left[ \frac{p_\theta(x, z)}{q_\phi(z|x)} \right] \\
&\ge \mathbb{E}_{q_\phi} \left[ \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \right] \quad (\text{Jensen Inequality}) \\
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
&= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
\end{aligned}
$$
*   첫 번째 항: **Reconstruction Error**
*   두 번째 항: **Regularization** (Prior $p(z)$와 $q_\phi$의 차이)

### 2.7 재파라미터화 트릭 (The Reparameterization Trick)
Backpropagation을 가능하게 하기 위해, 무작위성을 입력 레이어로 분리합니다.

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

이제 $z$는 $\mu$와 $\sigma$에 대한 결정론적 함수가 되어 미분이 가능해집니다.

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
