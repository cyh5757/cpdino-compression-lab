# 출력 지식 증류와 소형 Vision Transformer 백본 이식을 통한 CPDINO 경량화

## 초록

세포 인스턴스 분할 모델을 경량화하려면 모델 규모 감소에 따른 분할 정확도와 추론 비용의 변화를 함께 평가해야 한다. 본 연구는 DINOv3 기반 CPDINO에 출력 지식 증류와 소형 백본 이식을 적용한다. 첫 번째 실험에서는 CPDINO ViT-L을 teacher로 사용하고, 직접 미세조정한 ViT-B student의 flow 및 foreground logit에 대한 회귀 증류를 수행한다. 두 번째 실험에서는 공식 DINOv3 ViT-S 인코더에 CPDINO 출력 헤드를 연결하여 지도 미세조정한다. 학습, 검증 및 test 데이터는 각각 517장, 64장, 64장으로 구성된다. 기존 실험의 test 집계에서 증류 ViT-B는 AP@0.5 0.8212, AP@0.75 0.7396을 기록하였으며, 이미지당 평균 지연시간은 806.4 ms였다. ViT-S의 최종 test 결과는 기록 시점에 확보되지 않았다. 본 문서는 두 학습 방법의 구성과 관측 결과를 기술하고, 초기화 조건 및 측정 절차가 결과 해석에 미치는 영향을 논의한다.

## 1. 서론

CPDINO는 DINOv3의 공간 특징을 이용하여 세포의 flow와 foreground logit을 예측하고, Cellpose 후처리를 통해 인스턴스 마스크를 생성한다. 백본의 규모를 줄이면 파라미터 수와 주요 행렬 연산량이 감소하지만, 그에 따른 분할 품질의 변화는 데이터와 학습 조건에 의존한다.

본 연구의 목적은 ViT-L 기반 분할 모델을 기준으로 ViT-B의 추가 지식 증류와 ViT-S의 백본 이식을 분석하는 것이다. ViT-B에서는 아키텍처를 고정한 상태에서 teacher의 출력 정보를 추가하고, ViT-S에서는 공간 출력 형식을 유지한 상태에서 백본의 특징 차원을 축소한다. 두 실험의 초기화와 목적 함수를 명시하여 모델 규모와 학습 방법을 구분하여 해석한다.

## 2. 모델 구성

### 2.1. Transformer 백본

ViT-L, ViT-B 및 ViT-S의 블록 수는 각각 24, 12, 12이며, embedding dimension은 각각 1,024, 768, 384이다. Attention head 수는 16, 12, 6이고 head당 차원은 모두 64이다. 각 블록은 정규화, multi-head self-attention, 잔차 연결 및 MLP로 구성되며, 입력과 출력의 토큰 수 및 embedding dimension을 유지한다. MLP의 중간 차원은 embedding dimension의 4배이다.

백본의 파라미터 수는 ViT-L 303,154,176개, ViT-B 85,669,632개, ViT-S 21,601,152개이다. ViT-S는 ViT-B와 동일한 깊이를 유지하면서 특징 차원을 절반으로 축소한다. 고정된 MLP 확장 비율에서 주요 투영 행렬의 파라미터 수가 특징 차원의 제곱에 비례하므로, ViT-S의 백본 파라미터 수는 ViT-B의 약 25.2%이다.

### 2.2. CPDINO 공간 출력

패치 투영의 커널 크기는 16×16이며, stride 8과 padding 4를 사용한다. 입력 타일이 384×384인 경우 48×48 공간 격자를 생성한다. 공간 토큰 2,304개에 CLS 토큰 1개와 register 토큰 4개를 추가한 후 Transformer에 입력한다. 최종 LayerNorm 이후 보조 토큰을 제거하고 공간 토큰에 선형 출력 헤드를 적용한다.

출력 헤드의 입력 차원은 백본의 embedding dimension과 같으며, 출력 차원은 세 모델 모두 192이다. 고정 가중치 `W2`는 각 토큰의 192개 출력을 8×8×3으로 재배열한다. 이에 따라 최종 출력은 입력과 동일한 공간 해상도의 3채널 텐서가 된다. 채널은 수직 flow, 수평 flow 및 foreground logit에 대응한다. 인스턴스 마스크의 구성은 Cellpose의 flow dynamics와 후처리 설정을 따른다.

### 2.3. ViT-S 초기화와 학습 파라미터

ViT-S 인코더는 공식 `facebook/dinov3-vits16-pretrain-lvd1689m` 가중치로 초기화한다. 선형 출력 헤드 `Linear(384, 192)`는 새로 초기화한다. Hugging Face 형식의 가중치는 키 이름, Q/K/V 배열과 bias 표현을 DINOv3 구현에 맞게 변환한 뒤, 전체 텐서의 키와 형태를 검사하여 로드한다.

원본 CPDINO forward는 패치 투영에 `.data`를 사용하여 해당 가중치의 gradient를 차단한다. 공개 구현은 동일한 동작을 `requires_grad=False`로 명시한다. Transformer 블록과 출력 헤드는 최적화 대상이며, `W2`와 직경 관련 파라미터는 고정한다. Mask token은 이 분할 forward 경로에서 사용하지 않는다.

## 3. 학습 방법

### 3.1. Cellpose 지도 손실

Student 출력은 $p=(p_f,p_c)$로 표기한다. $p_f$는 두 flow 채널이며 $p_c$는 foreground logit이다. 라벨 텐서에서 instance ID 채널을 제거한 뒤 foreground와 두 flow 채널을 사용한다. Foreground 라벨은 0.5를 기준으로 이진화한다. 지도 손실은 다음과 같다.

$$
\mathcal{L}_{\mathrm{task}}=\frac{1}{2}\mathrm{MSE}(p_f,5y_f)+\mathrm{BCEWithLogits}(p_c,y_c).
$$

Flow 라벨의 배율 5와 MSE 계수 1/2는 원본 학습 코드의 설정을 유지한다. 각 손실 항은 해당 텐서 원소에 대해 평균한다.

### 3.2. ViT-L에서 ViT-B로의 출력 증류

Teacher는 Pro_ver2 CPDINO ViT-L 체크포인트로 초기화하고 모든 파라미터를 고정한다. Student는 분할 데이터로 직접 미세조정한 CPDINO ViT-B 체크포인트에서 초기화한다. 동일한 증강 입력에 대해 teacher와 student의 출력을 계산하고, 지도 손실에 다음 증류 항을 추가한다.

$$
\mathcal{L}_{\mathrm{KD}}=\mathrm{MSE}(p_f,\mathrm{stopgrad}(t_f))+\mathrm{MSE}(p_c,\mathrm{stopgrad}(t_c)).
$$

전체 학습 손실은 다음과 같으며, $\lambda=1$로 설정한다.

$$
\mathcal{L}=\mathcal{L}_{\mathrm{task}}+\lambda\mathcal{L}_{\mathrm{KD}}
$$

Foreground에 대한 증류는 sigmoid 적용 전 logit을 회귀 대상으로 사용한다. Teacher는 평가 모드 및 `no_grad()`에서 실행한다. 두 모델의 내부 특징 차원은 달라도 최종 출력 형태가 동일하므로 별도의 특징 투영 없이 출력 손실을 계산할 수 있다.

### 3.3. ViT-S 지도 미세조정

ViT-S는 사전학습 인코더와 새 CPDINO 출력 헤드에 대해 지도 손실만 적용한다. 실험 설정은 `teacher=None`, `teacher_weight=0`이다. 따라서 이 실험은 백본 이식 후 지도 미세조정에 해당한다. Meta의 공식 DINOv3 가중치에 적용된 사전학습 단계의 증류와 본 데이터에 대한 출력 증류는 서로 다른 학습 단계이다.

## 4. 실험 설정

학습, 검증 및 test split은 각각 517장, 64장, 64장으로 구성된다. 학습·검증 배치는 Cellpose 정규화와 회전·크기 변환을 적용하여 384×384 타일로 생성한다. 크기 변환 범위는 `scale_range=0.5`이다. 이미지 및 flow 파일의 대응 관계와 split 간 파일명 중복을 검사한다.

두 실험 모두 AdamW, learning rate 1e-5, weight decay 0.1 및 seed 20261112를 사용한다. 목표 학습 길이는 500 epochs이며, batch size는 ViT-B 13, ViT-S 29이다. 첫 epoch와 이후 10 epochs마다 검증을 수행한다. 검증에는 teacher 출력 손실을 포함하지 않고 지도 손실만 사용한다. 원본 학습기는 검증 손실이 감소한 시점의 모델을 `best.pt`로 저장한다.

Test 추론은 NVIDIA GeForce RTX 5090에서 PyTorch BF16으로 수행하였다. Batch size는 8, 타일 크기는 384, tile overlap은 0.1이다. Cell probability threshold는 -1.0, flow threshold는 0.4, 최소 마스크 크기는 15로 설정하였다. 정규화와 resampling을 적용하고 test-time augmentation은 사용하지 않았다. 전체 설정은 [평가 기록](../results/historical-summary.json)에 수록하였다.

## 5. 결과

표 1은 2026년 11월 13일 16:52 KST에 수록한 기존 실험의 test 집계 결과이다.

| 모델 | AP@0.5 | AP@0.75 | 세포 수 MAE | 평균 이미지 지연시간 (ms) |
|---|---:|---:|---:|---:|
| CPDINO ViT-L / Pro_ver2 | 0.8246 | 0.7443 | 27.5469 | 1,265.5 |
| 직접 미세조정 ViT-B | 0.8201 | 0.7322 | 25.2656 | 843.7 |
| 증류 ViT-B | 0.8212 | 0.7396 | 26.6719 | 806.4 |

AP는 해당 IoU 임계값에서 Cellpose가 계산하는 `TP / (TP + FP + FN)`의 이미지별 평균이다. 세포 수 MAE는 예측 인스턴스 수와 라벨 인스턴스 수의 절대 오차 평균이다. 지연시간은 각 이미지에 대해 3회 측정한 중앙값을 사용한 뒤 전체 이미지에 대해 평균하였다. 디스크 읽기와 모델 로딩 시간은 측정에서 제외하였다.

증류 ViT-B의 AP@0.5와 AP@0.75는 직접 미세조정 ViT-B보다 각각 0.0011과 0.0074 높게 관측되었다. 반면 세포 수 MAE는 25.2656에서 26.6719로 증가하였다. 증류 ViT-B의 AP@0.75는 ViT-L보다 0.0047 낮았다. 이 결과는 평가 지표에 따라 변화의 방향과 크기가 다름을 보여준다.

수록 시점의 ViT-S 학습 기록은 317/500 epochs이며, 최저 검증 손실은 310 epoch의 0.14115이다. 최종 test 결과가 없으므로 ViT-S의 정확도 및 지연시간은 표 1의 비교에 포함하지 않았다. 이 학습 상태는 수록 시점의 스냅샷이며 실시간 진행 상태를 나타내지 않는다.

## 6. 논의

ViT-B의 추가 증류에서 AP@0.75의 증가가 관측되었으나, 단일 데이터 분할과 학습 실행에 대한 결과이므로 효과의 통계적 유의성을 평가할 수 없다. 세포 수 MAE의 증가도 함께 고려해야 한다. 일부 평가 라벨에는 모델 기반 생성 이력이 있어 독립적인 수동 주석을 기준으로 한 추가 평가가 필요하다.

직접 미세조정 ViT-B와 증류 ViT-B는 동일한 아키텍처를 사용한다. GPU 동시 작업 통제와 반복 실험의 분산이 기록되지 않았으므로, 두 모델 사이의 지연시간 차이를 증류에 따른 연산량 감소로 해석할 근거는 부족하다. 백본 규모에 따른 속도 차이를 정량화하려면 동일한 실행 조건에서 반복 측정해야 한다.

ViT-B는 기존 세포 분할 체크포인트에서 출발하고, ViT-S는 일반 시각 사전학습 인코더와 새 출력 헤드에서 출발한다. 따라서 두 모델의 성능 차이는 백본 크기뿐 아니라 초기화 및 분할 학습 이력의 영향을 포함한다. 아키텍처의 독립적인 효과를 평가하려면 초기화 조건과 학습 예산을 통제한 비교가 필요하다.

## 7. 구현 검증과 재현 범위

공개 구현은 원본의 임시 전역 생성 함수 교체를 인코더 주입 방식으로 변경하고 CPDINO의 forward를 상속한다. 가중치 파일의 SHA-256, state dict 키 및 텐서 형태를 검사한다. 모델, optimizer와 난수 상태를 저장하여 epoch 경계에서 재개하며, 재개 시 실험 설정 및 데이터 식별자의 일치를 확인한다.

CPU 단위 테스트는 출력 텐서 규격, 손실 값, student 파라미터 갱신, teacher gradient 차단, 고정 패치 투영, 검증 집계 및 체크포인트 복원을 포함한다. 원본 ViT-S와 공개 구현의 동일 seed 비교에서 state dict가 일치하였고, 합성 입력 `[1, 3, 64, 64]`에 대한 학습·평가 모드의 예측 차이는 0이었다. 비교에 사용한 지도·증류 손실과 예측에 대한 gradient도 일치하였다. 해당 검증은 프로그램의 동작 일치를 검사하며 분할 정확도에 대한 평가를 대체하지 않는다.

공개 모듈로 500 epoch 전체 학습을 재실행하지는 않았다. 공개 노트북은 `last.pt` 저장과 재개를 제공하며, 원본 학습기의 validation 기반 `best.pt` 자동 선정은 포함하지 않는다. 표 1은 원본 실험 산출물에서 읽은 결과이고, 공개 모듈로 재학습한 결과는 별도의 실행 기록이 필요하다. 상세 검증 범위는 [verification.md](verification.md)에 기술한다.

## 8. 결론

본 연구는 CPDINO의 공통 공간 출력 구조를 유지하면서 ViT-B 출력 증류와 ViT-S 백본 이식을 구성하였다. ViT-B 증류에서는 직접 미세조정 대비 AP 증가와 세포 수 MAE 증가가 함께 관측되었다. ViT-S는 백본 파라미터를 ViT-B의 약 25.2%로 축소하며, 최종 정확도와 추론 비용에 대한 평가는 별도로 필요하다. 공개 코드는 두 방법의 모델 구성, 학습 목적 함수 및 실행 절차를 모듈과 노트북으로 제공한다.

## 참고 구현

1. MouseLand. [Cellpose CPDINO 구현](https://github.com/MouseLand/cellpose/blob/main/cellpose/vit.py). 실험 의존성 버전 4.2.1.1.
2. Meta. [DINOv3](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751). 실험 의존성 커밋 `6876159a11b4df116f30f667f8c9888617df0751`.
3. Meta. [DINOv3 ViT-S/16 LVD-1689M 사전학습 가중치](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).
