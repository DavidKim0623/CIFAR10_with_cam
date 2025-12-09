#os : 운영체제 활용(시스템 활용), random(랜덤 시드 배정)
import os, random
import argparse

import torch #pytorch 임포트
import torch.nn as nn #신경망 클래스
import torch.optim as optim #최적화 클래스
import torch.nn.functional as F #펑셔널 클래스(레이어 추가)
from torch.utils.data import DataLoader, SubsetRandomSampler
#DataLoader : pytorch 자체적으로 가지고 있는 데이터를 로딩
#Subset Random Sampler : 데이터셋의 일부를 train, valid, test로 나누기 위함
import torchvision #Vision 데이터(이미지, 영상)활용
import torchvision.transforms as transforms #Vision 데이터(이미지, 영상)

import matplotlib.pyplot as plt #시각화 라이브러리
import wandb #wandb 라이브러리를 추가

#커스터마이징한 함수
import custom
import main

#cam을 생성하는 함수
#model = 훈련이 끝난 모델
#input_tensor = 데이터셋에서 들어오는 1개의 이미지 데이터
#name = 이름
#class_idx = 클래스 인덱스
def genernate_cam(model, input_tensor, name, class_idx = None):
	#모델을 평가 구조로 전환 -> 가중치 업데이트를 하지 않겠다.
	model.eval()

	#index가 None일 때는, input_tensor를 model에 돌려서 예측된 class를 사용함
	with torch.no_gard():
		#outputs = model(input_tensor) -> 순전파(model.train()) or 추론(model.eval())
		#logits 확률 -> 라벨(클래스)에 대한 확률
		logits = model(input_tensor)
		if class_idx is None:
			#logits을 통해 퍼센트를 계산하고,
			#가장 높은 가능성을 가진 한 개(dim = 1)의 인덱스 숫자(.item())을 가져옴
			class_idx = logits.argmax(dim = 1).item()

	#만약 class_idx가 None이 아닌 경우
	if isinstance(class_idx, torch.Tensor):
		class_idx = class_idx.item()

	#구현 사항: model.last_conv_features를 가져옴(1, 128, 8, 8)
	conv_feature = model.last_conv_features
	conv_feature = conv_feature.squeeze(0) #(1, 128, 8, 8)의 0번 채널을 없애고, (128, 8, 8)로 축소함

	#model.classifier.weight[class_idx]로 class weight의 성분을 가져옴
	class_weight = model.classifier.weight[class_idx]

	#con_features의 128개의 feature에 해당하는 class_weight 성분을
	#각 8 * 8 heatmap에 곱해서 다 더해줌(더하는 건 곱한 다음에 feature차원 방향으로) -> 8 * 8
	cam_base = torch.zeros(8, 8, dtype = conv_feature.dtype)

	#128개 채널의 길이만큼
	for c in range(conv_feature.shape[0]):
		cam_base += class_weight[c] * conv_feature[c]

	#nn.ReLu <-> F.relu
	cam_base = F.relu(cam_base)

	#cam값을 0~1사이로 min-max 정규화
	cam_base_min, cam_base_max = cam_base.min(), cam_base.max()
	cam_result = (cam_base - cam_base_min)

	#interpolate (8, 8) -> (32, 32)사이즈로 업스케일 하기 위해 interpolate(보간)
	cam__draw = F.interpolate(cam_result, size = (32, 32), mode = 'bilinear')

	#다시 2차원으로 축소
	#(1, 128, 8, 8) - > (32, 32)
	cam_draw = cam_draw.squeeze(0).squeeze(0)

	draw_graph(input_tensor, cam_draw, name)

def denormalize(image, mean, std):
	return image * std + mean

#input_tensor(데이터셋의 그림 데이터 1장), cam(가중치에서 주목한 곳), name(이름), alpha(투명도)
def draw_graph(input_tensor, cam, name, alpha = 0.5):
	#인풋 텐서의 차원 적용 및 정규화
	#cpu로 옮기고 차원 재배치, 복원
	input_tensor = input_tensor[0]
	image = input_tensor.detach().cpu().permute(1, 2, 0).numpy().astype('float32')

	mean = (0.4914, 0.4822, 0.4465)
	mean = np.array(mean, dtype = 'float32')

	std = (0.2470, 0.2435, 0.2616)
	std = np.array(std, dtype = 'float32')

	image = denormalize(image, mean, std)

	#cam의 numpy변환
	cam = cam.detach().cpu().numpy()

	#그려주는 코드
	plt.figure(figsize = (4, 4))
	plt.imshow(image)
	plt.imshow(cam, camp = 'bwr', alpha = alpha)
	plt.axis('off')
	plt.savefig(f'./{name}.png', transparent = True)