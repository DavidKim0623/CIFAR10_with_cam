#os : 운영체제 활용(시스템 활용), random(랜덤 시드 배정)
import os, random
#argparse: argument parsing하여 옵션 적용하기 위함
import numpy as np

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

#wandb 라이브러리를 추가
import wandb

#랜덤 시드를 저장하기 위한 옵션 함수
def set_seed(seed = 42):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministric = True
	torch.backends.cudnn.benchmark = False

#정규화 되어있던 이미지를 복원하는 함수
def denormalize(img, mean, std):
	#torch.tensor(대상).view(채널) -> '채널' 사이즈로 복원.
	#텐서.view(3, 1, 1) => 텐서를 3, 1, 1(C, H, W)로 브로드캐스팅
	mean = torch.tensor(mean).view(3, 1, 1)
	std = torch.tensor(std).view(3, 1, 1)
	return img * std + mean #이미지를 원래 방식으로 복원

#모델 클래스
class SimpleCIFAR10CNN(nn.Module):
	#모델의 레이 등 '필요한' 컴포넌트를 정의
	#num_classes = 10은 CIFAR10의 클래스 수가 10개이므로
	def __init__(self, num_classes = 10):
		super(SimpleCIFAR10CNN, self).__init__()
		#네트워크의 레이어를 자동으로 셋업하여 흐르게 할 수 있도록 'sequential'을 사용하여 조립
		#특징 추출부(features), 크기 정리 및 특징 요약(gap), 10개의 클래스 라벨에 대한 분류(classifier)
		self.features = nn.Sequential(
			#(128, 3, 32, 32)
			nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			#pooling Layer인데, Max값(최대값)기준 pooling -> pooling함으로써 파라미터 사이즈 축소 + Max값 특징 극대화
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2)
			) # 최종적으로, 128채널의 8, 8 짜리 output이 나오게 됨. -> (B, C, H, W)가 (128, 128, 8, 8)

		#max pooling(정해진 구역에서 max 값 추출), avg pooling(정해진 구역에서 avg값 추출)
		#Adaptive Pooling -> 최종 output의 크기를 정해놓고, 여기에 맞춰 특징을 줄이는 것
		self.gap = nn.AdaptiveAvgPool2d((1, 1)) # 채널별로 (1, 1)짜리 output을 가지겠어. -> (B, 128, 1, 1)

		#(B, 128*1*1) -> 1개의 이미지당, 128개의 특징 
		self.classifier = nn.Linear(128, num_classes)

		#콘볼루셔널 저장하는 코드 -> 가중치를 마지막에 저장해서 cam을 그리기 위함
		self.last_conv_features = None

	#x는 배치치 사이즈 ?개의 3 채널 이미지 데이터	
	def forward(self, x):
		x = self.features(x)
		self.last_conv_features = x #features를 추출한 내용(가중치)을 last_conv_features에 저장
		x = self.gap(x) #(B, 128, 1, 1) -> (B, 128*1*1)로 변형하기 위한 레이어가 필요
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

import custom

#데이터를 구성하기 위한 함수
def get_dataloader(
	#데이터를 다운로드 후 저장할 위치
	data_dir,
	#배치 사이즈
	batch_size = 128, 
	#병렬 작업을 수행할 크기
	num_workers = 2):

	#이미지 정규화를 위한 채널별 평균값, 표준편차값
	mean = (0.4914, 0.4822, 0.4465)
	std = (0.2470, 0.2435, 0.2616)

	#CIFAR10을 다운로드 해서, 개별 이미지에 적용할 전처리 묶음
	train_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
		transforms.Lambda(custom.add_noise_wrapper)
		])

	test_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
		])

	#pytorch 내부의 dataset을 다운로드
	full_train_set = torchvision.datasets.CIFAR10(
		root = data_dir,		#데이터를 다운로드 받아 어느 위치에 저장할지,
		train = True,		#train 데이터셋을 다운로드 할 여부 
		download = True,	#download로 진행할 여부(혹은 스트리밍 할 여부)
		transform = train_transform 	#준비한 데이터셋ㅇ[ 어떤 전처리 묶음 적용할 것인지
		)

	#CIFAR10은 train 5만개, test 만개 -> 이 중 train 5만개를 다운로드함(full_train_set)
	#full_train_set을 2등분함 -> train, valid
	num_train = len(full_train_set) #5만개
	indices = list(range(num_train)) #전체 길이만큼의 리스트 생성
	split = int(num_train * 0.1) #전체의 10%만큼을 split하기 위해 split 변수명에 비율 저장

	#랜덤으로 우선 한번 섞고(suffle) 정해진 길이(split)만큼 나눔
	random.shuffle(indices)

	val_indices = indices[:split]	#전체[:10]
	train_indices = indices[split:]	#전체[10:]

	train_sampler = SubsetRandomSampler(train_indices)
	val_sampler = SubsetRandomSampler(val_indices)

	#Subset으로 추출한 데이터셋을 실제 훈련 가능한 배치 사이즈 크기로 잘라줌
	train_loader = DataLoader(
		full_train_set,
		batch_size = batch_size,
		sampler = train_sampler,
		num_workers = num_workers,
		pin_memory = True
		)
	val_loader = DataLoader(
		full_train_set,
		batch_size = batch_size,
		sampler = val_sampler,
		num_workers = num_workers,
		pin_memory = True
		)

	#테스트 셋을 넣어줌
	test_set = torchvision.datasets.CIFAR10(
		root = data_dir,
		train = False,
		download = True,
		transform = test_transform
		)

	#테스트도 동일하게 배치사이즈 세팅
	test_loader = DataLoader(
		test_set,
		batch_size = batch_size,
		shuffle = False,
		num_workers = num_workers,
		pin_memory = True
		)

	return train_loader, val_loader, test_loader 

from tqdm import tqdm	

#한 번의 훈련 에포크를 수행하는 함수
def train_one_epoch(model, loader, criterion, optimizer, device, cur_epoch):

	#모델을 '훈련 모드'로 전환하고, 훈련에 대한 지표를 초기화함
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	#inputs = 이미지, targets = 라벨(클래스에 대한 인덱스)
	for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc = f'Epoch {cur_epoch}...')):
		#계산을 위해 device(일반적으로는 gpu)의 메모리로 inputs, targets을 보내는 것
		inputs = inputs.to(device)
		targets = targets.to(device)

		#역전파를 위한 gradient 초기화
		optimizer.zero_grad()

		#순전파
		outputs = model(inputs)

		#손실함수 계산
		loss = criterion(outputs, targets) #타겟(실제 라벨)과 순전파 결과를 비교하여 손실 계산
		loss.backward()
		optimizer.step()

		#손실을 더해 히스토리에 남김
		running_loss += loss.item() * inputs.size(0)

		_, predicted = outputs.max(1) #가장 '높은 확률로' 예측한 한 자리 수를 리턴
		total += targets.size(0) #이번 사이클에 들어온 '문제'의 수를 더해 총 개수를 구함
		correct += predicted.eq(targets).sum().item()

	avg_loss = running_loss / total
	acc = 100.0 * correct / total
	print(f'Train Epochs {cur_epoch} | Loss : {avg_loss : .4f} | Acc : {acc : .2f}%')

	return avg_loss, acc


#평가를 수행하는 함수
def evalute(model, loader, criterion, device, mode = 'Test'):

	#모델을 '평가 모드'로 전환
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):

			#계산을 위해 device(일반적으로는 gpu)의 메모리로 inputs, targets을 보내는 것
			inputs = inputs.to(device)
			targets = targets.to(device)

			outputs = model(inputs)
			loss = criterion(outputs, targets)

			#전체 loss와 정확도 측정을 위해
			running_loss += loss.item() * inputs.size(0)
			_, predicted = outputs.max(1)

			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

		avg_loss = running_loss / total
		acc = 100.0 * correct / total

		print(f'[{mode}] | Loss {avg_loss:.4f} | Acc {acc:.2f}%')

	return avg_loss, acc

		
#전체 구성-훈련-평가를 진행하는 함수
#데이터셋 구성 -> 훈련 -> 평가 과정 필요 => 'main'함수에서 순차적으로 발생
def main():
	parser = argparse.ArgumentParser(description = 'CIFA10 CNN Training Parsing')
	#python ./main.py --data_dir
	parser.add_argument('--data_dir', type = str, default = './data', help = 'CIFA10이미지 저장 경로')
	parser.add_argument('--epochs', type = int, default = 100, help = '학습 epochs 수')
	parser.add_argument('--batch_size', type = int, default = 128, help = '배치 크기')
	parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
	parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'L2 weight_decay')
	parser.add_argument('--seed', type = int, default = 42, help = 'random seed')
	parser.add_argument('--save_dir', type = str, default = './checkpoint', help = '모델 저장 경로')

	#아규먼트 활용
	args, _ = parser.parse_known_args()


	#1.훈련을 위한 옵션 및 장치 세팅
	#랜덤 시드 세팅
	set_seed(args.seed)

	#훈련 결가를 저장할 디렉토리 생성
	os.makedirs(args.save_dir, exist_ok = True) #exist_ok = 이미 생성되어 있어어도 괜찮냐?

	#디바이스
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Using device : {device}')

	#2.모델과 데이터셋 로드
	train_loader, val_loader, test_loader = get_dataloader(data_dir = args.data_dir,
		batch_size = args.batch_size,
		num_workers = 2)

	#모델
	model = SimpleCIFAR10CNN(num_classes = 10).to(device)

	#최적화 함수, 손실함수 정의
	criterion = nn.CrossEntropyLoss() #분류이기 때문에 크로스 엔트로피 손실 함수를 씀
	optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

	#--------------- wandb 세팅 ------------------------#
	wandb.init(
		#프로젝트 CIFAR10 / #실행 이름: 실험을 할 때마다다 정해주는 이름
		project = 'CIFAR10_CNN',	#프로젝트 이름
		config = vars(args),	#하이퍼파라미터 셋팅
		name = '1try',		#실행 이름
		reinit = True	#재초기화 가능한가?
	)
	#---------------------------------------------------#

	#3.실제 훈련
	#4.실제 평가
	#훈련과 평가 관련된 지표 초기화
	starts = 0 
	best_acc = 0
	best_model_path = os.path.join(args.save_dir, 'cifar10_cnn_best.pth') #저장위치 폴더에 'cifar10_cnn_best.path' 이름으로 pt파일 생성

	#가장 마지막에 훈련한 가중치를 save
	latest_model_path = os.path.join(args.save_dir, 'cifar10_cnn_latest.pth')

	if os.path.exists(latest_model_path):
		#기존에 훈련 이력이 있는 경우, checkpoint를 처음부터 시작하지 말고,
		#이전 훈련의 가장 마지막 결과를 가져다가 쓰세요!
		#map_location -> 어디에다가 이 데이터를 보내놓을까요? -> device라는 옵션에 맞춰 보내놓으세요.
		checkpoint = torch.load(latest_model_path, map_location = device)

		starts = checkpoint['epoch'] + 1
		best_acc = checkpoint['best_acc']


	for epoch in range(starts, args.epochs + 1):

		torch.save({
			'epoch' : epoch,
			'model_state_dict' : model.state_dict(),
			'best_acc' : best_acc
			}, latest_model_path)

		# 여기서 이번 epoch에 대한 1회 훈련이 진행됨
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

		#여기서 이번 epoch에 대한 평가가 진행됨
		val_loss, val_acc = evalute(model, val_loader, criterion, device, mode = 'validation')

		#Best한 가중치 파일을 저장하는 프로세스 생성
		if val_acc > best_acc:
			#1에포크 -> best, ....나이지는 지점 -> best로 교체
			best_acc = val_acc

			#해당하는 스텝에서 저장!
			torch.save({
				'epoch':epoch,
				'model_state_dict':model.state_dict(),
				'optimizer_state_dict':optimizer.state_dict(),
				'best_acc':best_acc
				}, best_model_path)

			print(f'New Best Model Saved! Val Accuracy: {best_acc:.2f}')

		#artifact 세팅: 실험 관리 단위(프로젝트랑 동일)
		artifact = wandb.artifact('CIFAR10_CNN_BEST', type = 'model') #model <-> AI
		artifact.add_file(best_model_path)
		wandb.log_artifact(artifact)



	#모든 에포크가 종료된 이후, validaation 기준으로 가장 높은 acc보인 결과물을 save, 위치를 알림			
	print(f'Training finished. Best validation Acc {best_acc:.2f}%')
	print(f'Best model path: {best_model_path}')

	#저장 후 재사용
	#저장한 가중치를 '불러옴' -> 지금 사용할거니까 gpu로 업로드(device로 보냄)
	checkpoint = torch.load(best_model_path, map_location = device)

	#가져온 가중치를 gpu에 있느는 모델에 연결
	model.load_state_dict(checkpoint['model_state_dict'])

	print('Evaluating Starts...')
	evalute(model, test_loader, criterion, device, mode = 'Test')

# 파이썬 파일의 시작점(진입점)
if __name__ == '__main__':
	main()