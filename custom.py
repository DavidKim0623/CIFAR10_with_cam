import cv2
import numpy as np
from PIL import Image

import random

import torch
from torchvision import transforms

#get_dataloader 함수에서 data를 가져 올 때 transform.Compose()
#transforms.ToTensor(), transforms.Normalize()
#이거 말고, 데이터셋에 '강건성(robustness)'을 부여하기 위해 '증강'을 수행함
#랜덤 크롭(일부를 잘라줌(crop))

#솔트 앤 페퍼 노이즈(흰색, 검은색 픽셀 노이즈)
def salt_and_pepper(image, ratio = 0.01):
	#원본 이미지를 손상시키지 않도록 카피함
	image = image.copy() 

	#원본 이미지의 크기를 미리 기억함
	W, H, C = image.shape

	#솔트/페퍼 노이즈의 비율 반반
	sp = 0.5
	amount = ratio

	#솔트 적용(흰 점 노이즈)
	#흰 점(255)을 만들기 위해 일부 픽셀을 변경
	salt =  np.ceil(amount * W * H * sp).astype(int)
	#image.shape[:-1] 너비, 높이 채널에 대해서 존재하난 i(픽셀)에 random으로 salt를 적용
	random_salt = [np.random.randint(0, i, salt) for i in image.shape[:-1]]
	image[random_salt[0], random_salt[1], :] = 255 #흰 점이어야 하기 때문에(255)

	#페퍼 적용(검은 점 노이즈)
	#검은 점(0)을 만들기 위해 일부 픽셀을 변경
	salt =  np.ceil(amount * W * H * sp).astype(int)
	#image.shape[:-1] 너비, 높이 채널에 대해서 존재하난 i(픽셀)에 random으로 salt를 적용
	random_salt = [np.random.randint(0, i, salt) for i in image.shape[:-1]]
	image[random_salt[0], random_salt[1], :] = 255 #흰 점이어야 하기 때문에(255)
	return image

#HSV 채널 변경(조명, 색상 이슈를 해소하기 위해 적용)
def augmented_hsv(image, h = 0.5, s = 0.5, v = 0.5):
	#변형을 하고 싶은 경우
	if h or s or v:
		#np.random.uniform : 최소값 ~ 최대값 사이의 균일 분포에서 무작위로 숫자(random) 세팅
		#np.random.uniform(최소, 최대, 생성숫자) : -1 ~ 1까지의 수 중에서 3개를 선택 -> 이 세개를 h, s, v와 곱 +1
		#(랜덤 1 * h) + 1, (랜덤 2 * s) + 1, (랜덤 3 * v) + 
		r = np.random.uniform(-1, 1, 3) * (h, s, v) + 1

		#실제 이미지의 hsv 채널 분리
		hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
		dtype = image.dtype

		#0에서 255까지 값을 가진 x생성
		x = no.arange(0, 255, dtype = r.dtype)

		#H, S, V 각각에 대해 랜덤 계수를 곱하여
		#.astype(dtype) -> 원래 이미지가 가지고 있는 dtype으로 타입 변경(astype)
		lut_h = ((x * r[0]) % 180).astype(dtype) #Hue 범위(0~179)에 맞게 % 180으로 나머지 연산

		#x * r[1] 한 값을 0에서 255사이로 clip처리(하한이나 상한선을 넘지 않게 처리)한 후 dtype 연산
		lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
		lut_v = np.clip(x * r[2], 0, 255).astype(dtype)

		hsv_image = cv2.merge((cv2.LUT(hue, lut_h), (cv2.LUT(sat, lut_s), (cv2.LUT(val, lut_v)))))

		cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR, dst = image)

#클래스로 crop을 적용
#crop(자름) -> 어디에서 어디까지 자를거야?(범위설정) -> 이미지의 사이즈를 기억
class RandomCrop:
	def __init__(self, size, padding = 4):
		# 결과1 if 조건 else 결과2 : '조건'에 맞으면 결과1, 그렇지 않으면 결과2
		# 이 코드에서는 조건(isinstance)에 맞으면 (size, size)의 튜플 상태로 만들어 self.size에 저장해둠
		self.size = (size, size) if isinstance(size, int) else size
		self.padding = padding

	#이 클래스를 '함수처럼' 호출 가능해짐
	def __call__(self, image):
		if isinstance(image, torch.Tensor):
			#transforms -> 이미지 전처리 transforms.Compose([])
			#ToPILImage : 텐서 형태나, ndarray 형태의 이미지를 PIL라이브러리가 활용 가능한 형태로 만듦
			image = transforms.ToPILImage()

		#이지지 넘파이 변환 및 크롭 사이즈 세팅
		np_image = np.array(image)
		H, W = np_image.shape[:2] #H, W, C 중에서 H, W만 가져오기 위해 [:2]을 진행
		crop_h, crop_w = self.size

		#패딩이 0이 아닌 경우
		if self.padding > 0:				#위쪽, 아래쪽					왼쪽, 오른쪽	
			np_padding = np.pad(np_image, ((self.padding, self.padding), (self.padding, self.padding)), mode = 'constant') #고정된 값(상수)으로 패딩 부분 채우기
			#경계 값을 그대로 유지(edge), 자연스럽게 경계만들기(reflect)

			image_h, image_w = np_padding.shape[:2]
			np_image = np_padding

		#패딩을 이미지에 적용하지 않은 경우
		#크롭을 적용하지 않고 그냥 return 해줌!
		if H == image_h and W == image_w :
			return Image.fromarray(np_image)

		#크롭을 적용
		#image_h - H => 패딩을 한(넓어진) 길이 - 원래 이미지 길이
		#(0에서, 패딩한 만큼의 크기를) => 랜덤으로 크롭하는 사이즈 지정
		top = random.randint(0, image_h - H)
		left = random.randint(0, image_w - W)

		#이미지의 픽셀에 접근하여 랜덤하게 이미지를 잘라냄
		cropped = np_image[top:top + H, left + W]
		return Image.fromarray(cropped)

#랜덤크롭, 솔트앤페퍼, hsv증강을 모두 한꺼번에 수행할 함수
#tensor_image는 train_loader에서 직접적으로 들어올 이미지 데이터
#p 이 변경(증강)을 얼마의 확률로 적용할 건가?
#hsv_p, sp_p, crop_p : 위의 세 전처리를 확률적으로 수행할 것이다.
#crop_size는 크롭을 얼마나 할 것인가? 
def add_noise(tensor_image, p, hsv_p, sp_p, crop_size, crop_p):

	#텐서 이미지를 복사 opencv일 때 copy(), tensor일 때 clone()
	current_tensor = tensor_image.clone()

	#'텐서'를 numpy배열로 변환!
	# mul(): multiply, 곱셈. 255를 곱합(텐서는 0 ~ 1사이로 정규화되어 있으므로)
	#permute(): 차원 순서 바꾸기. C, H, W(텐서) -> H, W, C(넘파이)
	#numpy(): 넘파이 변환
	#astype(): 데이터의 타입 변환
	image_np = corrent_tensor.mul(255).permute(1, 2, 0).numpy().astype(np.int8)

	#절반의 확률로 증강을 실행
	if random.random() > p:
		pillow = Image.fromarray(image_np)
		cropper = RandomCrop(crop_size)
		pillow = cropper(pillow)
		image_np = np.array(pillow)

		#hsv 변형을 수행
		image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

		#hsv의 색상 변형을 적용할 지 확률값으로 결정함
		if random.random() > hsv_p:
			augmented_hsv(image_bgr, h = 0.1, s = 0.1, v = 0.4)

		#hsv 증강을 한 이후 다시 rgb
		image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

		#솔트 앤 페퍼
		if random.random() > sp_p:
			#ratio에 따라 전체의 10%를 검은 점이나 흰점으로 변환(전체 10%를 솔트 + 페퍼로 변환)
			image_np = salt_and_pepper(image_np, ratio = 0.1)
		#다시 넘파이 -> 텐서로 변환. purmute로 채널정리, float로 소수 변환, div로 0 ~ 1사이로 재정규화	
		augmented_tensor = (torch.from_numpy(image_np.copy()).purmute(2, 0, 1).float().div(255))

		return augmented_tensor

	else:
		tensor_image

#add_noise라는 함수를 직접 수행할 수 없음(lambda에 매개변수를 바로 넣어줄 수 없기 때문에)
#wrapper함수를 만들어줌
def add_noise_wrapper(image):
	return add_noise(image, p = 0.5, hsv_p = 0.5, sp_p = 0.3, crop_size = 32, crop_p = 0.2)