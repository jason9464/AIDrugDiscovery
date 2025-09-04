### Transformer_class.py
Class (결합가능여부) 예측 모델 학습 코드

### Transformer_IC50.py
IC50 예측 모델 학습 코드

### Transformer_inference_test_dataset.py
test dataset을 입력 받아 test dataset에 대한 IC50/Class을 predict

### Transformer_inference_new.py
항체 항원 두 시퀀스를 입력 받아 IC50/Class을 predict

### model.py
Transformer 모델 정의

### data_prep.py
데이터 전처리 및 dataset 정의

### rdkit_prep.py
FASTA -> Graph 관련 데이터 전처리 (with rdkit)

### VirusNet.csv
class predict에 사용되는 dataset

### CATNAP_data.csv
IC50 predict에 사용되는 dataset

### VirusNet2.csv
class/IC50 predict에 사용되는 dataset
보고서의 성능 평가에 사용됨

### 실행 코드
#### 모델 학습
```
python Transformer_classification.py
```
--lr : learning rate 조절  
--reg : weight decay 조절  
--batch_size : batch size 조절  
--num_epochs : epoch 수 조절  
--gpu : 사용할 gpu  

일반적으로 default로 사용

```
python Transformer_IC50.py
```
--lr : learning rate 조절  
--reg : weight decay 조절  
--batch_size : batch size 조절  
--num_epochs : epoch 수 조절  
--gpu : 사용할 gpu  

일반적으로 default로 사용

#### 예측
```
python Transformer_inference_new.py
```
--task : classification / regression task 선택  
--model_save_path : 학습된 모델 저장 위치  
--ab  : 항체 문자열    
--vir : 항원 문자열    
--gpu : 사용할 gpu    

항체 항원 문자열을 매개변수 필요

```
python Transformer_inference.py
```
--test_path : test dataset 저장 위치  
--model_save_path : 학습된 모델 저장 위치  
--gpu : 사용할 gpu    

항체 항원 test dataset 필요
