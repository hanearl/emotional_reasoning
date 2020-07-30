# emotional_reasoning

자연어 문장에 내포된 감정을 BERT 기반의 Text Classification Model을 이용해 추론하는 프로젝트. 

사용 기술 : Python3, Tensorflow 2.2, HuggingFace Transformers
데이터 셋
- 문장에 내포된 1개 이상의 감정이 내포된 **Multi-label Dataset**
- 34종의 감정 정보 추론
- **Data Sampling**, **Balanced Cross Entropy Loss**를 이용해 Imbalanced Class 문제를 해결

![Untitled](https://user-images.githubusercontent.com/11373762/88898362-8f9bed00-d287-11ea-8fe4-f756f4d6e8fa.png)
