
## 2024.11.15 16:06
코드를 Part 별로 정돈.
## 2024.11.18 2:06
- 논문 읽으면서 채우긴 함
- 근데 Positional Encoding이 뭐죠?
- Masking은?
- Layer Normalization은 뭐죠?
## 2024.11.18 18:20
- 찐 Encoder& Decorder도 채움
- q,k,v 어떻게 변환하더라?
- Masking 뭐 어쩌라고요?
- Dropout...불러놓고 안 씀
- Residual Connection.... 안 한 듯
- 여전히...Positional Encoding이 뭐죠?
## 2024.11.18 23:46
- Multi Head Attention까지 완료
- **Dimension 확인 필요**
## 2024.11.19 1:06
- Masking 어떻게 하는 거죠?
- device 확인 좀
- argument 그냥 `arg=arg`로 하는 게 맘 편할 듯
## 2024.11.19
- masking 어떻게든 해봄
- Debugging
- 1 epoch 돌려보는 중
- loss가 대박 짱 크다
- https://wikidocs.net/217018
## 2024.11.23 
- GPU에 안 올라간 애가 있는 것 같아요->아닌가봄
- masking shape 확인하기 -> (512,512)
