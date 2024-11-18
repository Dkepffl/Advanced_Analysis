각 Class의 Argument 정리용
## **Positional Encoding**
```python
pe = PositionalEncoding(device, max_len, d_model)
pos_emb = pe(x)
```
## **ScaledDotProductAttention**
```python
attention = ScaledDotProductAttention()
attention_value = attention(q, k, v, mask=None)
```
### **Multi Head Attention(수정 필요)**
```python
multiheadattention = MultiHeadAttention(d_model, num_head)
output = multiheadattention(q, k, v, mask=None)
```
### **PositionwiseFeedForwardNetwork**
```python
ffn = PositionwiseFeedForwardNetwork(d_model, num_head)
output = ffn(x)
```
