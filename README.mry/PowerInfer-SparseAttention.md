## 火焰图

build_llama_variants

修改 self-attention 模块

GGML_OP_MUL_MAT --> GGML_OP_MUL_MAT_SPARSE_ATTN



case GGML_OP_MUL_MAT_SPARSE_ATTN:

  {

​    ggml_compute_forward_mul_mat_sparse_attn(params, tensor->src[0], tensor->src[1], tensor);

  } break;

## Analysis

<img src="https://github.com/mryvae/picture_bed/assets/83715643/c9863cb3-7deb-4e22-b00c-1c94517e7662" style="zoom:67%;" />

### 1 Multi-Head-Attention

KV cache

对于Layer L在一个context中，不同token激活的head不同。

如果激活head对应的kv cache没有被保存，需要重新生成。

例如【shipping】激活了head 44，但是head 44对应的kv cache没有被保存。这时候需要进行计算。

这样的话，即使每一步激活了20%的head，但是总的计算量要大于20%。

### 2 Multi-Query-Attention

不同的head共享一份kv参数矩阵

没有需要进行计算激活head对应的kv cache。





N head

20%

## Implement

struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);

按wq的ne1切分





