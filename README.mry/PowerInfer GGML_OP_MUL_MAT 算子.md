# PowerInfer GGML_OP_MUL_MAT 算子

Transformer中最基础的算子是GEMM/GEMV。在PowerInfer中，GGML_OP_MUL_MAT 算子其实就是完成GEMM/GEMV。

## 1 ggml_tensor

PowerInfer中使用ggml_tensor表示多维矩阵/向量。首先，看一下ggml_tensor这个数据结构：

```c
// n-dimensional tensor
struct ggml_tensor {
    enum ggml_type         type;	// element的类型。例如，INT32, FP16, INT4, INT5等
    int     n_dims;
    int64_t ne[GGML_MAX_DIMS]; // number of elements
    						   // 例如，一个三维tensor，ne = {4096, 64, 8}, ne[0]维度的4096个element连续存储
    size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                               // nb[0] = ggml_type_size(type)
                               // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding。
    					       // nb[1] 计算比较复杂，原因是其使用了block存储以满足字节对齐的要求
                               // nb[i] = nb[i-1] * ne[i-1]
    void * data;
}
```

## 2 ggml_compute_forward_mul_mat

GGML_OP_MUL_MAT 算子的实现函数是：

```c
static void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst)
```

dst = src0×src1。

### 2.1 dst 维度

了解清楚 src0，src1，dst 三个tensor的维度，应该就可以对这个函数的流程有个大致了解。

```c
{ne00, ne01, ne02, ne03} = src0->ne;
{ne10, ne11, ne12, ne13} = src1->ne;
{ne0, ne1, ne2, ne3} = dst;
ne0 = ne01, ne1 = ne11, ne2 = ne12, ne3 = ne13
```

在实际推理过程中，src0为模型参数矩阵/KV cache，src1为hidden_states。

- src0为模型参数矩阵
  - src0 = {n_embd, n_embd}, src1 = {embed_n, n_tokens}
  - 这时候比较好理解，就是常规的矩阵乘
- src0为模型参数KV cache
  - src0 = {n_embd_head, n_kv, n_head_kv}, src1 = {n_embd_head, n_tokens, n_head}
  - 这时候，src1会选择src0中对应的head进行矩阵乘
  - 这种tensor乘法适用于multi-head attention，multi-query attention和group-head attention

<img src="https://github.com/mryvae/picture_bed/assets/83715643/3258f203-e66c-4a0d-abb7-4cecc4976fd0" style="zoom: 50%;" />
