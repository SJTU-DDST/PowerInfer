# PowerInfer attention 动态剪枝

## 1 目标

如下图，在推理过程中，只有少部分head被激活。我们希望在推理过程中，对attention模块进行动态剪枝。

<img src="https://github.com/mryvae/picture_bed/assets/83715643/c9863cb3-7deb-4e22-b00c-1c94517e7662" style="zoom:67%;" />

因时间关系，并没有训练 attention 模块动态剪枝的 predictor，我们只是留下了调用predictor的接口。

## 2 代码方案

经过调研分析，代码实现无需修改llama模型的计算图，只需进行简单的算子替换就可以满足要求。

### 2.1 新增算子 ggml_compute_forward_mul_mat_sparse_attn

我们将部分 ggml_compute_forward_mul_mat 算子替换为了 ggml_compute_forward_mul_mat_sparse_attn

- ggml_compute_forward_mul_mat

  在PowerInfer中，GGML_OP_MUL_MAT 算子其实就是完成GEMM/GEMV。详见 `PowerInfer GGML_OP_MUL_MAT 算子.md`

  ```c
  static void ggml_compute_forward_mul_mat(
          const struct ggml_compute_params * params,
          const struct ggml_tensor * src0,
          const struct ggml_tensor * src1,
                struct ggml_tensor * dst)
  ```

- ggml_compute_forward_mul_mat_sparse_attn

  对 attention 模块动态剪枝后，只有激活head对应的模型参数/kv cache生效。对应到代码实现，src0和src1中只有部分是有效参与计算的。

  ```c
  static void ggml_compute_forward_mul_mat_sparse_attn(
          const struct ggml_compute_params * params,
          const struct ggml_tensor * src0,
          const struct ggml_tensor * src1,
                struct ggml_tensor * dst,
                bool sparse_src0,	// sparse_src0 = true，对src0进行稀疏；否则，对对src1进行稀疏
                int sparse_ne, // 对哪一维度进行稀疏，例如，sparse_ne=0，对ne[0]进行稀疏
                bool init,	// 是否对dst进行初始化
                int64_t lower_bound, // ne[i]中的[lower_bound, upper_bound)部分有效，其余部分无效
                int64_t upper_bound
                )
  ```

  这样做的好处是无需修改计算图的拓扑结构，只需要进行简单的算子替换

### 2.2 计算图算子替换

```c
struct ggml_tensor * activated_head = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_head);
cb(activated_head, "activated_head", -1);	// 初始化 activated_head，activated_head[i] = 1 表示head i 被激活

// self-attention
{
    // 使用sparse_inference，并且处于decode阶段
    if (llama_use_sparse_inference(&model) && n_tokens == 1) {
        ...
        // 对 model.layers[il].wq的ne[1]进行稀疏，激活head部分有效
        struct ggml_tensor * Qcur = ggml_mul_mat_sparse_attn_v1(ctx0, model.layers[il].wq, cur, activated_head);
        ...
        cur = llm_build_kqv_sparse_attn(ctx0, hparams, kv_self,
                model.layers[il].wo, NULL,
                Qcur, KQ_scale, KQ_mask, activated_head, n_ctx, n_tokens, n_kv, -1.0f, cb, il);
}

/*
	各个tensor维度：
    q --> [n_embd_head, n_tokens, n_head]
    k --> [n_embd_head, n_kv, n_head_kv]
    v --> [n_kv, n_embd_head, n_head_kv]
    kq --> [n_kv, n_tokens, n_head]
    kqv --> [n_embd_head, n_tokens, n_head] --> [n_embd_head, n_head] --> [n_embd_head * n_head]
*/
static struct ggml_tensor * llm_build_kqv_sparse_attn(){
    ...
	// 对q的ne[2] 进行稀疏，激活head部分有效
    struct ggml_tensor * kq = ggml_mul_mat_sparse_attn_v2(ctx, k, q, activated_head);
    ...
    // 对kq的ne[2] 进行稀疏，激活head部分有效
    struct ggml_tensor * kqv = ggml_mul_mat_sparse_attn_v2(ctx, v, kq, activated_head);
    ...
    // 对cur的ne[0] 进行稀疏，激活head部分有效
    cur = ggml_mul_mat_sparse_attn_v3(ctx, wo, cur, activated_head);
    ...
    return cur;
}
```

