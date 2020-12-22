/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

#if MXNET_USE_MKLDNN == 1

#include <utility>
#include <vector>
#include <string>
#include <mxnet/base.h>
#include "../common.h"
//#include "mkldnn_common.h"
#include "../../contrib/transformer-inl.h"
#include "../../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

static bool MKLDNNInterleavedMatMulSelfAttQKShape(const NodeAttrs& attrs,
                                            mxnet::ShapeVector* in_shape,
                                            mxnet::ShapeVector* out_shape) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U) << "Input:[queries_keys_values] currently have, "
                                 << in_shape->size() << " inputs";
  auto qkv_shape = in_shape->at(0);
  CHECK_EQ(qkv_shape.ndim(), 3U)
    << "Input queries_keys_values should be 3D in seq_length-batch-proj_dim, "
    << "currently is: " << qkv_shape.ndim() << "D";
  out_shape->resize(1);
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
    mxnet::TShape({params.heads * qkv_shape[1], qkv_shape[0], qkv_shape[0]}));
  return true;
}

void strided_batch_sgemm2(bool transA, bool transB,
                         index_t m, index_t n, index_t k,
                         float alpha, const float *a, index_t lda,
                         index_t strideA, const float *b, index_t ldb,
                         index_t strideB, float beta, float *c, index_t ldc,
                         index_t strideC, int32_t batchCount) {
  std::vector<const float*> pp_A(batchCount, nullptr);
  std::vector<const float*> pp_B(batchCount, nullptr);
  std::vector<float*> pp_C(batchCount, nullptr);

  for (int i = 0; i < batchCount; i++) {
    pp_A[i] = a + i * strideA;
    pp_B[i] = b + i * strideB;
    pp_C[i] = c + i * strideC;
  }

#if (MSHADOW_USE_MKL && INTEL_MKL_VERSION >= 20160000)
  const int GROUP_SIZE = 1;
  MKL_INT p_m[GROUP_SIZE] = {m};
  MKL_INT p_n[GROUP_SIZE] = {n};
  MKL_INT p_k[GROUP_SIZE] = {k};
  MKL_INT p_lda[GROUP_SIZE] = {lda};
  MKL_INT p_ldb[GROUP_SIZE] = {ldb};
  MKL_INT p_ldc[GROUP_SIZE] = {ldc};

  float p_alpha[GROUP_SIZE] = {alpha};
  float p_beta[GROUP_SIZE] = {beta};

  CBLAS_TRANSPOSE cblas_a_trans = transA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE cblas_b_trans = transB ? CblasTrans : CblasNoTrans;

  MKL_INT p_group_sizeb[GROUP_SIZE] = {batchCount};
  CBLAS_TRANSPOSE p_transa[GROUP_SIZE] = {cblas_a_trans};
  CBLAS_TRANSPOSE p_transb[GROUP_SIZE] = {cblas_b_trans};

  cblas_sgemm_batch(CblasColMajor, p_transa, p_transb,
                    p_m, p_n, p_k, p_alpha, pp_A.data(), p_lda, pp_B.data(),
                    p_ldb, p_beta, pp_C.data(), p_ldc, GROUP_SIZE, p_group_sizeb);
#else
  for (int i = 0; i < batchCount; ++i) {
    cblas_sgemm(CblasColMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                m, n, k,
                alpha, pp_A[i], lda,
                pp_B[i], ldb, beta, pp_C[i], ldc);
  }
#endif
}

void MKLDNNInterleavedMatMulSelfAttQKCPU(const nnvm::NodeAttrs& attrs,
                                   const OpContext &ctx,
                                   const std::vector<TBlob> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<TBlob> &outputs) {
  // const index_t qkv_seq_len    = inputs[0].shape_[0];
  // const index_t sequences      = inputs[0].shape_[1];
  // const index_t output_lin_dim = inputs[0].shape_[2];
  // const index_t embed_dim      = output_lin_dim / 3;
  // const index_t head_dim       = embed_dim / params.heads;
  // const index_t attn_batches   = params.heads * sequences;
  // const index_t lead_dim       = attn_batches * 3 * head_dim;
  // const index_t batch_stride   = 3 * head_dim;
  // const float beta             = req[0] == kAddTo ? 1.f : 0.f;
  // const float scale            = 1.0 / sqrt(static_cast<float>(head_dim));
  
  const auto& param = nnvm::get<InterleavedMatMulParam>(attrs.parsed);

  const dnnl::memory::dim HEADS = param.heads;
  const dnnl::memory::dim BS = inputs[0].shape_[1];
  const dnnl::memory::dim SEQ_LEN = inputs[0].shape_[0];
  const dnnl::memory::dim EMBED = inputs[0].shape_[2];

  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(engine);

  dnnl::memory::dims src1_dims = {HEADS*BS, SEQ_LEN, EMBED/HEADS/3};
  dnnl::memory::dims src2_dims = {HEADS*BS, EMBED/HEADS/3, SEQ_LEN};
  dnnl::memory::dims dst_dims = {HEADS*BS, SEQ_LEN, SEQ_LEN};

  dnnl::memory::dims src1_strides = {3*(EMBED/HEADS/3), EMBED*BS, 1};
  dnnl::memory::dims src2_strides = {3*(EMBED/HEADS/3), 1, EMBED*BS};

  auto src1_md = dnnl::memory::desc(src1_dims, dnnl::memory::data_type::f32, src1_strides);
  auto src2_md = dnnl::memory::desc(src2_dims, dnnl::memory::data_type::f32, src2_strides);
  auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc);

  dnnl::primitive_attr attr;
  const float scale = 1.0f / sqrt(static_cast<float>(EMBED/HEADS/3));
  attr.set_output_scales(0, {scale});

  auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
  auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine);

  // const float scale = 1.0f / sqrt(static_cast<float>(head_dim);
  // const float alpha = 0.f;
  // const float beta = 0.f;

  
  // post_ops matmul_ops;
  // post_ops.set_o
  // matmul_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
  // primitive_attr matmul_attr;
  // matmul_attr.set_post_ops(matmul_ops);

  auto matmul_prim = dnnl::matmul(matmul_pd);

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  float* queries_keys_values = inputs[0].FlatTo2D<cpu, float>(s).dptr_;
  float* output = outputs[0].FlatTo2D<cpu, float>(s).dptr_;

  auto src1_mem = dnnl::memory(src1_md, engine, queries_keys_values);
  auto src2_mem = dnnl::memory(src2_md, engine, queries_keys_values+(EMBED/HEADS/3));
  auto dst_mem = dnnl::memory(dst_md, engine, output);

  std::unordered_map<int, dnnl::memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, src1_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, src2_mem});
  matmul_args.insert({DNNL_ARG_DST, dst_mem});

  matmul_prim.execute(engine_stream, matmul_args);
  engine_stream.wait();


  // Create primitive post-ops (ReLU).
  
  

  // const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  // if (req[0] == kNullOp)
  //   return;

  // CHECK_EQ(inputs[0].type_flag_, mshadow::kFloat32)
  //   << "Only FP32 is supported on CPU at the moment";

  // mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  // const float* queries_keys_values = inputs[0].FlatTo2D<cpu, float>(s).dptr_;
  // float* output = outputs[0].FlatTo2D<cpu, float>(s).dptr_;

  
  // strided_batch_sgemm2(true,
  //                     false,
  //                     qkv_seq_len,
  //                     qkv_seq_len,
  //                     head_dim,
  //                     scale,
  //                     queries_keys_values + head_dim,
  //                     lead_dim,
  //                     batch_stride,
  //                     queries_keys_values,
  //                     lead_dim,
  //                     batch_stride,
  //                     beta,
  //                     output,
  //                     qkv_seq_len,
  //                     qkv_seq_len * qkv_seq_len,
  //                     attn_batches);
}

NNVM_REGISTER_OP(_sg_mkldnn_contrib_interleaved_matmul_selfatt_qk)
.describe(R"code(_sg_mkldnn_contrib_interleaved_matmul_selfatt_qk)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"queries_keys_values"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", MKLDNNInterleavedMatMulSelfAttQKShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", MKLDNNInterleavedMatMulSelfAttQKCPU)
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("queries_keys_values", "NDArray-or-Symbol", "Interleaved queries, keys and values")
.add_arguments(InterleavedMatMulParam::__FIELDS__());


static bool MKLDNNInterleavedMatMulSelfAttValAttShape(const NodeAttrs& attrs,
                                                mxnet::ShapeVector* in_shape,
                                                mxnet::ShapeVector* out_shape) {
  CHECK_EQ(in_shape->size(), 2U) << "Input:[queries_keys_values, attention] currently have, "
                                 << in_shape->size() << " inputs";
  auto qkv_shape = in_shape->at(0);
  auto att_shape = in_shape->at(1);
  CHECK_EQ(qkv_shape.ndim(), 3U)
    << "Input queries_keys_values should be 3D in seq_length-batch-3*proj_dim, "
    << "currently is: " << qkv_shape.ndim() << "D";
  CHECK_EQ(att_shape.ndim(), 3U)
    << "Input attention should be 3D in batch-seq_length-seq_length, "
    << "currently is: " << att_shape.ndim() << "D";
  CHECK_EQ(qkv_shape[0], att_shape[1])
    << "queries_keys_values.shape[0] and attention.shape[1] should be the same, "
    << "currently are " << qkv_shape[0] << " and " << att_shape[1];
  CHECK_EQ(qkv_shape[0], att_shape[2])
    << "queries_keys_values.shape[0] and attention.shape[2] should be the same, "
    << "currently are " << qkv_shape[0] << " and " << att_shape[2];
  CHECK_EQ(qkv_shape[2] % 3, 0)
    << "queries_keys_values.shape[2] should be a multiple of 3, "
    << "currently is " << qkv_shape[2];
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
    mxnet::TShape({qkv_shape[0], qkv_shape[1], qkv_shape[2] / 3}));
  return true;
}

void MKLDNNInterleavedMatMulSelfAttValAttCPU(const nnvm::NodeAttrs& attrs,
                                       const OpContext &ctx,
                                       const std::vector<TBlob> &inputs,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<TBlob> &outputs) {
  const auto& param = nnvm::get<InterleavedMatMulParam>(attrs.parsed);

  const dnnl::memory::dim HEADS = param.heads;
  const dnnl::memory::dim BS = inputs[0].shape_[1];
  const dnnl::memory::dim SEQ_LEN = inputs[0].shape_[0];
  const dnnl::memory::dim EMBED = inputs[0].shape_[2];

  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(engine);

  dnnl::memory::dims src1_dims = {BS*HEADS, SEQ_LEN, SEQ_LEN}; // 2,2,2
  dnnl::memory::dims src2_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3}; // 2,2,1
  dnnl::memory::dims dst_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3}; // 2,2,1

  // dnnl::memory::dims src1_strides = {SEQ_LEN*SEQ_LEN, SEQ_LEN, 1};
  dnnl::memory::dims src2_strides = {3*(EMBED/HEADS/3), EMBED*BS, 1};

  auto src1_md = dnnl::memory::desc(src1_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc);
  auto src2_md = dnnl::memory::desc(src2_dims, dnnl::memory::data_type::f32, src2_strides);
  auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::bac);

  auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
  auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, engine);

  // const float scale = 1.0f / sqrt(static_cast<float>(head_dim);
  // const float alpha = 0.f;
  // const float beta = 0.f;

  
  // post_ops matmul_ops;
  // post_ops.set_o
  // matmul_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
  // primitive_attr matmul_attr;
  // matmul_attr.set_post_ops(matmul_ops);

  auto matmul_prim = dnnl::matmul(matmul_pd);

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  float* queries_keys_values = inputs[0].FlatTo2D<cpu, float>(s).dptr_;
  float* attention_maps      = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
  float* output              = outputs[0].FlatTo2D<cpu, float>(s).dptr_;

  auto src1_mem = dnnl::memory(src1_md, engine, attention_maps);
  auto src2_mem = dnnl::memory(src2_md, engine, queries_keys_values+2*(EMBED/HEADS/3));
  auto dst_mem = dnnl::memory(dst_md, engine, output);

  std::unordered_map<int, dnnl::memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, src1_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, src2_mem});
  matmul_args.insert({DNNL_ARG_DST, dst_mem});

  matmul_prim.execute(engine_stream, matmul_args);
  engine_stream.wait();



  // const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  // if (req[0] == kNullOp)
  //   return;

  // CHECK_EQ(inputs[0].type_flag_, mshadow::kFloat32)
  //   << "Only FP32 is supported on CPU at the moment";

  // mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  // const float* queries_keys_values = inputs[0].FlatTo2D<cpu, float>(s).dptr_;
  // const float* attention_maps      = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
  // float* output                    = outputs[0].FlatTo2D<cpu, float>(s).dptr_;
  // const index_t qkv_seq_len    = inputs[0].shape_[0];
  // const index_t sequences      = inputs[0].shape_[1];
  // const index_t output_lin_dim = inputs[0].shape_[2];
  // const index_t embed_dim      = output_lin_dim / 3;
  // const index_t head_dim       = embed_dim / params.heads;
  // const index_t attn_batches   = params.heads * sequences;
  // const index_t lead_dim       = attn_batches * 3 * head_dim;
  // const index_t batch_stride   = 3 * head_dim;
  // const float alpha             = 1.f;
  // const float beta              = req[0] == kAddTo ? 1.f : 0.f;

  // strided_batch_sgemm2(false,
  //                     false,
  //                     head_dim,
  //                     qkv_seq_len,
  //                     qkv_seq_len,
  //                     alpha,
  //                     queries_keys_values + 2 * head_dim,
  //                     lead_dim,
  //                     batch_stride,
  //                     attention_maps,
  //                     qkv_seq_len,
  //                     qkv_seq_len * qkv_seq_len,
  //                     beta,
  //                     output,
  //                     head_dim * attn_batches,
  //                     head_dim,
  //                     attn_batches);
}


NNVM_REGISTER_OP(_sg_mkldnn_contrib_interleaved_matmul_selfatt_valatt)
.describe(R"code(_sg_mkldnn_contrib_interleaved_matmul_selfatt_valatt)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"queries_keys_values", "attention"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", MKLDNNInterleavedMatMulSelfAttValAttShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", MKLDNNInterleavedMatMulSelfAttValAttCPU)
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("queries_keys_values", "NDArray-or-Symbol", "Queries, keys and values interleaved")
.add_argument("attention", "NDArray-or-Symbol", "Attention maps")
.add_arguments(InterleavedMatMulParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

#endif