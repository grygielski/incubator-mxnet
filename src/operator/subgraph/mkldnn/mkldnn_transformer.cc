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
#include "./mkldnn_transformer-inl.h"
//#include "mkldnn_common.h"
#include "../../contrib/transformer-inl.h"
#include "../../tensor/elemwise_unary_op.h"

#include "../../quantization/quantization_utils.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(MKLDNNInterleavedMatMulParam);

static bool MKLDNNInterleavedMatMulSelfAttQKShape(const NodeAttrs& attrs,
                                            mxnet::ShapeVector* in_shape,
                                            mxnet::ShapeVector* out_shape) {
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    auto qkv_shape = in_shape->at(0);
    out_shape->resize(3);
    SHAPE_ASSIGN_CHECK(*out_shape, 0,
      mxnet::TShape({param.heads * qkv_shape[1], qkv_shape[0], qkv_shape[0]}));  // output

    if (!param.enable_float_output) {
      SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape({1}));                     // min output
      SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape({1}));                     // max output
    }
    return true;
  } else {
    CHECK_EQ(in_shape->size(), 1U) << "Input:[queries_keys_values] currently have, "
                                  << in_shape->size() << " inputs";
    auto qkv_shape = in_shape->at(0);
    CHECK_EQ(qkv_shape.ndim(), 3U)
      << "Input queries_keys_values should be 3D in seq_length-batch-proj_dim, "
      << "currently is: " << qkv_shape.ndim() << "D";
    out_shape->resize(1);
    SHAPE_ASSIGN_CHECK(*out_shape, 0,
      mxnet::TShape({param.heads * qkv_shape[1], qkv_shape[0], qkv_shape[0]}));
    return true;
  }
}

static bool MKLDNNInterleavedMatMulSelfAttQKInferType(const nnvm::NodeAttrs &attrs,
                                std::vector<int> *in_types,
                                std::vector<int> *out_types) {
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kInt8);    // qkv input
    TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kFloat32); // min value
    TYPE_ASSIGN_CHECK(*in_types, 2, mshadow::kFloat32); // max value

    if (param.enable_float_output) {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);     // output
    } else {
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);      // output
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);     // output
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);     // min output
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);     // max output
    }
    return true;
  } else {
    return DefaultSubgraphOpType(attrs, in_types, out_types);
  }
}

void MKLDNNInterleavedMatMulSelfAttQKCPU(const nnvm::NodeAttrs& attrs,
                                   const OpContext &ctx,
                                   const std::vector<TBlob> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<TBlob> &outputs) {
  
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);

  if (param.quantized) {
    if (param.enable_float_output) {
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

      auto src1_md = dnnl::memory::desc(src1_dims, dnnl::memory::data_type::s8, src1_strides);
      auto src2_md = dnnl::memory::desc(src2_dims, dnnl::memory::data_type::s8, src2_strides);
      auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc);

      // const float scale = 1.0f / sqrt(static_cast<float>(EMBED/HEADS/3));
      float min_data = inputs[1].dptr<float>()[0];
      float max_data = inputs[2].dptr<float>()[0];

      float data_scale = GetQuantizeScale(mshadow::kInt8, min_data, max_data);
      const float scale = 1.0f / (data_scale * data_scale) / sqrt(static_cast<float>(EMBED/HEADS/3));

      dnnl::primitive_attr attr;
      attr.set_output_scales(0, {scale});

      // CODE FOR HANLDING MASKING
      // float* mask = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
      // memcpy(output, mask, sizeof(float)*HEADS*BS*SEQ_LEN*SEQ_LEN);
      // dnnl::post_ops post_op;
      // post_op.append_sum(1);
      // attr.set_post_ops(post_op);

      auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
      auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine);

      auto matmul_prim = dnnl::matmul(matmul_pd);

      mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
      int8_t* queries_keys_values = inputs[0].FlatTo2D<cpu, int8_t>(s).dptr_;
      
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
    } else if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
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

      auto src1_md = dnnl::memory::desc(src1_dims, dnnl::memory::data_type::s8, src1_strides);
      auto src2_md = dnnl::memory::desc(src2_dims, dnnl::memory::data_type::s8, src2_strides);
      auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::s8, dnnl::memory::format_tag::abc);

      // const float scale = 1.0f / sqrt(static_cast<float>(EMBED/HEADS/3));
      float min_data = inputs[1].dptr<float>()[0];
      float max_data = inputs[2].dptr<float>()[0];

      float data_scale = GetQuantizeScale(mshadow::kInt8, min_data, max_data);
      const float scale = 1.0f / (data_scale * data_scale) / sqrt(static_cast<float>(EMBED/HEADS/3));

      dnnl::primitive_attr attr;
      attr.set_output_scales(0, {scale});

      // CODE FOR HANLDING MASKING
      // float* mask = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
      // memcpy(output, mask, sizeof(float)*HEADS*BS*SEQ_LEN*SEQ_LEN);
      // dnnl::post_ops post_op;
      // post_op.append_sum(1);
      // attr.set_post_ops(post_op);

      auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
      auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine);

      auto matmul_prim = dnnl::matmul(matmul_pd);

      mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
      int8_t* queries_keys_values = inputs[0].FlatTo2D<cpu, int8_t>(s).dptr_;
      
      int8_t* output = outputs[0].FlatTo2D<cpu, int8_t>(s).dptr_;
      float* min_output = outputs[1].dptr<float>();
      float* max_output = outputs[2].dptr<float>();
      min_output[0] = param.min_calib_range.value();
      max_output[0] = param.max_calib_range.value();

      auto src1_mem = dnnl::memory(src1_md, engine, queries_keys_values);
      auto src2_mem = dnnl::memory(src2_md, engine, queries_keys_values+(EMBED/HEADS/3));
      auto dst_mem = dnnl::memory(dst_md, engine, output);

      std::unordered_map<int, dnnl::memory> matmul_args;
      matmul_args.insert({DNNL_ARG_SRC, src1_mem});
      matmul_args.insert({DNNL_ARG_WEIGHTS, src2_mem});
      matmul_args.insert({DNNL_ARG_DST, dst_mem});

      matmul_prim.execute(engine_stream, matmul_args);
      engine_stream.wait();
    } else {
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

      auto src1_md = dnnl::memory::desc(src1_dims, dnnl::memory::data_type::s8, src1_strides);
      auto src2_md = dnnl::memory::desc(src2_dims, dnnl::memory::data_type::s8, src2_strides);
      auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::s32, dnnl::memory::format_tag::abc);

      // const float scale = 1.0f / sqrt(static_cast<float>(EMBED/HEADS/3));
      float min_data = inputs[1].dptr<float>()[0];
      float max_data = inputs[2].dptr<float>()[0];

      // float data_scale = GetQuantizeScale(mshadow::kInt8, min_data, max_data);
      const float scale = 1.0f / sqrt(static_cast<float>(EMBED/HEADS/3));

      dnnl::primitive_attr attr;
      attr.set_output_scales(0, {scale});

      // CODE FOR HANLDING MASKING
      // float* mask = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
      // memcpy(output, mask, sizeof(float)*HEADS*BS*SEQ_LEN*SEQ_LEN);
      // dnnl::post_ops post_op;
      // post_op.append_sum(1);
      // attr.set_post_ops(post_op);

      auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
      auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine);

      auto matmul_prim = dnnl::matmul(matmul_pd);

      mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
      int8_t* queries_keys_values = inputs[0].FlatTo2D<cpu, int8_t>(s).dptr_;
      
      int32_t* output = outputs[0].FlatTo2D<cpu, int32_t>(s).dptr_;
      float* min_output = outputs[1].dptr<float>();
      float* max_output = outputs[2].dptr<float>();

      mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
                s, 1, min_output, max_output, &min_data, &max_data, &min_data,
                &max_data);

      // min_output[0] = min_data;
      // max_output[0] = max_data;

      auto src1_mem = dnnl::memory(src1_md, engine, queries_keys_values);
      auto src2_mem = dnnl::memory(src2_md, engine, queries_keys_values+(EMBED/HEADS/3));
      auto dst_mem = dnnl::memory(dst_md, engine, output);

      std::unordered_map<int, dnnl::memory> matmul_args;
      matmul_args.insert({DNNL_ARG_SRC, src1_mem});
      matmul_args.insert({DNNL_ARG_WEIGHTS, src2_mem});
      matmul_args.insert({DNNL_ARG_DST, dst_mem});

      matmul_prim.execute(engine_stream, matmul_args);
      engine_stream.wait();
    }
  } else {
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

    const float scale = 1.0f / sqrt(static_cast<float>(EMBED/HEADS/3));

    dnnl::primitive_attr attr;
    attr.set_output_scales(0, {scale});

    // CODE FOR HANLDING MASKING
    // float* mask = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
    // memcpy(output, mask, sizeof(float)*HEADS*BS*SEQ_LEN*SEQ_LEN);
    // dnnl::post_ops post_op;
    // post_op.append_sum(1);
    // attr.set_post_ops(post_op);

    auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
    auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine);

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
  }
}

nnvm::ObjectPtr SgMKLDNNInterleavedMatMulSelfAttQKQuantizedOp(const NodeAttrs& attrs) {
  nnvm::ObjectPtr node = nnvm::Node::Create();
  auto const &param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  node->attrs.op = Op::Get("_sg_mkldnn_contrib_interleaved_matmul_selfatt_qk");
  node->attrs.name = "quantized_" + attrs.name;
  node->attrs.dict = attrs.dict;
  node->attrs.dict["heads"] = std::to_string(param.heads);
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

NNVM_REGISTER_OP(_sg_mkldnn_contrib_interleaved_matmul_selfatt_qk)
.describe(R"code(_sg_mkldnn_contrib_interleaved_matmul_selfatt_qk)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    return 3;
  } else {
    return 1;
  }
})
.set_num_outputs([](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized && !param.enable_float_output) {
    return 3;
  } else {
    return 1;
  }
})
.set_attr_parser(ParamParser<MKLDNNInterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  std::vector<std::string> input_names {"queries_keys_values"};
  if (param.quantized) {
    input_names.emplace_back("min_qkv");
    input_names.emplace_back("max_qkv");
  }
  return input_names;
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  std::vector<std::string> output_names {"output"};
  if (param.quantized && !param.enable_float_output) {
    output_names.emplace_back("min_output");
    output_names.emplace_back("max_output");
  }
  return output_names;
})
.set_attr<mxnet::FInferShape>("FInferShape", MKLDNNInterleavedMatMulSelfAttQKShape)
.set_attr<nnvm::FInferType>("FInferType", MKLDNNInterleavedMatMulSelfAttQKInferType)
.set_attr<FCompute>("FCompute<cpu>", MKLDNNInterleavedMatMulSelfAttQKCPU)
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FQuantizable>("FQuantizable", [](const NodeAttrs& attrs) {
    return QuantizeType::kMust;
})
.set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNInterleavedMatMulSelfAttQKQuantizedOp)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
.add_argument("queries_keys_values", "NDArray-or-Symbol", "Interleaved queries, keys and values")
.add_arguments(MKLDNNInterleavedMatMulParam::__FIELDS__());

/********************************************************************************************************/

static bool MKLDNNInterleavedMatMulSelfAttValAttShape(const NodeAttrs& attrs,
                                                mxnet::ShapeVector* in_shape,
                                                mxnet::ShapeVector* out_shape) {
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    auto qkv_shape = in_shape->at(0);
    
    out_shape->resize(3);
    SHAPE_ASSIGN_CHECK(*out_shape, 0,
      mxnet::TShape({qkv_shape[0], qkv_shape[1], qkv_shape[2] / 3}));
    if (!param.enable_float_output) {
      SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape({1}));                     // min output
      SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape({1}));                     // max output
    }
    
    return true;
  } else {
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
}

static bool MKLDNNInterleavedMatMulSelfAttValAttInferType(const nnvm::NodeAttrs &attrs,
                                std::vector<int> *in_types,
                                std::vector<int> *out_types) {
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kInt8);    // qkv input
    TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kUint8);    // att input

    TYPE_ASSIGN_CHECK(*in_types, 2, mshadow::kFloat32); // min qkv
    TYPE_ASSIGN_CHECK(*in_types, 3, mshadow::kFloat32); // max qkv

    TYPE_ASSIGN_CHECK(*in_types, 4, mshadow::kFloat32); // min att
    TYPE_ASSIGN_CHECK(*in_types, 5, mshadow::kFloat32); // max att

    if (param.enable_float_output) {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);     // output
    } else {
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);      // output
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);     // output
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);     // min output
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);     // max output
    }
    return true;
  } else {
    return DefaultSubgraphOpType(attrs, in_types, out_types);
  }
}

nnvm::ObjectPtr SgMKLDNNInterleavedMatMulSelfAttValAttQuantizedOp(const NodeAttrs& attrs) {
  nnvm::ObjectPtr node = nnvm::Node::Create();
  auto const &param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  node->attrs.op = Op::Get("_sg_mkldnn_contrib_interleaved_matmul_selfatt_valatt");
  node->attrs.name = "quantized_" + attrs.name;
  node->attrs.dict = attrs.dict;
  node->attrs.dict["heads"] = std::to_string(param.heads);
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

void MKLDNNInterleavedMatMulSelfAttValAttCPU(const nnvm::NodeAttrs& attrs,
                                       const OpContext &ctx,
                                       const std::vector<TBlob> &inputs,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<TBlob> &outputs) {
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);

  if (param.quantized) {
    if (param.enable_float_output) {
      const dnnl::memory::dim HEADS = param.heads;
      const dnnl::memory::dim BS = inputs[0].shape_[1];
      const dnnl::memory::dim SEQ_LEN = inputs[0].shape_[0];
      const dnnl::memory::dim EMBED = inputs[0].shape_[2];

      dnnl::engine engine(dnnl::engine::kind::cpu, 0);
      dnnl::stream engine_stream(engine);

      dnnl::memory::dims src1_dims = {BS*HEADS, SEQ_LEN, SEQ_LEN};
      dnnl::memory::dims src2_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};
      dnnl::memory::dims dst_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};

      // dnnl::memory::dims src1_strides = {SEQ_LEN*SEQ_LEN, SEQ_LEN, 1};
      dnnl::memory::dims src2_strides = {3*(EMBED/HEADS/3), EMBED*BS, 1};

      auto src1_md = dnnl::memory::desc(src1_dims, dnnl::memory::data_type::u8, dnnl::memory::format_tag::abc); // CHECK IF IT IS U8 FOR SURE
      auto src2_md = dnnl::memory::desc(src2_dims, dnnl::memory::data_type::s8, src2_strides);
      auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::bac);

      float min_qkv = inputs[2].dptr<float>()[0];
      float max_qkv = inputs[3].dptr<float>()[0];
      float min_att = inputs[4].dptr<float>()[0];
      float max_att = inputs[5].dptr<float>()[0];

      float qkv_scale = GetQuantizeScale(mshadow::kInt8, min_qkv, max_qkv);
      float att_scale = GetQuantizeScale(mshadow::kUint8, min_att, max_att);

      const float scale = 1.0f / (qkv_scale * att_scale);

      dnnl::primitive_attr attr;
      attr.set_output_scales(0, {scale});

      auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
      auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine);

      auto matmul_prim = dnnl::matmul(matmul_pd);

      mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
      int8_t* queries_keys_values = inputs[0].FlatTo2D<cpu, int8_t>(s).dptr_;
      uint8_t* attention_maps     = inputs[1].FlatTo2D<cpu, uint8_t>(s).dptr_;
      float* output               = outputs[0].FlatTo2D<cpu, float>(s).dptr_;

      auto src1_mem = dnnl::memory(src1_md, engine, attention_maps);
      auto src2_mem = dnnl::memory(src2_md, engine, queries_keys_values+2*(EMBED/HEADS/3));
      auto dst_mem = dnnl::memory(dst_md, engine, output);

      std::unordered_map<int, dnnl::memory> matmul_args;
      matmul_args.insert({DNNL_ARG_SRC, src1_mem});
      matmul_args.insert({DNNL_ARG_WEIGHTS, src2_mem});
      matmul_args.insert({DNNL_ARG_DST, dst_mem});

      matmul_prim.execute(engine_stream, matmul_args);
      engine_stream.wait();
    } else if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
      const dnnl::memory::dim HEADS = param.heads;
      const dnnl::memory::dim BS = inputs[0].shape_[1];
      const dnnl::memory::dim SEQ_LEN = inputs[0].shape_[0];
      const dnnl::memory::dim EMBED = inputs[0].shape_[2];

      dnnl::engine engine(dnnl::engine::kind::cpu, 0);
      dnnl::stream engine_stream(engine);

      dnnl::memory::dims src1_dims = {BS*HEADS, SEQ_LEN, SEQ_LEN};
      dnnl::memory::dims src2_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};
      dnnl::memory::dims dst_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};

      // dnnl::memory::dims src1_strides = {SEQ_LEN*SEQ_LEN, SEQ_LEN, 1};
      dnnl::memory::dims src2_strides = {3*(EMBED/HEADS/3), EMBED*BS, 1};

      auto src1_md = dnnl::memory::desc(src1_dims, dnnl::memory::data_type::u8, dnnl::memory::format_tag::abc); // CHECK IF IT IS U8 FOR SURE
      auto src2_md = dnnl::memory::desc(src2_dims, dnnl::memory::data_type::s8, src2_strides);
      auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::s8, dnnl::memory::format_tag::bac);

      float min_qkv = inputs[2].dptr<float>()[0];
      float max_qkv = inputs[3].dptr<float>()[0];
      float min_att = inputs[4].dptr<float>()[0];
      float max_att = inputs[5].dptr<float>()[0];

      float qkv_scale = GetQuantizeScale(mshadow::kInt8, min_qkv, max_qkv);
      float att_scale = GetQuantizeScale(mshadow::kUint8, min_att, max_att);

      const float scale = 1.0f / (qkv_scale * att_scale);

      dnnl::primitive_attr attr;
      attr.set_output_scales(0, {scale});

      auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
      auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine);

      auto matmul_prim = dnnl::matmul(matmul_pd);

      mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
      int8_t* queries_keys_values = inputs[0].FlatTo2D<cpu, int8_t>(s).dptr_;
      uint8_t* attention_maps     = inputs[1].FlatTo2D<cpu, uint8_t>(s).dptr_;
      int8_t* output              = outputs[0].FlatTo2D<cpu, int8_t>(s).dptr_;

      float* min_output = outputs[1].dptr<float>();
      float* max_output = outputs[2].dptr<float>();
      min_output[0] = param.min_calib_range.value();
      max_output[0] = param.max_calib_range.value();

      auto src1_mem = dnnl::memory(src1_md, engine, attention_maps);
      auto src2_mem = dnnl::memory(src2_md, engine, queries_keys_values+2*(EMBED/HEADS/3));
      auto dst_mem = dnnl::memory(dst_md, engine, output);

      std::unordered_map<int, dnnl::memory> matmul_args;
      matmul_args.insert({DNNL_ARG_SRC, src1_mem});
      matmul_args.insert({DNNL_ARG_WEIGHTS, src2_mem});
      matmul_args.insert({DNNL_ARG_DST, dst_mem});

      matmul_prim.execute(engine_stream, matmul_args);
      engine_stream.wait();
    } else {
      const dnnl::memory::dim HEADS = param.heads;
      const dnnl::memory::dim BS = inputs[0].shape_[1];
      const dnnl::memory::dim SEQ_LEN = inputs[0].shape_[0];
      const dnnl::memory::dim EMBED = inputs[0].shape_[2];

      dnnl::engine engine(dnnl::engine::kind::cpu, 0);
      dnnl::stream engine_stream(engine);

      dnnl::memory::dims src1_dims = {BS*HEADS, SEQ_LEN, SEQ_LEN};
      dnnl::memory::dims src2_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};
      dnnl::memory::dims dst_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};

      // dnnl::memory::dims src1_strides = {SEQ_LEN*SEQ_LEN, SEQ_LEN, 1};
      dnnl::memory::dims src2_strides = {3*(EMBED/HEADS/3), EMBED*BS, 1};

      auto src1_md = dnnl::memory::desc(src1_dims, dnnl::memory::data_type::u8, dnnl::memory::format_tag::abc); // CHECK IF IT IS U8 FOR SURE
      auto src2_md = dnnl::memory::desc(src2_dims, dnnl::memory::data_type::s8, src2_strides);
      auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::s32, dnnl::memory::format_tag::bac);

      float min_qkv = inputs[2].dptr<float>()[0];
      float max_qkv = inputs[3].dptr<float>()[0];
      float min_att = inputs[4].dptr<float>()[0];
      float max_att = inputs[5].dptr<float>()[0];

      // float qkv_scale = GetQuantizeScale(mshadow::kInt8, min_qkv, max_qkv);
      // float att_scale = GetQuantizeScale(mshadow::kUint8, min_att, max_att);

      // const float scale = 1.0f / (qkv_scale * att_scale);

      // dnnl::primitive_attr attr;
      // attr.set_output_scales(0, {scale});

      auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
      auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, engine);

      auto matmul_prim = dnnl::matmul(matmul_pd);

      mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
      int8_t* queries_keys_values = inputs[0].FlatTo2D<cpu, int8_t>(s).dptr_;
      uint8_t* attention_maps     = inputs[1].FlatTo2D<cpu, uint8_t>(s).dptr_;
      int32_t* output             = outputs[0].FlatTo2D<cpu, int32_t>(s).dptr_;

      float* min_output = outputs[1].dptr<float>();
      float* max_output = outputs[2].dptr<float>();

      mxnet_op::Kernel<QuantizationRangeForS8U8MultiplicationStruct, cpu>::Launch(
                s, 1, min_output, max_output, &min_qkv, &max_qkv, &min_att,
                &max_att);

      auto src1_mem = dnnl::memory(src1_md, engine, attention_maps);
      auto src2_mem = dnnl::memory(src2_md, engine, queries_keys_values+2*(EMBED/HEADS/3));
      auto dst_mem = dnnl::memory(dst_md, engine, output);

      std::unordered_map<int, dnnl::memory> matmul_args;
      matmul_args.insert({DNNL_ARG_SRC, src1_mem});
      matmul_args.insert({DNNL_ARG_WEIGHTS, src2_mem});
      matmul_args.insert({DNNL_ARG_DST, dst_mem});

      matmul_prim.execute(engine_stream, matmul_args);
      engine_stream.wait();
    }
  } else {
    const dnnl::memory::dim HEADS = param.heads;
    const dnnl::memory::dim BS = inputs[0].shape_[1];
    const dnnl::memory::dim SEQ_LEN = inputs[0].shape_[0];
    const dnnl::memory::dim EMBED = inputs[0].shape_[2];

    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream(engine);

    dnnl::memory::dims src1_dims = {BS*HEADS, SEQ_LEN, SEQ_LEN};
    dnnl::memory::dims src2_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};
    dnnl::memory::dims dst_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};

    // dnnl::memory::dims src1_strides = {SEQ_LEN*SEQ_LEN, SEQ_LEN, 1};
    dnnl::memory::dims src2_strides = {3*(EMBED/HEADS/3), EMBED*BS, 1};

    auto src1_md = dnnl::memory::desc(src1_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc);
    auto src2_md = dnnl::memory::desc(src2_dims, dnnl::memory::data_type::f32, src2_strides);
    auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::bac);

    auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
    auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, engine);

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
  }
}

NNVM_REGISTER_OP(_sg_mkldnn_contrib_interleaved_matmul_selfatt_valatt)
.describe(R"code(_sg_mkldnn_contrib_interleaved_matmul_selfatt_valatt)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    return 6;
  } else {
    return 2;
  }
})
.set_num_outputs([](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized && !param.enable_float_output) {
    return 3;
  } else {
    return 1;
  }
})
.set_attr_parser(ParamParser<MKLDNNInterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  std::vector<std::string> input_names {"queries_keys_values", "attention"};
  if (param.quantized) {
    input_names.emplace_back("min_qkv");
    input_names.emplace_back("max_qkv");

    input_names.emplace_back("min_attention");
    input_names.emplace_back("max_attention");
  }
  return input_names;
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  std::vector<std::string> output_names {"output"};
  if (param.quantized && !param.enable_float_output) {
    output_names.emplace_back("min_output");
    output_names.emplace_back("max_output");
  }
  return output_names;
})
.set_attr<mxnet::FInferShape>("FInferShape", MKLDNNInterleavedMatMulSelfAttValAttShape)
.set_attr<nnvm::FInferType>("FInferType", MKLDNNInterleavedMatMulSelfAttValAttInferType)
.set_attr<FCompute>("FCompute<cpu>", MKLDNNInterleavedMatMulSelfAttValAttCPU)
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FQuantizable>("FQuantizable", [](const NodeAttrs& attrs) {
    return QuantizeType::kMust;
})
.set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNInterleavedMatMulSelfAttValAttQuantizedOp)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
.add_argument("queries_keys_values", "NDArray-or-Symbol", "Queries, keys and values interleaved")
.add_argument("attention", "NDArray-or-Symbol", "Attention maps")
.add_arguments(MKLDNNInterleavedMatMulParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

#endif