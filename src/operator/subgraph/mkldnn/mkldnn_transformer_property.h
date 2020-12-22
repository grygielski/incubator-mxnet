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


#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_TRANSFORMER_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_TRANSFORMER_PROPERTY_H_
#if MXNET_USE_MKLDNN == 1

#include <string>
#include <vector>
#include "../common.h"
#include "../../tensor/matrix_op-inl.h"
#include "../../contrib/transformer-inl.h"
#include "mkldnn_subgraph_base-inl.h"

namespace mxnet {
namespace op {

class SgMKLDNNTransformerSelector : public SubgraphSelector {
 public:
  explicit SgMKLDNNTransformerSelector() {}

  bool Select(const nnvm::Node &n, const std::shared_ptr<NodeAttr>& node_attr) override {
    if (n.op() == Op::Get("_contrib_interleaved_matmul_selfatt_qk") ||
        n.op() == Op::Get("_contrib_interleaved_matmul_selfatt_valatt")) {
      return true;
    }
    return false;
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }
};

class SgMKLDNNTransformerProperty : public SubgraphProperty {
 public:
  SgMKLDNNTransformerProperty() {}

  static SubgraphPropertyPtr Create() {
    static const std::string &name = "MKLDNN Transformer optimization pass";
    auto property = std::make_shared<SgMKLDNNTransformerProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_MKLDNN_TRANSFORMER_OPT", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::ObjectPtr n = nnvm::Node::Create();
    // This op has single output, remove duplicated.
    auto last_node = sym.outputs[0].node;
    nnvm::Symbol new_sym;
    new_sym.outputs.emplace_back(last_node);
    std::ostringstream node_name;
    node_name << "_sg_mkldnn";
  
    std::string op_name;
    InterleavedMatMulParam param;
    DFSVisit(new_sym.outputs, [&](const nnvm::ObjectPtr &node) {
      if (node->op() && 
          (node->op()->name == "_contrib_interleaved_matmul_selfatt_qk" ||
           node->op()->name == "_contrib_interleaved_matmul_selfatt_valatt")) {
        op_name = node->op()->name;
        param = nnvm::get<InterleavedMatMulParam>(node->attrs.parsed);
      }
    });
    node_name << op_name << "_" << std::to_string(subgraph_id);


    n->attrs.name = node_name.str();
    n->attrs.op = Op::Get("_sg_mkldnn" + op_name);
    CHECK(n->attrs.op);
    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
    n->attrs.parsed = param;
    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgMKLDNNTransformerSelector>();
    return selector;
  }

  void ConnectSubgraphOutputs(
      const nnvm::ObjectPtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr = nnvm::NodeEntry{n, entry_ptr->index, 0};
    }
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_TRANSFORMER_PROPERTY_H_
