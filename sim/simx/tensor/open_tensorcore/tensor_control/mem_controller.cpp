// tensor_mem_manager.cpp
//
// A/B/C/DMem 单实例状态管理器实现。

#include "tensor_mem_manager.h"

namespace vortex {

// ============================================================================
// AMem
// ============================================================================

bool TensorMemManager::is_a_ready(const tud::AMemState& s) {
  return s.a_ready;
}

void TensorMemManager::clear_a_storage(AMem* amem, MetaMem* metamem) {
  amem->clear();
  metamem->clear();
}

void TensorMemManager::set_amemstate(tud::AMemState* s,
                                         const IntrTcuArgs& args,
                                         AMem* amem,
                                         MetaMem* metamem) {
  clear_a_storage(amem, metamem);
  s->reset();
  //s->descriptor = args.descriptor;
  s->fmt_a = args.fmt_a;
  s->a_sparse_mode = args.a_sparse_mode;
  s->transpose_a = args.transpose_a;
}

void TensorMemManager::mark_a_pending(tud::AMemState* s) {
  //s->a_pending = true;
  s->a_valid = false;
  s->a_ready = false;
}

void TensorMemManager::mark_a_valid(tud::AMemState* s) {
  //s->a_pending = false;
  s->a_valid = true;
  s->a_ready = false;
}

void TensorMemManager::mark_a_ready(tud::AMemState* s) {
  //s->a_wmma_pending = false;
    if (! s->ws && (s->use || s->lastuse)) {
    s->a_ws_locked = true;
    s->a_valid = true;
    s->a_ready = false;
  } else {
    s->a_valid = false;
    s->a_ready = true;
    s->a_ws_locked = false;
  }
}

// ============================================================================
// BMem
// ============================================================================

bool TensorMemManager::is_b_ready(const tud::BMemState& s) {
  return s.b_ready;
}

void TensorMemManager::clear_b_storage(BMem* bmem) {
  bmem->clear();
}

void TensorMemManager::set_bmemstate(tud::BMemState* s,
                                         const IntrTcuArgs& args,
                                         BMem* bmem) {
  clear_b_storage(bmem);
  s->reset();
  //s->descriptor = args.descriptor;
  s->fmt_b = args.fmt_b;
  s->transpose_b = args.transpose_b;
}

void TensorMemManager::mark_b_pending(tud::BMemState* s) {
  //s->b_pending = true;
  s->b_valid = false;
  s->b_ready = false;
}

void TensorMemManager::mark_b_valid(tud::BMemState* s) {
  //s->b_pending = false;
  s->b_valid = true;
  s->b_ready = false;
}

void TensorMemManager::mark_b_ready(tud::BMemState* s) {
  //s->b_wmma_pending = false;
  if (s->ws && (s->use || s->lastuse)) {
    s->b_ws_locked = true;
    s->b_valid = true;
    s->b_ready = false;
  } else {
    s->b_valid = false;
    s->b_ready = true;
    s->b_ws_locked = false;
  }
}

// ============================================================================
// CMem
// ============================================================================

bool TensorMemManager::is_c_ready(const tud::CMemState& s) {
  return s.c_ready;
}

void TensorMemManager::clear_c_storage(CMem* cmem) {
  cmem->clear();
}

void TensorMemManager::set_cmemstate(tud::CMemState* s,
                                         const IntrTcuArgs& args,
                                         CMem* cmem) {
  clear_c_storage(cmem);
  s->reset();
  //s->descriptor = args.descriptor;
  s->fmt_c = args.fmt_c;
  //s->fmt_d = args.fmt_d;
  s->output_resident = args.output_resident;
}

void TensorMemManager::mark_c_pending(tud::CMemState* s) {
  //s->c_pending = true;
  s->c_valid = false;
  s->c_ready = false;
}

void TensorMemManager::mark_c_valid(tud::CMemState* s) {
  //s->c_pending = false;
  s->c_valid = true;
  s->c_ready = false;
}

void TensorMemManager::mark_c_ready(tud::CMemState* s) {
  //s->c_pending = false;
  s->c_valid = false;
  s->c_ready = true;
}

// ============================================================================
// DMem
// ============================================================================

bool TensorMemManager::is_d_ready(const tud::DMemState& s) {
  return s.d_ready;
}

void TensorMemManager::clear_d_storage(DMem* dmem) {
  dmem->clear();
}

void TensorMemManager::set_dmemstate(tud::DMemState* s,
                                         const IntrTcuArgs& args,
                                         DMem* dmem) {
  clear_d_storage(dmem);
  s->reset();
  //s->descriptor = args.descriptor;
  s->fmt_d = args.fmt_d;
  s->d_ready = false;
}

// ============================================================================
// 快照发布
// ============================================================================

void TensorMemManager::snapshot_for_scheduler(const tud::AMemState& a_state,
                                              const tud::BMemState& b_state,
                                              const tud::CMemState& c_state,
                                              const tud::DMemState& d_state,
                                              tud::AMemState* pub_a,
                                              tud::BMemState* pub_b,
                                              tud::CMemState* pub_c,
                                              tud::DMemState* pub_d) {
  *pub_a = a_state;
  *pub_b = b_state;
  *pub_c = c_state;
  *pub_d = d_state;
}

} // namespace vortex
