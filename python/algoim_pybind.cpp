#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <optional>
#include <string>

#include "algoim_batch.hpp"

extern "C" {
struct AlgoimPackedQuadrature {
  const double *points;
  const double *weights;
  const std::uint64_t *offsets;
  int point_dimension;
  std::uint64_t point_count;
  std::uint64_t offset_count;
};

struct AlgoimBatchQuadratureHandle;

AlgoimBatchQuadratureHandle *algoim_generate_batch_quadrature(
    const double *cells, std::uint64_t batch_size, int node_count,
    int spatial_dimension, std::uint32_t flags, const char *line_rule,
    int line_points, char *error_buffer, std::uint64_t error_buffer_size);

void algoim_batch_quadrature_free(AlgoimBatchQuadratureHandle *handle);
int algoim_batch_quadrature_has_inside(
    const AlgoimBatchQuadratureHandle *handle);
int algoim_batch_quadrature_has_outside(
    const AlgoimBatchQuadratureHandle *handle);
int algoim_batch_quadrature_has_surface(
    const AlgoimBatchQuadratureHandle *handle);
int algoim_batch_quadrature_get_inside(
    const AlgoimBatchQuadratureHandle *handle, AlgoimPackedQuadrature *out);
int algoim_batch_quadrature_get_outside(
    const AlgoimBatchQuadratureHandle *handle, AlgoimPackedQuadrature *out);
int algoim_batch_quadrature_get_surface(
    const AlgoimBatchQuadratureHandle *handle, AlgoimPackedQuadrature *out);
}

namespace {
struct BatchQuadratureHandleImpl {
  algoim::python::BatchQuadratureResult result;
};

BatchQuadratureHandleImpl *unwrap(AlgoimBatchQuadratureHandle *handle) {
  return reinterpret_cast<BatchQuadratureHandleImpl *>(handle);
}

const BatchQuadratureHandleImpl *
unwrap(const AlgoimBatchQuadratureHandle *handle) {
  return reinterpret_cast<const BatchQuadratureHandleImpl *>(handle);
}

void writeError(const std::string &message, char *error_buffer,
                std::uint64_t error_buffer_size) {
  if (error_buffer == nullptr || error_buffer_size == 0)
    return;

  std::size_t count = std::min<std::size_t>(
      message.size(), static_cast<std::size_t>(error_buffer_size - 1));
  std::memcpy(error_buffer, message.data(), count);
  error_buffer[count] = '\0';
}

int packQuadrature(const algoim::python::PackedQuadrature &packed,
                   AlgoimPackedQuadrature *out) {
  if (out == nullptr)
    return 0;

  out->points = packed.points.empty() ? nullptr : packed.points.data();
  out->weights = packed.weights.empty() ? nullptr : packed.weights.data();
  out->offsets = packed.offsets.empty() ? nullptr : packed.offsets.data();
  out->point_dimension = packed.point_dimension;
  out->point_count = static_cast<std::uint64_t>(packed.weights.size());
  out->offset_count = static_cast<std::uint64_t>(packed.offsets.size());
  return 1;
}
} // namespace

extern "C" {
AlgoimBatchQuadratureHandle *algoim_generate_batch_quadrature(
    const double *cells, std::uint64_t batch_size, int node_count,
    int spatial_dimension, std::uint32_t flags, const char *line_rule,
    int line_points, char *error_buffer, std::uint64_t error_buffer_size) {
  try {
    std::string resolved_line_rule =
        line_rule != nullptr ? std::string(line_rule) : std::string();
    auto *handle =
        new BatchQuadratureHandleImpl{algoim::python::generateBatchQuadrature(
            cells, batch_size, node_count, spatial_dimension, flags,
            resolved_line_rule, line_points)};
    return reinterpret_cast<AlgoimBatchQuadratureHandle *>(handle);
  } catch (const std::exception &error) {
    writeError(error.what(), error_buffer, error_buffer_size);
    return nullptr;
  }
}

void algoim_batch_quadrature_free(AlgoimBatchQuadratureHandle *handle) {
  delete unwrap(handle);
}

int algoim_batch_quadrature_has_inside(
    const AlgoimBatchQuadratureHandle *handle) {
  return handle != nullptr && unwrap(handle)->result.has_inside ? 1 : 0;
}

int algoim_batch_quadrature_has_outside(
    const AlgoimBatchQuadratureHandle *handle) {
  return handle != nullptr && unwrap(handle)->result.has_outside ? 1 : 0;
}

int algoim_batch_quadrature_has_surface(
    const AlgoimBatchQuadratureHandle *handle) {
  return handle != nullptr && unwrap(handle)->result.has_surface ? 1 : 0;
}

int algoim_batch_quadrature_get_inside(
    const AlgoimBatchQuadratureHandle *handle, AlgoimPackedQuadrature *out) {
  return handle != nullptr && unwrap(handle)->result.has_inside
             ? packQuadrature(unwrap(handle)->result.inside, out)
             : 0;
}

int algoim_batch_quadrature_get_outside(
    const AlgoimBatchQuadratureHandle *handle, AlgoimPackedQuadrature *out) {
  return handle != nullptr && unwrap(handle)->result.has_outside
             ? packQuadrature(unwrap(handle)->result.outside, out)
             : 0;
}

int algoim_batch_quadrature_get_surface(
    const AlgoimBatchQuadratureHandle *handle, AlgoimPackedQuadrature *out) {
  return handle != nullptr && unwrap(handle)->result.has_surface
             ? packQuadrature(unwrap(handle)->result.surface, out)
             : 0;
}
}