#ifndef ALGOIM_PYTHON_BATCH_HPP
#define ALGOIM_PYTHON_BATCH_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../algoim/gaussquad.hpp"
#include "../algoim/hyperrectangle.hpp"
#include "../algoim/multiloop.hpp"
#include "../algoim/quadrature_general.hpp"

namespace algoim {
namespace python {
enum QuadratureFlags : std::uint32_t {
  Inside = 1u << 0,
  Outside = 1u << 1,
  Surface = 1u << 2,
};

struct PackedQuadrature {
  int point_dimension = 0;
  std::vector<real> points;
  std::vector<real> weights;
  std::vector<std::uint64_t> offsets;

  void reserve_offsets(std::size_t cell_count, int dimension) {
    point_dimension = dimension;
    offsets.clear();
    offsets.reserve(cell_count + 1);
    offsets.push_back(0);
  }

  void finish_cell() {
    offsets.push_back(static_cast<std::uint64_t>(weights.size()));
  }
};

struct BatchQuadratureResult {
  int spatial_dimension = 0;
  bool has_inside = false;
  bool has_outside = false;
  bool has_surface = false;
  PackedQuadrature inside;
  PackedQuadrature outside;
  PackedQuadrature surface;
};

template <int N> class GaussInterpolantLevelSet {
public:
  GaussInterpolantLevelSet(const real *nodal_values, int node_count)
      : extent_(node_count) {
    if (node_count < 1 || node_count > 100)
      throw std::invalid_argument(
          "node_count must satisfy 1 <= node_count <= 100");

    std::vector<real> nodes(node_count);
    for (int i = 0; i < node_count; ++i)
      nodes[i] = GaussQuad::x(node_count, i);

    std::vector<std::vector<real>> M(node_count,
                                     std::vector<real>(node_count, 0.0));
    for (int i = 0; i < node_count; ++i) {
      std::vector<real> p = {1.0};
      for (int j = 0; j < node_count; ++j) {
        if (i == j)
          continue;
        real denom = nodes[i] - nodes[j];
        std::vector<real> next_p(p.size() + 1, 0.0);
        for (size_t k = 0; k < p.size(); ++k) {
          next_p[k + 1] += p[k] / denom;
          next_p[k] -= p[k] * (nodes[j] - 0.5) / denom;
        }
        p = next_p;
      }
      for (int j = 0; j < node_count; ++j)
        M[i][j] = p[j];
    }

    std::vector<real> coeffs(nodal_values,
                             nodal_values + valueCount(node_count));
    for (int dim = 0; dim < N; ++dim) {
      std::vector<real> next_coeffs(coeffs.size(), 0.0);
      for (MultiLoop<N> index(uvector<int, N>(0), extent_); ~index; ++index) {
        int i = index(dim);
        for (int j = 0; j < node_count; ++j) {
          uvector<int, N> next_idx = index();
          next_idx(dim) = j;
          next_coeffs[util::furl(next_idx, extent_)] +=
              M[i][j] * coeffs[util::furl(index(), extent_)];
        }
      }
      coeffs = next_coeffs;
    }
    taylor_coeffs_ = coeffs;
  }

  template <typename T> T operator()(const uvector<T, N> &x) const {
    int stride = 1;
    for (int d = 1; d < N; ++d)
      stride *= extent_(d);
    return evalRecursion(x, 0, 0, stride);
  }

  template <typename T> uvector<T, N> grad(const uvector<T, N> &x) const {
    uvector<T, N> g;
    int stride = 1;
    for (int d = 1; d < N; ++d)
      stride *= extent_(d);
    for (int k = 0; k < N; ++k) {
      g(k) = evalDerivativeRecursion(x, 0, 0, stride, k);
    }
    return g;
  }

private:
  template <typename T>
  T evalDerivativeRecursion(const uvector<T, N> &x, int dim, int base_offset,
                            int stride, int diff_dim) const {
    if (dim == N - 1) {
      if (dim == diff_dim) {
        if (extent_(dim) <= 1)
          return T(0.0);
        T val = T(taylor_coeffs_[base_offset + (extent_(dim) - 1)]) *
                T(extent_(dim) - 1);
        for (int k = extent_(dim) - 2; k >= 1; --k) {
          val = val * (x(dim) - T(0.5)) +
                T(taylor_coeffs_[base_offset + k]) * T(k);
        }
        return val;
      } else {
        T val = T(taylor_coeffs_[base_offset + (extent_(dim) - 1)]);
        for (int k = extent_(dim) - 2; k >= 0; --k) {
          val = val * (x(dim) - T(0.5)) + T(taylor_coeffs_[base_offset + k]);
        }
        return val;
      }
    } else {
      int next_stride = stride / extent_(dim + 1);
      if (dim == diff_dim) {
        if (extent_(dim) <= 1)
          return T(0.0);
        T val =
            evalRecursion(x, dim + 1, base_offset + (extent_(dim) - 1) * stride,
                          next_stride) *
            T(extent_(dim) - 1);
        for (int k = extent_(dim) - 2; k >= 1; --k) {
          val =
              val * (x(dim) - T(0.5)) +
              evalRecursion(x, dim + 1, base_offset + k * stride, next_stride) *
                  T(k);
        }
        return val;
      } else {
        T val = evalDerivativeRecursion(
            x, dim + 1, base_offset + (extent_(dim) - 1) * stride, next_stride,
            diff_dim);
        for (int k = extent_(dim) - 2; k >= 0; --k) {
          val = val * (x(dim) - T(0.5)) +
                evalDerivativeRecursion(x, dim + 1, base_offset + k * stride,
                                        next_stride, diff_dim);
        }
        return val;
      }
    }
  }

  template <typename T>
  T evalRecursion(const uvector<T, N> &x, int dim, int base_offset,
                  int stride) const {
    if (dim == N - 1) {
      T val = T(taylor_coeffs_[base_offset + (extent_(dim) - 1)]);
      for (int k = extent_(dim) - 2; k >= 0; --k) {
        val = val * (x(dim) - T(0.5)) + T(taylor_coeffs_[base_offset + k]);
      }
      return val;
    } else {
      int next_stride = stride / extent_(dim + 1);
      T val = evalRecursion(
          x, dim + 1, base_offset + (extent_(dim) - 1) * stride, next_stride);
      for (int k = extent_(dim) - 2; k >= 0; --k) {
        val = val * (x(dim) - T(0.5)) +
              evalRecursion(x, dim + 1, base_offset + k * stride, next_stride);
      }
      return val;
    }
  }

  static std::size_t valueCount(int node_count) {
    std::size_t count = 1;
    for (int dim = 0; dim < N; ++dim)
      count *= static_cast<std::size_t>(node_count);
    return count;
  }

  std::vector<real> taylor_coeffs_;
  uvector<int, N> extent_;
};

template <int N> class NegatedInterpolantLevelSet {
public:
  explicit NegatedInterpolantLevelSet(
      const GaussInterpolantLevelSet<N> &level_set)
      : level_set_(level_set) {}

  template <typename T> T operator()(const uvector<T, N> &x) const {
    return -level_set_(x);
  }

  template <typename T> uvector<T, N> grad(const uvector<T, N> &x) const {
    return -level_set_.grad(x);
  }

private:
  const GaussInterpolantLevelSet<N> &level_set_;
};

template <int N>
inline void appendRule(const QuadratureRule<N> &rule,
                       PackedQuadrature &packed) {
  packed.points.reserve(packed.points.size() + N * rule.nodes.size());
  packed.weights.reserve(packed.weights.size() + rule.nodes.size());
  for (const auto &node : rule.nodes) {
    for (int dim = 0; dim < N; ++dim)
      packed.points.push_back(node.x(dim));
    packed.weights.push_back(node.w);
  }
}

inline int resolveLinePointCount(int node_count, int line_points) {
  int q = line_points < 0 ? node_count : line_points;
  if (q < 1 || q > 10)
    throw std::invalid_argument(
        "line_points must satisfy 1 <= line_points <= 10 for algoim::quadGen");
  return q;
}

inline void validateLineRule(const std::string &line_rule) {
  if (line_rule.empty() || line_rule == "gauss-legendre" ||
      line_rule == "gauss_legendre")
    return;
  throw std::invalid_argument(
      "Only gauss-legendre 1D quadrature is implemented at the moment");
}

template <int N>
inline BatchQuadratureResult
generateBatchQuadratureImpl(const real *cells, std::size_t batch_size,
                            int node_count, std::uint32_t flags, int q) {
  BatchQuadratureResult result;
  result.spatial_dimension = N;
  result.has_inside = (flags & Inside) != 0;
  result.has_outside = (flags & Outside) != 0;
  result.has_surface = (flags & Surface) != 0;

  if (result.has_inside)
    result.inside.reserve_offsets(batch_size, N);
  if (result.has_outside)
    result.outside.reserve_offsets(batch_size, N);
  if (result.has_surface)
    result.surface.reserve_offsets(batch_size, N);

  HyperRectangle<real, N> unit_cell(uvector<real, N>(0.0),
                                    uvector<real, N>(1.0));
  std::size_t values_per_cell = 1;
  for (int dim = 0; dim < N; ++dim)
    values_per_cell *= static_cast<std::size_t>(node_count);

  for (std::size_t cell_index = 0; cell_index < batch_size; ++cell_index) {
    const real *cell_values = cells + cell_index * values_per_cell;

    GaussInterpolantLevelSet<N> phi(cell_values, node_count);

    if (result.has_inside) {
      appendRule(quadGen<N>(phi, unit_cell, -1, -1, q), result.inside);
      result.inside.finish_cell();
    }

    if (result.has_outside) {
      NegatedInterpolantLevelSet<N> minus_phi(phi);
      appendRule(quadGen<N>(minus_phi, unit_cell, -1, -1, q), result.outside);
      result.outside.finish_cell();
    }

    if (result.has_surface) {
      appendRule(quadGen<N>(phi, unit_cell, N, -1, q), result.surface);
      result.surface.finish_cell();
    }
  }

  return result;
}

inline BatchQuadratureResult generateBatchQuadrature(
    const real *cells, std::size_t batch_size, int node_count,
    int spatial_dimension, std::uint32_t flags,
    const std::string &line_rule = "gauss-legendre", int line_points = -1) {
  if (cells == nullptr)
    throw std::invalid_argument("cells pointer must not be null");
  if (batch_size == 0)
    throw std::invalid_argument("batch_size must be positive");
  if (flags == 0)
    throw std::invalid_argument(
        "At least one quadrature flag must be specified");

  validateLineRule(line_rule);
  int q = resolveLinePointCount(node_count, line_points);

  if (spatial_dimension == 2)
    return generateBatchQuadratureImpl<2>(cells, batch_size, node_count, flags,
                                          q);
  if (spatial_dimension == 3)
    return generateBatchQuadratureImpl<3>(cells, batch_size, node_count, flags,
                                          q);

  throw std::invalid_argument("spatial_dimension must be 2 or 3");
}
} // namespace python
} // namespace algoim

#endif