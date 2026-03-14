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

struct QuadratureGeneratorConfig {
  int spatial_dimension = 0;
  int node_count = 0;
  int basis = 1; // 0 for Lagrange, 1 for Monomial
  std::string line_rule = "gauss-legendre";
  int line_points = -1;
  std::vector<double> nodes;
  std::vector<double> lagrange_to_monomial_1d;
  std::vector<double> base_to_monomial_1d;

  QuadratureGeneratorConfig(int dim, int n, int bas, const std::string &rule,
                            int pts)
      : spatial_dimension(dim), node_count(n), basis(bas), line_rule(rule),
        line_points(pts) {

    nodes.reserve(n);
    for (int i = 0; i < n; ++i)
      nodes.push_back(algoim::GaussQuad::x(n, i));

    // Precompute Taylor mappings
    lagrange_to_monomial_1d.resize(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
      std::vector<double> poly(1, 1.0);
      for (int k = 0; k < n; ++k) {
        if (k == i)
          continue;
        double denom = nodes[i] - nodes[k];
        std::vector<double> next_poly(poly.size() + 1, 0.0);
        for (size_t m = 0; m < poly.size(); ++m) {
          next_poly[m + 1] += poly[m] / denom;
          next_poly[m] -= poly[m] * (nodes[k] - 0.5) / denom;
        }
        poly = next_poly;
      }
      for (int j = 0; j < n; ++j) {
        lagrange_to_monomial_1d[j * n + i] = poly[j];
      }
    }
  }
};

template <int N> class MonomialInterpolantLevelSet {
public:
  MonomialInterpolantLevelSet(const real *nodal_values,
                              const QuadratureGeneratorConfig &config)
      : nodal_values_(nodal_values,
                      nodal_values + valueCount(config.node_count)),
        extent_(config.node_count), config_(&config) {

    int n = config.node_count;
    std::vector<real> coeffs(nodal_values, nodal_values + valueCount(n));
    for (int dim = 0; dim < N; ++dim) {
      std::vector<real> next_coeffs(coeffs.size(), 0.0);
      for (MultiLoop<N> index(uvector<int, N>(0), extent_); ~index; ++index) {
        int i = index(dim);
        for (int j = 0; j < n; ++j) {
          uvector<int, N> old_idx = index();
          old_idx(dim) = j;
          next_coeffs[util::furl(index(), extent_)] +=
              config.lagrange_to_monomial_1d[i * n + j] *
              coeffs[util::furl(old_idx, extent_)];
        }
      }
      coeffs = std::move(next_coeffs);
    }
    taylor_coeffs_ = std::move(coeffs);
  }

    template <typename T> T operator()(const uvector<T, N> &x) const {
      int total_elements = 1;
      for (int i = 0; i < N; ++i)
        total_elements *= extent_(i);
      return evalRecursion(x, 0, 0, total_elements / extent_(0));
    }

    template <typename T> uvector<T, N> grad(const uvector<T, N> &x) const {
      uvector<T, N> g(T(0));
      int total_elements = 1;
      for (int i = 0; i < N; ++i)
        total_elements *= extent_(i);
      for (int k = 0; k < N; ++k) {
        g(k) = evalRecursion(x, 0, 0, total_elements / extent_(0), k);
      }
      return g;
    }

    static constexpr int valueCount(int n) {
      int count = 1;
      for (int i = 0; i < N; ++i)
        count *= n;
      return count;
    }

  private:
    const std::vector<real> nodal_values_;
    std::vector<real> taylor_coeffs_;
    const uvector<int, N> extent_;
    const QuadratureGeneratorConfig *config_;

    template <typename T>
    T evalRecursion(const uvector<T, N> &x, int dim, int base_offset,
                    int stride, int diff_dim = -1) const {
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
          T val = evalRecursion(x, dim + 1,
                                base_offset + (extent_(dim) - 1) * stride,
                                next_stride) *
                  T(extent_(dim) - 1);
          for (int k = extent_(dim) - 2; k >= 1; --k) {
            val = val * (x(dim) - T(0.5)) +
                  evalRecursion(x, dim + 1, base_offset + k * stride,
                                next_stride) *
                      T(k);
          }
          return val;
        } else {
          T val = evalRecursion(x, dim + 1,
                                base_offset + (extent_(dim) - 1) * stride,
                                next_stride, diff_dim);
          for (int k = extent_(dim) - 2; k >= 0; --k) {
            val = val * (x(dim) - T(0.5)) +
                  evalRecursion(x, dim + 1, base_offset + k * stride,
                                next_stride, diff_dim);
          }
          return val;
        }
      }
    }
  };

  template <int N> class LagrangeInterpolantLevelSet {
  public:
    LagrangeInterpolantLevelSet(const real *nodal_values,
                                const QuadratureGeneratorConfig &config)
        : nodal_values_(nodal_values,
                        nodal_values + valueCount(config.node_count)),
          extent_(config.node_count), config_(&config) {}

    template <typename T> T operator()(const uvector<T, N> &x) const {
      T val = T(0.0);
      int n = config_->node_count;
      std::vector<std::vector<T>> L(N, std::vector<T>(n, T(1.0)));
      for (int dim = 0; dim < N; ++dim) {
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            if (i == j)
              continue;
            L[dim][i] *= (x(dim) - T(config_->nodes[j])) /
                         T(config_->nodes[i] - config_->nodes[j]);
          }
        }
      }
      for (MultiLoop<N> index(uvector<int, N>(0), extent_); ~index; ++index) {
        T term = T(nodal_values_[util::furl(index(), extent_)]);
        for (int dim = 0; dim < N; ++dim) {
          term *= L[dim][index(dim)];
        }
        val += term;
      }
      return val;
    }

    template <typename T> uvector<T, N> grad(const uvector<T, N> &x) const {
      uvector<T, N> g(T(0));
      int n = config_->node_count;
      std::vector<std::vector<T>> L(N, std::vector<T>(n, T(1.0)));
      std::vector<std::vector<T>> dL(N, std::vector<T>(n, T(0.0)));

      for (int dim = 0; dim < N; ++dim) {
        for (int i = 0; i < n; ++i) {
          T l_i = T(1.0);
          T dl_i = T(0.0);
          for (int j = 0; j < n; ++j) {
            if (i == j)
              continue;
            T term = (x(dim) - T(config_->nodes[j])) /
                     T(config_->nodes[i] - config_->nodes[j]);
            T dterm = T(1.0) / T(config_->nodes[i] - config_->nodes[j]);

            dl_i = dl_i * term + l_i * dterm;
            l_i *= term;
          }
          L[dim][i] = l_i;
          dL[dim][i] = dl_i;
        }
      }

      for (MultiLoop<N> index(uvector<int, N>(0), extent_); ~index; ++index) {
        T term0 = T(nodal_values_[util::furl(index(), extent_)]);
        for (int k = 0; k < N; ++k) {
          T v = term0;
          for (int dim = 0; dim < N; ++dim) {
            if (dim == k)
              v *= dL[dim][index(dim)];
            else
              v *= L[dim][index(dim)];
          }
          g(k) += v;
        }
      }
      return g;
    }

    static constexpr int valueCount(int n) {
      int count = 1;
      for (int i = 0; i < N; ++i)
        count *= n;
      return count;
    }

  private:
    const std::vector<real> nodal_values_;
    const uvector<int, N> extent_;
    const QuadratureGeneratorConfig *config_;
  };

  template <int N, typename Poly> struct NegatedInterpolantLevelSet {
    NegatedInterpolantLevelSet(const Poly &level_set) : level_set_(level_set) {}
    template <typename T> T operator()(const uvector<T, N> &x) const {
      return -level_set_(x);
    }
    template <typename T> uvector<T, N> grad(const uvector<T, N> &x) const {
      return -level_set_.grad(x);
    }
    const Poly &level_set_;
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
      throw std::invalid_argument("line_points must satisfy 1 <= line_points "
                                  "<= 10 for algoim::quadGen");
    return q;
  }

  inline void validateLineRule(const std::string &line_rule) {
    if (line_rule.empty() || line_rule == "gauss-legendre" ||
        line_rule == "gauss_legendre")
      return;
    throw std::invalid_argument(
        "Only gauss-legendre 1D quadrature is implemented at the moment");
  }

  template <int N, typename Poly>
  inline void
  generateBatchQuadraturePolyImpl(const real *cells, std::size_t batch_size,
                                  const QuadratureGeneratorConfig &config,
                                  std::uint32_t flags, int q,
                                  BatchQuadratureResult &result) {
    HyperRectangle<real, N> unit_cell(uvector<real, N>(0.0),
                                      uvector<real, N>(1.0));
    std::size_t values_per_cell = 1;
    for (int dim = 0; dim < N; ++dim)
      values_per_cell *= static_cast<std::size_t>(config.node_count);

    for (std::size_t cell_index = 0; cell_index < batch_size; ++cell_index) {
      const real *cell_values = cells + cell_index * values_per_cell;
      Poly phi(cell_values, config);

      if (result.has_inside) {
        appendRule(quadGen<N>(phi, unit_cell, -1, -1, q), result.inside);
        result.inside.finish_cell();
      }

      if (result.has_outside) {
        NegatedInterpolantLevelSet<N, Poly> minus_phi(phi);
        appendRule(quadGen<N>(minus_phi, unit_cell, -1, -1, q), result.outside);
        result.outside.finish_cell();
      }

      if (result.has_surface) {
        appendRule(quadGen<N>(phi, unit_cell, N, -1, q), result.surface);
        result.surface.finish_cell();
      }
    }
  }

  template <int N>
  inline BatchQuadratureResult
  generateBatchQuadratureImpl(const real *cells, std::size_t batch_size,
                              const QuadratureGeneratorConfig &config,
                              std::uint32_t flags, int q) {
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

    if (config.basis == 0) { // Lagrange
      generateBatchQuadraturePolyImpl<N, LagrangeInterpolantLevelSet<N>>(
          cells, batch_size, config, flags, q, result);
    } else {
      generateBatchQuadraturePolyImpl<N, MonomialInterpolantLevelSet<N>>(
          cells, batch_size, config, flags, q, result);
    }

    return result;
  }

  inline BatchQuadratureResult
  generateBatchQuadrature(const real *cells, std::size_t batch_size,
                          const QuadratureGeneratorConfig &config,
                          std::uint32_t flags) {
    if (cells == nullptr)
      throw std::invalid_argument("cells pointer must not be null");
    if (batch_size == 0)
      throw std::invalid_argument("batch_size must be positive");
    if (flags == 0)
      throw std::invalid_argument(
          "At least one quadrature flag must be specified");

    validateLineRule(config.line_rule);
    int q = resolveLinePointCount(config.node_count, config.line_points);

    if (config.spatial_dimension == 2)
      return generateBatchQuadratureImpl<2>(cells, batch_size, config, flags,
                                            q);
    if (config.spatial_dimension == 3)
      return generateBatchQuadratureImpl<3>(cells, batch_size, config, flags,
                                            q);

    throw std::invalid_argument("spatial_dimension must be 2 or 3");
  }
} // namespace python
} // namespace algoim
#endif
