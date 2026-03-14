#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../algoim/gaussquad.hpp"
#include "../python/algoim_batch.hpp"

namespace {
constexpr double tol = 5.0e-5;

template <int N>
std::vector<double> makeConstantCell(int node_count, double value) {
  std::size_t count = 1;
  for (int dim = 0; dim < N; ++dim)
    count *= static_cast<std::size_t>(node_count);
  return std::vector<double>(count, value);
}

template <int N, typename F>
std::vector<double> sampleCell(int node_count, const F &phi) {
  std::vector<double> cell;
  std::size_t count = 1;
  for (int dim = 0; dim < N; ++dim)
    count *= static_cast<std::size_t>(node_count);
  cell.resize(count);

  for (algoim::MultiLoop<N> index(algoim::uvector<int, N>(0),
                                  algoim::uvector<int, N>(node_count));
       ~index; ++index) {
    algoim::uvector<double, N> x;
    for (int dim = 0; dim < N; ++dim)
      x(dim) = algoim::GaussQuad::x(node_count, index(dim));
    cell[algoim::util::furl(index(), algoim::uvector<int, N>(node_count))] =
        phi(x);
  }

  return cell;
}

double sumWeights(const algoim::python::PackedQuadrature &packed) {
  double total = 0.0;
  for (double weight : packed.weights)
    total += weight;
  return total;
}

void require(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}
void requireNear(double actual, double expected, const std::string &message) {
  if (std::abs(actual - expected) > tol) {
    std::cerr << message << ": expected " << expected << ", got " << actual
              << "\n";
    std::exit(EXIT_FAILURE);
  }
}

template <int N> void testConstantPositive(int node_count) {
  auto cell = makeConstantCell<N>(node_count, 1.0);
  auto result = algoim::python::generateBatchQuadrature(
      cell.data(), 1, node_count, N,
      algoim::python::Inside | algoim::python::Outside |
          algoim::python::Surface);

  requireNear(sumWeights(result.inside), 0.0,
              "Constant positive cell should have zero inside measure");
  requireNear(sumWeights(result.outside), 1.0,
              "Constant positive cell should have full outside measure");
  require(result.surface.weights.empty(),
          "Constant positive cell should have empty surface quadrature");
}

template <int N> void testConstantNegative(int node_count) {
  auto cell = makeConstantCell<N>(node_count, -1.0);
  auto result = algoim::python::generateBatchQuadrature(
      cell.data(), 1, node_count, N,
      algoim::python::Inside | algoim::python::Outside |
          algoim::python::Surface);

  requireNear(sumWeights(result.inside), 1.0,
              "Constant negative cell should have full inside measure");
  requireNear(sumWeights(result.outside), 0.0,
              "Constant negative cell should have zero outside measure");
  require(result.surface.weights.empty(),
          "Constant negative cell should have empty surface quadrature");
}

void testStraightLine2D(int node_count) {
  auto cell =
      sampleCell<2>(node_count, [](const auto &x) { return x(0) - 0.5; });
  auto result = algoim::python::generateBatchQuadrature(
      cell.data(), 1, node_count, 2,
      algoim::python::Inside | algoim::python::Outside |
          algoim::python::Surface);

  requireNear(sumWeights(result.inside), 0.5,
              "2D line split should give half-cell inside measure");
  requireNear(sumWeights(result.outside), 0.5,
              "2D line split should give half-cell outside measure");
  requireNear(sumWeights(result.surface), 1.0,
              "2D line split should give unit surface measure");
  for (std::size_t i = 0; i < result.surface.weights.size(); ++i)
    requireNear(result.surface.points[2 * i], 0.5,
                "2D line split surface points should lie on x=0.5");
}
} // namespace

int main() {
  constexpr int node_count = 3;
  testConstantPositive<2>(node_count);
  testConstantNegative<2>(node_count);
  testStraightLine2D(node_count);
  testConstantPositive<3>(node_count);
  testConstantNegative<3>(node_count);

  std::cout << "batch quadrature smoke test passed\n";
  return EXIT_SUCCESS;
}