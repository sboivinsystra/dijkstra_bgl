#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <limits>

namespace py = pybind11;

struct CSRGraph {
    int n;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> weights;
};

/* ---------------- Single-source Dijkstra ---------------- */

void dijkstra_csr(
    const CSRGraph& g,
    int source,
    double* dist,
    int* pred,
    double cutoff
) {
    const double INF = std::numeric_limits<double>::infinity();

    // Init
    for (int i = 0; i < g.n; ++i) {
        dist[i] = INF;
        pred[i] = -1;
    }

    using Item = std::pair<double, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;

    dist[source] = 0.0;
    pq.emplace(0.0, source);

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue;
        if (d > cutoff) break;

        for (int ei = g.indptr[u]; ei < g.indptr[u + 1]; ++ei) {
            int v = g.indices[ei];
            double nd = d + g.weights[ei];

            if (nd < dist[v] && nd <= cutoff) {
                dist[v] = nd;
                pred[v] = u;
                pq.emplace(nd, v);
            }
        }
    }
}

/* ---------------- Python-exposed function ---------------- */

std::tuple<py::array_t<double>, py::array_t<int>>
multi_source_dijkstra(
    py::array_t<int, py::array::c_style | py::array::forcecast> indptr,
    py::array_t<int, py::array::c_style | py::array::forcecast> indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,
    py::array_t<int, py::array::c_style | py::array::forcecast> sources,
    double cutoff = std::numeric_limits<double>::infinity()
) {
    // ---- Validate shapes ----
    if (indices.size() != weights.size()) {
        throw std::runtime_error("indices and weights must have same length");
    }

    int n = indptr.size() - 1;
    int num_sources = sources.size();

    // ---- Build graph (once) ----
    CSRGraph g;
    g.n = n;
    g.indptr.assign(indptr.data(), indptr.data() + indptr.size());
    g.indices.assign(indices.data(), indices.data() + indices.size());
    g.weights.assign(weights.data(), weights.data() + weights.size());

    // ---- Allocate outputs ----
    py::array_t<double> distances({num_sources, n});
    py::array_t<int> predecessors({num_sources, n});

    auto* dist_ptr = distances.mutable_data();
    auto* pred_ptr = predecessors.mutable_data();
    const int* src_ptr = sources.data();

    // ---- Run Dijkstra for each source ----
    for (int i = 0; i < num_sources; ++i) {
        dijkstra_csr(
            g,
            src_ptr[i],
            dist_ptr + i * n,
            pred_ptr + i * n,
            cutoff
        );
    }

    return {distances, predecessors};
}

/* ---------------- pybind11 module ---------------- */

PYBIND11_MODULE(fast_dijkstra, m) {
    m.doc() = "Fast CSR-based multi-source Dijkstra";

    m.def("dijkstra", &multi_source_dijkstra,
        py::arg("indptr"),
        py::arg("indices"),
        py::arg("weights"),
        py::arg("sources"),
        py::arg("cutoff") = std::numeric_limits<double>::infinity()
    );
}
