#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <vector>
#include <tuple>
#include <map>

namespace py = pybind11;



std::tuple<py::array_t<double>, py::array_t<int>>
directed_dijkstra(const py::object &csr_matrix,
                  const std::vector<int> &sources)
{

    using namespace boost;
    typedef adjacency_list<vecS, vecS, directedS,
                           no_property,
                           property<edge_weight_t, double>> Graph;


    // Extract CSR arrays from scipy.sparse.csr_matrix
    py::object indptr_obj = csr_matrix.attr("indptr");
    py::object indices_obj = csr_matrix.attr("indices");
    py::object data_obj = csr_matrix.attr("data");

    std::vector<int> indptr = indptr_obj.cast<std::vector<int>>();
    std::vector<int> indices = indices_obj.cast<std::vector<int>>();
    std::vector<double> data = data_obj.cast<std::vector<double>>();

    int num_nodes = indptr.size() - 1;

    Graph g(num_nodes);

    // Build graph from CSR
    for (int u = 0; u < num_nodes; ++u) {
        for (int idx = indptr[u]; idx < indptr[u + 1]; ++idx) {
            int v = indices[idx];
            double w = data[idx];
            add_edge(u, v, w, g);  // still slow for adjacency_list
        }
    }

    int num_sources = sources.size();

    // Allocate output arrays
    py::array_t<double> distances_out({num_sources, num_nodes});
    py::array_t<int> predecessors_out({num_sources, num_nodes});

    auto distances_ptr = distances_out.mutable_data();
    auto predecessors_ptr = predecessors_out.mutable_data();

    #pragma omp parallel for
    for (int si = 0; si < num_sources; ++si){
        int source = sources[si];

        // Directly point to the row in the output arrays
        double *dist_row = distances_ptr + si * num_nodes;
        int *pred_row = predecessors_ptr + si * num_nodes;

        std::vector<double> distances(num_nodes);

        dijkstra_shortest_paths(
            g, source,
            predecessor_map(pred_row)
                .distance_map(dist_row));

    }

    return std::make_tuple(distances_out, predecessors_out);
}







PYBIND11_MODULE(boostpy, m)
{
    m.doc() = "Boost Graph Library Dijkstra wrapper with multi-source support";
    m.def("directed_dijkstra", &directed_dijkstra,
          py::arg("csr_matrix"),
          py::arg("sources"));

}


