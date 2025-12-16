#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <vector>
#include <tuple>
#include <map>
namespace py = pybind11;
using namespace boost;

// ---- Edges as (source, target) pairs ----
using Edge = std::pair<int, int>;

typedef compressed_sparse_row_graph<directedS,
                    no_property,
                    property<edge_weight_t, double>> Graph;

void single_source_dijkstra(
    const Graph& g,
    int source,
    double* dist_ptr,
    int* pred_ptr
) {

    dijkstra_shortest_paths(
        g,
        source,
        predecessor_map(make_iterator_property_map(pred_ptr, get(vertex_index, g)))
        .distance_map(make_iterator_property_map(dist_ptr, get(vertex_index, g)))
    );
}


std::tuple<py::array_t<double>, py::array_t<int>>
directed_dijkstra(const std::vector<Edge> edges,
                  const std::vector<double> weights,
                  const std::vector<int> &sources)
{
    typedef compressed_sparse_row_graph<directedS,
                        no_property,
                        property<edge_weight_t, double>> Graph;


    // Find max vertex index
    int max_vertex = 0;
    for (const auto &e : edges) {
        if (e.first > max_vertex) max_vertex = e.first;
        if (e.second > max_vertex) max_vertex = e.second;
    }

    int num_nodes = max_vertex + 1;

    // ---- THIS CONSTRUCTOR IS STABLE ----
    Graph g(
        edges_are_unsorted,
        edges.begin(),
        edges.end(),
        weights.begin(),
        num_nodes
    );


    // ---- Output arrays ----
    const int num_sources = sources.size();

    py::array_t<double> distances_out({num_sources, num_nodes});
    py::array_t<int> predecessors_out({num_sources, num_nodes});

    auto distances_ptr = distances_out.mutable_data();
    auto predecessors_ptr = predecessors_out.mutable_data();

    for (int si = 0; si < num_sources; ++si){
        int source = sources[si];

        // Directly point to the row in the output arrays
        double *dist_row = distances_ptr + si * num_nodes;
        int *pred_row = predecessors_ptr + si * num_nodes;

        single_source_dijkstra(g, source, dist_row, pred_row);


    }

    return {distances_out, predecessors_out};
}



PYBIND11_MODULE(boostpy, m)
{
    m.doc() = "Boost Graph Library Dijkstra wrapper with multi-source support";
    m.def("directed_dijkstra", &directed_dijkstra,
          py::arg("edges"),
          py::arg("weights"),
          py::arg("sources"));

}


