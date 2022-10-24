use crate::interface::static_bigraph::StaticBigraph;
use traitgraph::index::GraphIndex;
use traitgraph::interface::NodeOrEdge;
use traitgraph_algo::traversal::PreOrderUndirectedBfs;

/// Returns the weakly connected components of a bidirected graph.
/// These are the weakly connected components as defined on a normal graph, but the reverse complement of each node is in the same WCC.
///
/// Assumes that the graph is a correct bidirected graph, i.e. that each node and each edge has a mirror.
///
/// If the graph is empty, no WCCs are returned.
/// Otherwise, an array is returned that maps each node to a root node representing its WCC.
/// Node that if the node ids are not consecutive, this mapping is still returned as consecutive array.
pub fn decompose_weakly_connected_components<Graph: Default + StaticBigraph>(
    graph: &Graph,
) -> Vec<Graph::NodeIndex>
where
    Graph::NodeData: Clone,
    Graph::EdgeData: Clone,
{
    let mut result: Vec<_> = graph.node_indices().collect();
    let mut visited = vec![false; graph.node_count()];
    let mut bfs = PreOrderUndirectedBfs::new_without_start(graph);

    for root_node in graph.node_indices() {
        if visited[root_node.as_usize()] {
            continue;
        }

        bfs.continue_traversal_from(root_node);
        for node_or_edge in bfs.by_ref() {
            let node = if let NodeOrEdge::Node(node) = node_or_edge {
                node
            } else {
                continue;
            };

            debug_assert!(!visited[node.as_usize()]);
            visited[node.as_usize()] = true;
            result[node.as_usize()] = root_node;
        }

        let mirror_root_node = graph.mirror_node(root_node).unwrap();
        if mirror_root_node != root_node && !visited[mirror_root_node.as_usize()] {
            bfs.continue_traversal_from(mirror_root_node);
            for node_or_edge in bfs.by_ref() {
                let node = if let NodeOrEdge::Node(node) = node_or_edge {
                    node
                } else {
                    continue;
                };

                debug_assert!(!visited[node.as_usize()]);
                visited[node.as_usize()] = true;
                result[node.as_usize()] = root_node;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use crate::algo::weakly_connected_components::decompose_weakly_connected_components;
    use crate::implementation::node_bigraph_wrapper::PetBigraph;
    use crate::interface::dynamic_bigraph::DynamicBigraph;
    use crate::interface::static_bigraph::StaticNodeCentricBigraph;
    use traitgraph::implementation::petgraph_impl::PetGraph;
    use traitgraph::interface::MutableGraphContainer;

    #[test]
    fn test_example_graph() {
        let mut graph = PetGraph::new();
        let n0 = graph.add_node(0);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        let n4 = graph.add_node(4);
        let n5 = graph.add_node(5);

        graph.add_edge(n0, n2, 10);
        graph.add_edge(n3, n1, 11);

        let mut graph = PetBigraph::new_unmapped(graph);
        graph.set_mirror_nodes(n0, n1);
        graph.set_mirror_nodes(n2, n3);
        graph.set_mirror_nodes(n4, n5);

        graph.verify_node_mirror_property();

        let wccs = decompose_weakly_connected_components(&graph);
        assert_eq!(
            wccs,
            vec![0.into(), 0.into(), 0.into(), 0.into(), 4.into(), 4.into()]
        );
    }
}
