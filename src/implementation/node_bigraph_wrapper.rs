use crate::interface::dynamic_bigraph::DynamicEdgeCentricBigraph;
use crate::interface::dynamic_bigraph::DynamicNodeCentricBigraph;
use crate::interface::{
    dynamic_bigraph::DynamicBigraph,
    static_bigraph::StaticBigraph,
    static_bigraph::{
        StaticBigraphFromDigraph, StaticEdgeCentricBigraph, StaticNodeCentricBigraph,
    },
    BidirectedData,
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use traitgraph::index::{GraphIndex, GraphIndices, OptionalGraphIndex};
use traitgraph::interface::{
    DynamicGraph, Edge, GraphBase, ImmutableGraphContainer, MutableGraphContainer, NavigableGraph,
    StaticGraph,
};

/// Represent arbitrary bigraphs with petgraph.
pub type PetBigraph<NodeData, EdgeData> =
    crate::implementation::node_bigraph_wrapper::NodeBigraphWrapper<
        crate::traitgraph::implementation::petgraph_impl::PetGraph<NodeData, EdgeData>,
    >;

/// Wrapper for a static graph that adds a mirror node mapping function.
///
/// Bigraphs can be represented with this struct by creating their topology as normal directed graph where each binode is split into its two parts.
/// The binode mapping function then associates the parts with each other.
///
/// ```rust
/// use traitgraph::implementation::petgraph_impl::PetGraph;
/// use bigraph::implementation::node_bigraph_wrapper::NodeBigraphWrapper;
/// use bigraph::interface::static_bigraph::{StaticBigraph, StaticBigraphFromDigraph};
/// use bigraph::traitgraph::interface::MutableGraphContainer;
/// use bigraph::interface::BidirectedData;
///
/// #[derive(Clone, Eq, PartialEq, Hash, Debug)]
/// struct NodeData(i32);
/// impl BidirectedData for NodeData {
///     fn mirror(&self) -> Self {
///         Self(1000 - self.0)
///     }
/// }
///
/// let mut graph = PetGraph::new();
/// let n1 = graph.add_node(NodeData(0));
/// let n2 = graph.add_node(NodeData(1000));
/// graph.add_edge(n1.clone(), n2.clone(), ());
/// graph.add_edge(n2.clone(), n1.clone(), ());
/// let bigraph = NodeBigraphWrapper::new(graph);
/// debug_assert_eq!(Some(n2.clone()), bigraph.mirror_node(n1.clone()));
/// debug_assert_eq!(Some(n1.clone()), bigraph.mirror_node(n2.clone()));
/// ```
#[derive(Debug, Clone)]
pub struct NodeBigraphWrapper<Topology: GraphBase> {
    /// The underlying topology of the bigraph.
    pub topology: Topology,
    binode_map: Vec<Topology::OptionalNodeIndex>,
}

impl<Topology: GraphBase> GraphBase for NodeBigraphWrapper<Topology> {
    type NodeData = Topology::NodeData;
    type EdgeData = Topology::EdgeData;
    type OptionalNodeIndex = Topology::OptionalNodeIndex;
    type OptionalEdgeIndex = Topology::OptionalEdgeIndex;
    type NodeIndex = Topology::NodeIndex;
    type EdgeIndex = Topology::EdgeIndex;
}

impl<Topology: GraphBase + ImmutableGraphContainer> NodeBigraphWrapper<Topology> {
    /// Converts the given topology into a bigraph without any mapping.
    /// This leaves the resulting type in a potentially illegal state.
    pub fn new_unmapped(topology: Topology) -> Self {
        let node_count = topology.node_count();
        Self {
            topology,
            binode_map: vec![<Self as GraphBase>::OptionalNodeIndex::new_none(); node_count],
        }
    }
}

impl<Topology: StaticGraph> StaticBigraph for NodeBigraphWrapper<Topology> {
    fn mirror_node(&self, node_id: Self::NodeIndex) -> Option<Self::NodeIndex> {
        self.binode_map[node_id.as_usize()].into()
    }
}

impl<Topology: StaticGraph> NodeBigraphWrapper<Topology>
where
    <Self as GraphBase>::NodeData: BidirectedData + Eq + Hash + Debug,
{
    fn new_internal(topology: Topology, checked: bool) -> Self {
        //let mut data_map = HashMap::new();
        let mut data_map: HashMap<<Self as GraphBase>::NodeData, <Self as GraphBase>::NodeIndex> =
            HashMap::new();
        let mut binode_map =
            vec![<Self as GraphBase>::OptionalNodeIndex::new_none(); topology.node_count()];

        for node_index in topology.node_indices() {
            let node_data = topology.node_data(node_index);

            if let Some(mirror_index) = data_map.get(node_data).cloned() {
                //debug_assert_ne!(node_index, mirror_index);
                debug_assert!(!binode_map[node_index.as_usize()].is_valid());
                debug_assert!(!binode_map[mirror_index.as_usize()].is_valid());
                debug_assert_eq!(node_data, &topology.node_data(mirror_index).mirror());
                binode_map[node_index.as_usize()] = mirror_index.into();
                binode_map[mirror_index.as_usize()] = node_index.into();
                data_map.remove(node_data);
            } else {
                let mirror_data = node_data.mirror();
                //debug_assert_ne!(&mirror_data, node_data);
                debug_assert_eq!(None, data_map.insert(mirror_data, node_index));
            }
        }

        if checked {
            debug_assert!(data_map.is_empty());
        } else {
            for node_index in topology.node_indices() {
                let node_data = topology.node_data(node_index);
                debug_assert!(!data_map.contains_key(node_data));
            }
        }
        Self {
            topology,
            binode_map,
        }
    }
}

impl<Topology: StaticGraph> StaticBigraphFromDigraph for NodeBigraphWrapper<Topology>
where
    <Self as GraphBase>::NodeData: BidirectedData + Eq + Hash + Debug,
{
    type Topology = Topology;

    fn new(topology: Self::Topology) -> Self {
        Self::new_internal(topology, true)
    }

    fn new_unchecked(topology: Self::Topology) -> Self {
        Self::new_internal(topology, false)
    }
}

impl<Topology: DynamicGraph> DynamicBigraph for NodeBigraphWrapper<Topology> {
    fn set_mirror_nodes(&mut self, a: Self::NodeIndex, b: Self::NodeIndex) {
        debug_assert!(self.contains_node_index(a));
        debug_assert!(self.contains_node_index(b));
        self.binode_map[a.as_usize()] = b.into();
        self.binode_map[b.as_usize()] = a.into();
    }
}

impl<Topology: ImmutableGraphContainer> ImmutableGraphContainer for NodeBigraphWrapper<Topology> {
    fn node_indices(&self) -> GraphIndices<Self::NodeIndex, Self::OptionalNodeIndex> {
        self.topology.node_indices()
    }

    fn edge_indices(&self) -> GraphIndices<Self::EdgeIndex, Self::OptionalEdgeIndex> {
        self.topology.edge_indices()
    }

    fn contains_node_index(&self, node_index: Self::NodeIndex) -> bool {
        self.topology.contains_node_index(node_index)
    }

    fn contains_edge_index(&self, edge_index: Self::EdgeIndex) -> bool {
        self.topology.contains_edge_index(edge_index)
    }

    fn node_count(&self) -> usize {
        self.topology.node_count()
    }

    fn edge_count(&self) -> usize {
        self.topology.edge_count()
    }

    fn node_data(&self, node_id: Self::NodeIndex) -> &Self::NodeData {
        self.topology.node_data(node_id)
    }

    fn edge_data(&self, edge_id: Self::EdgeIndex) -> &Self::EdgeData {
        self.topology.edge_data(edge_id)
    }

    fn node_data_mut(&mut self, node_id: Self::NodeIndex) -> &mut Self::NodeData {
        self.topology.node_data_mut(node_id)
    }

    fn edge_data_mut(&mut self, edge_id: Self::EdgeIndex) -> &mut Self::EdgeData {
        self.topology.edge_data_mut(edge_id)
    }

    fn contains_edge_between(&self, from: Self::NodeIndex, to: Self::NodeIndex) -> bool {
        self.topology.contains_edge_between(from, to)
    }

    fn edge_count_between(&self, from: Self::NodeIndex, to: Self::NodeIndex) -> usize {
        self.topology.edge_count_between(from, to)
    }

    fn edge_endpoints(&self, edge_id: Self::EdgeIndex) -> Edge<Self::NodeIndex> {
        self.topology.edge_endpoints(edge_id)
    }
}

impl<Topology: MutableGraphContainer + StaticGraph> MutableGraphContainer
    for NodeBigraphWrapper<Topology>
{
    fn add_node(&mut self, node_data: Self::NodeData) -> Self::NodeIndex {
        self.binode_map.push(Self::OptionalNodeIndex::new_none());
        self.topology.add_node(node_data)
    }

    fn add_edge(
        &mut self,
        from: Self::NodeIndex,
        to: Self::NodeIndex,
        edge_data: Self::EdgeData,
    ) -> Self::EdgeIndex {
        self.topology.add_edge(from, to, edge_data)
    }

    fn remove_node(&mut self, node_id: Self::NodeIndex) -> Option<Self::NodeData> {
        if let Some(mirror_node_id) = self.mirror_node(node_id) {
            self.binode_map[mirror_node_id.as_usize()] = Self::OptionalNodeIndex::new_none();
        }

        self.binode_map.remove(node_id.as_usize());
        for mirror_node_id in &mut self.binode_map {
            if let Some(mirror_node_usize) = mirror_node_id.as_usize() {
                assert_ne!(mirror_node_usize, node_id.as_usize());
                if mirror_node_usize > node_id.as_usize() {
                    *mirror_node_id = (mirror_node_usize - 1).into();
                }
            }
        }

        self.topology.remove_node(node_id)
    }

    fn remove_nodes_sorted_slice(&mut self, node_ids: &[Self::NodeIndex]) {
        debug_assert!(node_ids.windows(2).all(|w| w[0] < w[1]));
        let mut decrement_map = Vec::new();
        for (index, &node_id) in node_ids.iter().enumerate() {
            // build map to decrement mirror nodes by the right amount
            while decrement_map.len() < node_id.as_usize() {
                decrement_map.push(index);
            }

            // remove mirror references of deleted nodes
            if let Some(mirror_node_id) = self.mirror_node(node_id) {
                self.binode_map[mirror_node_id.as_usize()] = Self::OptionalNodeIndex::new_none();
            }
        }

        while decrement_map.len() < self.binode_map.len() {
            decrement_map.push(node_ids.len());
        }

        // decrement mirror nodes to match new node indices
        let mut index = 0;
        for node_id in 0..self.binode_map.len() {
            if let Some(remove_node_id) = node_ids.get(index) {
                if remove_node_id.as_usize() == node_id {
                    index += 1;
                    continue;
                }
            }

            let binode_id = self.binode_map[node_id]
                .as_usize()
                .map(|binode_id_usize| binode_id_usize - decrement_map[binode_id_usize])
                .into();

            self.binode_map[node_id - index] = binode_id;
        }
        self.binode_map.resize(
            self.binode_map.len() - node_ids.len(),
            Option::<usize>::None.into(),
        );

        self.topology.remove_nodes_sorted_slice(node_ids)
    }

    fn remove_edge(&mut self, edge_id: Self::EdgeIndex) -> Option<Self::EdgeData> {
        self.topology.remove_edge(edge_id)
    }

    fn remove_edges_sorted(&mut self, edge_ids: &[Self::EdgeIndex]) {
        self.topology.remove_edges_sorted(edge_ids)
    }

    fn clear(&mut self) {
        self.topology.clear();
        self.binode_map.clear();
    }
}

impl<Topology: NavigableGraph> NavigableGraph for NodeBigraphWrapper<Topology> {
    type OutNeighbors<'a> = <Topology as NavigableGraph>::OutNeighbors<'a> where Self: 'a, Topology: 'a;
    type InNeighbors<'a> = <Topology as NavigableGraph>::InNeighbors<'a> where Self: 'a, Topology: 'a;
    type EdgesBetween<'a> = <Topology as NavigableGraph>::EdgesBetween<'a> where Self: 'a, Topology: 'a;

    fn out_neighbors(&self, node_id: Self::NodeIndex) -> Self::OutNeighbors<'_> {
        self.topology.out_neighbors(node_id)
    }

    fn in_neighbors(&self, node_id: Self::NodeIndex) -> Self::InNeighbors<'_> {
        self.topology.in_neighbors(node_id)
    }

    fn edges_between(
        &self,
        from_node_id: Self::NodeIndex,
        to_node_id: Self::NodeIndex,
    ) -> Self::EdgesBetween<'_> {
        self.topology.edges_between(from_node_id, to_node_id)
    }
}

impl<Topology: Default + GraphBase> Default for NodeBigraphWrapper<Topology> {
    fn default() -> Self {
        Self {
            topology: Topology::default(),
            binode_map: vec![],
        }
    }
}

impl<Topology: StaticGraph> StaticNodeCentricBigraph for NodeBigraphWrapper<Topology> {}

impl<Topology: StaticGraph> StaticEdgeCentricBigraph for NodeBigraphWrapper<Topology> where
    <Topology as GraphBase>::EdgeData: BidirectedData + Eq
{
}

impl<Topology: DynamicGraph> DynamicNodeCentricBigraph for NodeBigraphWrapper<Topology>
where
    <Topology as GraphBase>::NodeData: BidirectedData,
    <Topology as GraphBase>::EdgeData: Clone,
{
}

impl<Topology: DynamicGraph> DynamicEdgeCentricBigraph for NodeBigraphWrapper<Topology> where
    <Topology as GraphBase>::EdgeData: BidirectedData + Eq
{
}

impl<Topology: GraphBase + PartialEq> PartialEq for NodeBigraphWrapper<Topology> {
    fn eq(&self, other: &Self) -> bool {
        self.topology == other.topology && self.binode_map == other.binode_map
    }
}

impl<Topology: GraphBase + Eq> Eq for NodeBigraphWrapper<Topology> {}

#[cfg(test)]
mod tests {
    use super::NodeBigraphWrapper;
    use crate::interface::dynamic_bigraph::{DynamicBigraph, DynamicNodeCentricBigraph};
    use crate::interface::static_bigraph::StaticEdgeCentricBigraph;
    use crate::interface::{
        static_bigraph::StaticBigraph, static_bigraph::StaticBigraphFromDigraph, BidirectedData,
    };
    use crate::traitgraph::interface::{ImmutableGraphContainer, MutableGraphContainer};
    use traitgraph::implementation::petgraph_impl::PetGraph;
    use traitgraph::index::OptionalGraphIndex;

    #[test]
    fn test_bigraph_creation() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        let n3 = graph.add_node(NodeData(2));
        let n4 = graph.add_node(NodeData(3));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        let bigraph = NodeBigraphWrapper::new(graph);

        debug_assert_eq!(Some(n2), bigraph.mirror_node(n1));
        debug_assert_eq!(Some(n1), bigraph.mirror_node(n2));
        debug_assert_eq!(Some(n4), bigraph.mirror_node(n3));
        debug_assert_eq!(Some(n3), bigraph.mirror_node(n4));
    }

    #[test]
    #[should_panic]
    fn test_bigraph_creation_unmapped_node() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        graph.add_node(NodeData(2));
        graph.add_node(NodeData(3));
        graph.add_node(NodeData(4));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        NodeBigraphWrapper::new(graph);
    }

    #[test]
    #[should_panic]
    fn test_bigraph_creation_wrongly_mapped_node() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 == 4 {
                    3
                } else if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        graph.add_node(NodeData(2));
        graph.add_node(NodeData(3));
        graph.add_node(NodeData(4));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        NodeBigraphWrapper::new(graph);
    }

    #[test]
    #[should_panic]
    fn test_bigraph_creation_self_mapped_node_without_mirror() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 == 4 {
                    4
                } else if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        graph.add_node(NodeData(2));
        graph.add_node(NodeData(3));
        graph.add_node(NodeData(4));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        NodeBigraphWrapper::new(graph);
    }

    #[test]
    #[should_panic]
    fn test_bigraph_creation_self_mapped_node_with_mirror() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 == 4 {
                    4
                } else if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        graph.add_node(NodeData(2));
        graph.add_node(NodeData(3));
        graph.add_node(NodeData(4));
        graph.add_node(NodeData(5));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        NodeBigraphWrapper::new(graph);
    }

    #[test]
    fn test_bigraph_unchecked_creation() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        let n3 = graph.add_node(NodeData(2));
        let n4 = graph.add_node(NodeData(3));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        let bigraph = NodeBigraphWrapper::new_unchecked(graph);

        debug_assert_eq!(Some(n2), bigraph.mirror_node(n1));
        debug_assert_eq!(Some(n1), bigraph.mirror_node(n2));
        debug_assert_eq!(Some(n4), bigraph.mirror_node(n3));
        debug_assert_eq!(Some(n3), bigraph.mirror_node(n4));
    }

    #[test]
    fn test_bigraph_unchecked_creation_unmapped_node() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        let n3 = graph.add_node(NodeData(2));
        let n4 = graph.add_node(NodeData(3));
        let n5 = graph.add_node(NodeData(4));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        let bigraph = NodeBigraphWrapper::new_unchecked(graph);

        debug_assert_eq!(Some(n2), bigraph.mirror_node(n1));
        debug_assert_eq!(Some(n1), bigraph.mirror_node(n2));
        debug_assert_eq!(Some(n4), bigraph.mirror_node(n3));
        debug_assert_eq!(Some(n3), bigraph.mirror_node(n4));
        debug_assert_eq!(None, bigraph.mirror_node(n5));
    }

    #[test]
    #[should_panic]
    fn test_bigraph_unchecked_creation_wrongly_mapped_node_at_end() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 == 4 {
                    3
                } else if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        graph.add_node(NodeData(2));
        graph.add_node(NodeData(3));
        graph.add_node(NodeData(4));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        NodeBigraphWrapper::new_unchecked(graph);
    }

    #[test]
    #[should_panic]
    fn test_bigraph_unchecked_creation_wrongly_mapped_node_at_beginning() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 == 4 {
                    3
                } else if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        graph.add_node(NodeData(4));
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        graph.add_node(NodeData(2));
        graph.add_node(NodeData(3));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        NodeBigraphWrapper::new_unchecked(graph);
    }

    #[test]
    #[should_panic]
    fn test_bigraph_unchecked_creation_self_mapped_node_without_mirror() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 == 4 {
                    4
                } else if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        graph.add_node(NodeData(2));
        graph.add_node(NodeData(3));
        graph.add_node(NodeData(4));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        NodeBigraphWrapper::new_unchecked(graph);
    }

    #[test]
    #[should_panic]
    fn test_bigraph_unchecked_creation_self_mapped_node_with_mirror() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 == 4 {
                    4
                } else if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        graph.add_node(NodeData(2));
        graph.add_node(NodeData(3));
        graph.add_node(NodeData(4));
        graph.add_node(NodeData(5));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        NodeBigraphWrapper::new_unchecked(graph);
    }

    #[test]
    fn test_bigraph_verification() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        graph.add_node(NodeData(2));
        graph.add_node(NodeData(3));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        let bigraph = NodeBigraphWrapper::new(graph);
        debug_assert!(bigraph.verify_node_pairing_without_self_mirrors());
        debug_assert!(bigraph.verify_node_pairing());
    }

    #[test]
    fn test_bigraph_verification_self_mapped_node() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        graph.add_node(NodeData(2));
        graph.add_node(NodeData(3));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        let mut bigraph = NodeBigraphWrapper::new(graph);
        bigraph.topology.add_node(NodeData(4));
        bigraph.binode_map.push(4usize.into());
        debug_assert!(!bigraph.verify_node_pairing_without_self_mirrors());
        debug_assert!(bigraph.verify_node_pairing());
    }

    #[test]
    fn test_bigraph_verification_self_unmapped_node() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        graph.add_node(NodeData(2));
        graph.add_node(NodeData(3));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        let mut bigraph = NodeBigraphWrapper::new(graph);
        bigraph.topology.add_node(NodeData(4));
        bigraph.binode_map.push(OptionalGraphIndex::new_none());
        debug_assert!(!bigraph.verify_node_pairing_without_self_mirrors());
        debug_assert!(!bigraph.verify_node_pairing());
    }

    #[test]
    fn test_bigraph_verification_wrongly_mapped_node() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct NodeData(i32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(if self.0 % 2 == 0 {
                    self.0 + 1
                } else {
                    self.0 - 1
                })
            }
        }

        let mut graph = PetGraph::new();
        let n1 = graph.add_node(NodeData(0));
        let n2 = graph.add_node(NodeData(1));
        graph.add_node(NodeData(2));
        graph.add_node(NodeData(3));
        graph.add_edge(n1, n2, ()); // Just to fix the EdgeData type parameter
        let mut bigraph = NodeBigraphWrapper::new(graph);
        bigraph.topology.add_node(NodeData(4));
        bigraph.binode_map.push(3usize.into());
        debug_assert!(!bigraph.verify_node_pairing_without_self_mirrors());
        debug_assert!(!bigraph.verify_node_pairing());
    }

    #[test]
    fn test_bigraph_add_mirror_nodes() {
        #[derive(Eq, PartialEq, Debug, Hash, Clone)]
        struct NodeData(u32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(1000 - self.0)
            }
        }

        let mut graph = PetGraph::new();
        let n0 = graph.add_node(NodeData(0));
        let n1 = graph.add_node(NodeData(1));
        graph.add_node(NodeData(2));
        graph.add_node(NodeData(3));
        graph.add_node(NodeData(997));
        graph.add_edge(n0, n1, ());
        let mut graph = NodeBigraphWrapper::new_unchecked(graph);
        debug_assert!(!graph.verify_node_pairing());
        graph.add_mirror_nodes();
        debug_assert!(graph.verify_node_pairing());
        debug_assert_eq!(graph.node_count(), 8);
    }

    #[test]
    fn test_bigraph_remove_node() {
        #[derive(Eq, PartialEq, Debug, Hash, Clone)]
        struct NodeData(u32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(1000 - self.0)
            }
        }

        let mut graph = NodeBigraphWrapper::new(PetGraph::new());
        let n0 = graph.add_node(NodeData(0));
        let n1 = graph.add_node(NodeData(1000));
        graph.set_mirror_nodes(n0, n1);
        let n2 = graph.add_node(NodeData(1));
        let n3 = graph.add_node(NodeData(999));
        graph.set_mirror_nodes(n2, n3);
        let _e0 = graph.add_edge(n0, n2, ());
        let _e1 = graph.add_edge(n3, n1, ());

        assert!(graph.verify_node_pairing());
        assert!(graph.verify_edge_mirror_property());

        graph.remove_node(n0);

        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.mirror_node(n0), None);
        assert_eq!(graph.mirror_node(n1), Some(n2));
        assert_eq!(graph.mirror_node(n2), Some(n1));

        graph.remove_node(n0);

        assert!(graph.verify_node_pairing());
        assert!(graph.verify_edge_mirror_property());
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.mirror_node(n0), Some(n1));
        assert_eq!(graph.mirror_node(n1), Some(n0));
    }

    #[test]
    fn test_bigraph_remove_nodes() {
        #[derive(Eq, PartialEq, Debug, Hash, Clone)]
        struct NodeData(u32);
        impl BidirectedData for NodeData {
            fn mirror(&self) -> Self {
                Self(1000 - self.0)
            }
        }

        let mut graph = NodeBigraphWrapper::new(PetGraph::new());
        for i in 0..100 {
            graph.add_node(NodeData(i));
        }
        graph.add_mirror_nodes();

        fn random(n: usize) -> usize {
            // hopefully results in a distribution of edges that covers all possible edge cases
            n.wrapping_mul(31)
                .wrapping_add(n.wrapping_mul(97))
                .wrapping_add(n / 31)
                .wrapping_add(n / 97)
                .wrapping_add(n)
                .wrapping_add(n.count_zeros() as usize)
        }

        let mut n = 0;
        while graph.edge_count() < 250 {
            let n1 = (n % graph.node_count()).into();
            n = random(n);
            let n2 = (n % graph.node_count()).into();
            n = random(n);

            if !graph.contains_edge_between(n1, n2) {
                graph.add_edge(n1, n2, ());
            }
        }
        graph.add_node_centric_mirror_edges();

        let mut g1 = graph.clone();
        let mut g2 = graph.clone();
        let remove: Vec<_> = [
            0, 1, 2, 5, 7, 24, 35, 36, 37, 38, 39, 40, 77, 88, 99, 100, 101, 102, 133, 134, 135,
            136, 188, 199,
        ]
        .into_iter()
        .map(Into::into)
        .collect();
        g1.remove_nodes_sorted_slice(&remove);
        for n in remove.into_iter().rev() {
            g2.remove_node(n);
        }
        assert!(g1.eq(&g2), "g1: {:?}\ng2: {:?}", g1, g2);

        let mut g1 = graph.clone();
        let mut g2 = graph.clone();
        let remove: Vec<_> = [
            2, 5, 7, 24, 35, 36, 37, 38, 39, 40, 77, 88, 99, 100, 101, 102,
        ]
        .into_iter()
        .map(Into::into)
        .collect();
        g1.remove_nodes_sorted_slice(&remove);
        for n in remove.into_iter().rev() {
            g2.remove_node(n);
        }
        assert!(g1.eq(&g2), "g1: {:?}\ng2: {:?}", g1, g2);
    }
}
