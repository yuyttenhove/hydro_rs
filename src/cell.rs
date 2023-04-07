use glam::DVec3;
use rstar::{primitives::GeomWithData, RTree};

use crate::part::Particle;

pub struct Cell {
    pub loc: DVec3,
    pub part_offset: usize,
    pub part_count: usize,
    ngb_cid: Vec<usize>,
    ngb_shift: Vec<DVec3>,
    tree: RTree<TreeNode>,
}

impl Cell {
    pub fn new(loc: DVec3, ngb_cid: Vec<usize>, ngb_shift: Vec<DVec3>) -> Self {
        Self {
            part_offset: 0,
            part_count: 0,
            ngb_cid,
            ngb_shift,
            loc,
            tree: RTree::new(),
        }
    }

    pub fn build_search_tree(&self, cells: &[Cell], parts: &[Particle]) -> RTree<TreeNode> {
        let mut tree_objects = vec![];

        // Add this cells parts to the search tree
        let this_parts = &parts[self.part_offset..(self.part_offset + self.part_count)];
        for (i, part) in this_parts.iter().enumerate() {
            let pid = i + self.part_offset;
            tree_objects.push(TreeNode::new((part.loc + DVec3::ZERO).to_array(), pid));
        }

        // Add the neighbouring cells parts to the search tree
        for (cid, shift) in self.ngb_cid.iter().zip(self.ngb_shift.iter()) {
            let ngb_cell = &cells[*cid];
            let ngb_parts =
                &parts[ngb_cell.part_offset..(ngb_cell.part_offset + ngb_cell.part_count)];
            for (i, part) in ngb_parts.iter().enumerate() {
                let pid = i + ngb_cell.part_offset;
                tree_objects.push(TreeNode::new((part.loc + *shift).to_array(), pid));
            }
        }
        RTree::bulk_load(tree_objects)
    }

    pub fn assign_search_tree(&mut self, tree: RTree<TreeNode>) {
        self.tree = tree;
    }

    pub fn nn_iter(&self, loc: DVec3) -> Box<dyn Iterator<Item = (DVec3, usize)> + '_> {
        let mut iter = self.tree.nearest_neighbor_iter(&loc.to_array());

        // skip the first element (will always be the query site itself)
        let first = iter.next().expect("Search tree cannot be empty");
        debug_assert_eq!(DVec3::from_slice(first.geom()), loc);

        // Return the rest of the iterator
        Box::new(iter.map(|node| (DVec3::from_slice(node.geom()), node.data)))
    }
}

type TreeNode = GeomWithData<[f64; 3], usize>;
