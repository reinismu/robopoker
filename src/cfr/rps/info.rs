use crate::cfr::rps::action::RpsAction;
use crate::cfr::rps::bucket::RpsBucket;
use crate::cfr::rps::node::RpsNode;
use crate::cfr::rps::player::RpsPlayer;
use crate::cfr::traits::tree::info::Info;

/// Indistinguishable states belonging to same InfoSets. Effectively, distribution of possile opponent actions.
#[derive(PartialEq, Eq)]
pub(crate) struct RpsInfo<'t> {
    roots: Vec<&'t RpsNode<'t>>,
}

impl<'t> Info for RpsInfo<'t> {
    type IPlayer = RpsPlayer;
    type IAction = RpsAction;
    type IBucket = RpsBucket;
    type INode = RpsNode<'t>;
    fn roots(&self) -> &Vec<&Self::INode> {
        &self.roots
    }
    fn bucket(&self) -> Self::IBucket {
        RpsBucket {}
    }
}
