use crate::play::action::Action;
use std::hash::Hash;

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub enum Edge {
    Choice(Action),
    Chance,
}

impl From<Action> for Edge {
    fn from(action: Action) -> Self {
        match action {
            Action::Draw(_) | Action::Blind(_) => Self::Chance,
            _ => Self::Choice(action),
        }
    }
}

impl From<u32> for Edge {
    fn from(_: u32) -> Self {
        todo!("serialization mechanism")
    }
}

impl From<Edge> for u32 {
    fn from(_: Edge) -> Self {
        todo!("serialization mechanism")
    }
}
