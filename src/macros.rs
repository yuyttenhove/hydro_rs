macro_rules! create_attr {
    ($group:expr, $data:expr, $name:expr) => {
        $group.new_attr_builder().with_data(&$data).create($name)
    };
}

macro_rules! create_dataset {
    ($group:expr, $data_iter:expr, $name:expr) => {
        $group
            .new_dataset_builder()
            .with_data(&$data_iter.collect::<Vec<_>>())
            .create($name)
    };
}

pub(crate) use create_attr;
pub(crate) use create_dataset;
