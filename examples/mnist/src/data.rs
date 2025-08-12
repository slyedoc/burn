use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*, vision::Transform2D,
};

#[derive(Clone, Debug, Default)]
pub struct MnistBatcher {}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert::<B::FloatElem>(), device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // normalize: make between [0,1] and make the mean =  0 and std = 1
            // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([(item.label as i64).elem::<B::IntElem>()]),
                    device,
                )
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        let t = Transform2D::translation(1, 1);

        MnistBatch { images, targets }
    }
}
