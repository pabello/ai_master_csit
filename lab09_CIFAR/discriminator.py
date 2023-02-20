import utils
import tensorflow_addons as tfa
import tensorflow_datasets as tfds


if __name__ == "__main__":
    train_dataset, test_dataset = tfds.load(name="cifar10", split=['train', 'test'], as_supervised=True)
    train_dataset, test_dataset = train_dataset.shuffle(1024).batch(32), test_dataset.shuffle(1024).batch(32)
    train_dataset, test_dataset = train_dataset.map(utils.normalize_img), test_dataset.map(utils.normalize_img)

    discriminator = utils.initialize_discriminator()

    discriminator.compile(optimizer='adam', loss=tfa.losses.TripletSemiHardLoss())
    history = discriminator.fit(train_dataset, epochs=5, validation_data=test_dataset)

    discriminator.save_weights("discriminator_weights.h5")