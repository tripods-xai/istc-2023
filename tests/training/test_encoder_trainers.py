import torch

from src.encoders import AffineConvolutionalEncoder, StreamedTurboEncoder
from src.interleavers import BatchRandomPermuteInterleaver, FixedPermuteInterleaver
from src.training import CodebookEncoderTrainer, BCJRTurboTrainer
from src.utils import DeviceManager
from src.channels import AWGN

from ..channels import create_fixed_noise_channel


def test_forward_batch_random():
    manager = DeviceManager(seed=2090)

    msg_length = 10
    base_encoder1 = AffineConvolutionalEncoder(
        torch.FloatTensor([[1, 1, 1], [1, 0, 1]]).to(manager.device),
        torch.FloatTensor([0, 0]).to(manager.device),
        num_steps=msg_length,
        device_manager=manager,
    )
    base_encoder2 = base_encoder1.get_encoder_channels([1])
    interleaver = BatchRandomPermuteInterleaver(msg_length, device_manager=manager)
    encoder = StreamedTurboEncoder(
        base_encoder1,
        base_encoder2,
        interleaver=interleaver,
        device_manager=manager,
    )
    channel = create_fixed_noise_channel(AWGN, torch.tensor(1.0))(
        snr=1.0, device_manager=manager
    )
    validation_channel = create_fixed_noise_channel(AWGN, torch.tensor(2.0))(
        snr=1.0, device_manager=manager
    )

    trainer = CodebookEncoderTrainer(
        input_size=msg_length,
        encoder=encoder,
        channel=channel,
        validation_channel=validation_channel,
        output_path=None,
        batch_normalization=False,
        device_manager=manager,
        use_inputs_for_loss=False,
    )

    msg = torch.randint(
        0,
        2,
        size=(100, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )
    logits, _ = trainer.forward(msg, validate=False)
    logits_2, _ = trainer.forward(msg, validate=False)
    assert not torch.allclose(logits_2, logits)

    logits, _ = trainer.forward(msg, validate=True)
    logits_2, _ = trainer.forward(msg, validate=True)
    assert not torch.allclose(logits_2, logits)


def test_forward_batch_random_codebook_matches_BCJR():
    manager = DeviceManager(seed=2090)

    msg_length = 10
    base_encoder1 = AffineConvolutionalEncoder(
        torch.FloatTensor([[1, 1, 1], [1, 0, 1]]).to(manager.device),
        torch.FloatTensor([0, 0]).to(manager.device),
        num_steps=msg_length,
        device_manager=manager,
    )
    base_encoder2 = base_encoder1.get_encoder_channels([1])
    interleaver = BatchRandomPermuteInterleaver(msg_length, device_manager=manager)
    encoder = StreamedTurboEncoder(
        base_encoder1,
        base_encoder2,
        interleaver=interleaver,
        device_manager=manager,
    )
    channel = create_fixed_noise_channel(AWGN, torch.tensor(1.0))(
        snr=1.0, device_manager=manager
    )
    validation_channel = create_fixed_noise_channel(AWGN, torch.tensor(2.0))(
        snr=1.0, device_manager=manager
    )

    codebook_trainer = CodebookEncoderTrainer(
        input_size=msg_length,
        encoder=encoder,
        channel=channel,
        validation_channel=validation_channel,
        output_path=None,
        batch_normalization=False,
        device_manager=manager,
        use_inputs_for_loss=False,
    )

    msg = torch.randint(
        0,
        2,
        size=(100, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )
    logits, _ = codebook_trainer.forward(msg, validate=False)
    fixed_interleaver = FixedPermuteInterleaver(
        msg_length, device_manager=manager, permutation=interleaver._permutation
    )
    fixed_encoder = StreamedTurboEncoder(
        base_encoder1,
        base_encoder2,
        interleaver=fixed_interleaver,
        device_manager=manager,
    )
    codebook_trainer_fixed = CodebookEncoderTrainer(
        input_size=msg_length,
        encoder=fixed_encoder,
        channel=channel,
        validation_channel=validation_channel,
        output_path=None,
        batch_normalization=False,
        device_manager=manager,
        use_inputs_for_loss=False,
    )
    logits_2, _ = codebook_trainer_fixed.forward(msg, validate=False)
    assert torch.allclose(logits_2, logits)
