import itertools as it
import torch

from src.interleavers import FixedPermuteInterleaver
from src.utils import DeviceManager, DEFAULT_DEVICE_MANAGER
from src.channels import AWGN

from src.training import *

from ..channels import create_fixed_noise_channel
from ..utils import test_manager


def generate_agent(
    input_size: int,
    no_interleaver=True,
    noiseless=True,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
):
    if no_interleaver:
        interleaver = FixedPermuteInterleaver(
            input_size=input_size,
            permutation=torch.arange(input_size, device=device_manager.device),
            device_manager=device_manager,
        )
    else:
        interleaver = FixedPermuteInterleaver(
            input_size=input_size, device_manager=device_manager
        )
    if noiseless:
        channel_type = create_fixed_noise_channel(AWGN, additive_noise=0.0)
    else:
        channel_type = AWGN
    channel = channel_type(snr=0.0, device_manager=device_manager)

    return TurboTableTrainerBCJR(
        input_size=input_size,
        window=5,
        interleaver=interleaver,
        channel=channel,
        validation_channel=channel,
        output_path=None,
        num_interleaved_streams=2,
        num_noninterleaved_streams=1,
        num_iter=6,
        use_max=False,
        device_manager=device_manager,
        constraint="unit_power",
        init_method="normal",
    )


def test_swarm_basic():
    block_len = 10
    num_agents = 10

    agents = [
        generate_agent(input_size=block_len, device_manager=test_manager)
        for _ in range(num_agents)
    ]
    masses = torch.arange(
        num_agents,
        dtype=torch.float32,
        device=test_manager.device,
    )
    swarm = L2DistanceSwarm(agents, masses=masses, device_manager=test_manager)

    for agent, other_agent in it.combinations(agents, r=2):
        assert abs(swarm.agent_distance(agent, other_agent).item()) > 1e-9

    assert swarm.agents == agents
    assert len(swarm) == num_agents
    assert torch.all(swarm.masses == masses)
    assert swarm.tolm == 1e-4
    assert swarm.merge_agents
    assert not swarm.kill_agents
    assert swarm.tolmerge == 1e-3
    assert swarm.communication_adj == 2
    assert swarm.step_adj == 1
    assert swarm.descent == 0.2
    assert swarm.shrinkage == 0.9
    assert swarm.failure == 10
    assert swarm.device_manager is test_manager
    assert swarm.heaviest_agent_i() == (num_agents - 1)


def test_swarm_same_generator():
    block_len = 10
    num_agents = 10

    agents = [
        generate_agent(input_size=block_len, device_manager=test_manager.clone())
        for _ in range(num_agents)
    ]
    masses = torch.randn(
        (num_agents,), device=test_manager.device, generator=test_manager.generator
    )
    swarm = L2DistanceSwarm(agents, masses=masses, device_manager=test_manager)

    for agent, other_agent in it.combinations(agents, r=2):
        assert abs(swarm.agent_distance(agent, other_agent).item()) < 1e-9


def test_swarm_clone():
    block_len = 10
    num_agents = 10

    agents = [
        generate_agent(input_size=block_len, device_manager=test_manager.clone())
        for _ in range(num_agents)
    ]
    masses = torch.randn(
        (num_agents,), device=test_manager.device, generator=test_manager.generator
    )
    swarm = L2DistanceSwarm(agents, masses=masses, device_manager=test_manager)
    new_swarm = swarm.clone()
    assert swarm is not new_swarm
    assert swarm.masses is not new_swarm.masses
    assert swarm.agents is new_swarm.agents
    assert torch.all(swarm.masses == new_swarm.masses)


def test_swarm_merge():
    block_len = 10
    num_agents = 10

    agents = [
        generate_agent(input_size=block_len, device_manager=test_manager)
        for _ in range(num_agents)
    ]
    merge_set_1 = [3, 5, 9]
    base_params = list(agents[0].parameters())
    for i in merge_set_1:
        params = list(agents[i].parameters())
        for p, bp in zip(params, base_params):
            noise = torch.randn(p.shape, generator=test_manager.generator)
            noise = noise / torch.norm(noise, p=2) * 1e-4
            p.data = bp.data + noise
    merge_set_2 = [2, 7]
    base_params = list(agents[8].parameters())
    for i in merge_set_2:
        params = list(agents[i].parameters())
        for p, bp in zip(params, base_params):
            noise = torch.randn(p.shape, generator=test_manager.generator)
            noise = noise / torch.norm(noise, p=2) * 1e-4
            p.data = bp.data + noise

    masses = torch.arange(
        num_agents,
        dtype=torch.float32,
        device=test_manager.device,
    )
    swarm = L2DistanceSwarm(
        agents, masses=masses, merge_agents=True, device_manager=test_manager
    )

    new_swarm = swarm.merge()

    assert len(new_swarm) == 5
    expected_masses = torch.tensor(
        [3 + 5 + 9, 1, 2 + 7 + 8, 4, 6], dtype=int, device=test_manager.device
    )
    assert torch.all(new_swarm.masses == expected_masses)
    assert new_swarm.agents == [agents[0], agents[1], agents[2], agents[4], agents[6]]


def test_swarm_kill():
    block_len = 10
    num_agents = 10

    agents = [
        generate_agent(input_size=block_len, device_manager=test_manager)
        for _ in range(num_agents)
    ]
    masses = torch.arange(
        num_agents,
        dtype=torch.float32,
        device=test_manager.device,
    )
    masses[3] = 1e-4 / num_agents
    masses[4] = 1e-4 / (num_agents - 2)
    swarm = L2DistanceSwarm(
        agents, masses=masses, kill_agents=True, device_manager=test_manager
    )

    new_swarm = swarm.kill()
    assert len(new_swarm) == num_agents - 2
    assert new_swarm.agents == agents[1:3] + agents[4:]

    new_swarm = new_swarm.kill()
    assert len(new_swarm) == num_agents - 3
    assert new_swarm.agents == agents[1:3] + agents[5:]


def test_swarm_mass_transition():
    block_len = 10
    num_agents = 10

    agents = [
        generate_agent(input_size=block_len, device_manager=test_manager)
        for _ in range(num_agents)
    ]
    masses = torch.arange(
        num_agents,
        dtype=torch.float32,
        device=test_manager.device,
    )
    swarm = L2DistanceSwarm(agents, masses=masses, device_manager=test_manager)
    losses = torch.arange(
        start=num_agents - 1,
        end=-1,
        step=-1,
        dtype=torch.float32,
        device=test_manager.device,
    )

    new_swarm = swarm.mass_transition(losses=losses)
    assert torch.allclose(torch.sum(new_swarm.masses), torch.sum(masses))
    for i in range(num_agents - 1):
        multiplier = 1 - (losses[i] / (num_agents - 1)) ** swarm.communication_adj
        expected_new_mass = max(masses[i] * multiplier, swarm.tolm * (1 / len(swarm)))

        assert torch.allclose(
            new_swarm.masses[i],
            torch.tensor(expected_new_mass),
        )
    expected_new_mass = torch.sum(masses) - torch.sum(new_swarm.masses[:-1])
    assert torch.allclose(new_swarm.masses[num_agents - 1], expected_new_mass)
    assert torch.all(new_swarm.masses >= 0)
